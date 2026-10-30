"""Worker executed inside the sandbox. Never imported by the API process.

Modes:
  smoke  - import a generated environment module and run the runtime smoke test
  train  - run the autonomous PPO loop against a generated environment

Everything it needs is staged into the case directory; everything it produces is written back there
(or reported as RESULT), so the API never has to share memory or a working directory with it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

EVENT_PREFIX = "EVENT "


def _emit(event: dict[str, Any]) -> None:
    try:
        sys.stdout.write(EVENT_PREFIX + json.dumps(event, default=str) + "\n")
        sys.stdout.flush()
    except Exception:
        pass


def _load_records(path: str | None) -> list[dict[str, Any]] | None:
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else None


def _write_result(out_path: str, payload: dict[str, Any]) -> None:
    Path(out_path).write_text(json.dumps(payload, default=str), encoding="utf-8")


def run_smoke(job: dict[str, Any]) -> dict[str, Any]:
    from rl_ide.synthesizer.env_builder import load_synthesized_env_class
    from rl_ide.synthesizer.verifier import SandboxSmokeTester

    env_cls = load_synthesized_env_class(job["env_module"])
    records = _load_records(job.get("records_path"))
    tester = SandboxSmokeTester(llm_client=None)
    res = tester.run_smoke_test(
        env_cls,
        dataset_records=records,
        expected_state_keys=job.get("expected_state_keys") or None,
        expected_obs_dim=job.get("expected_obs_dim") or None,
        expected_n_actions=job.get("expected_n_actions") or None,
    )
    return {"smoke": res, "warnings": list(tester.last_warnings)}


def run_observations(job: dict[str, Any]) -> dict[str, Any]:
    """Replays the environment and returns real observation vectors.

    Used by export for ONNX parity and the drift baseline so the API process never has to
    instantiate the generated module.
    """
    from rl_ide.synthesizer.env_builder import instantiate_env, load_synthesized_env_class

    env_cls = load_synthesized_env_class(job["env_module"])
    env = instantiate_env(env_cls, dataset_records=_load_records(job.get("records_path")))
    episodes = int(job.get("episodes") or 5)
    steps = int(job.get("steps") or 10)
    samples: list[list[float]] = []
    for _ in range(episodes):
        obs, _info = env.reset()
        samples.append([float(x) for x in obs])
        for _step in range(steps):
            obs, _reward, terminated, truncated, _info = env.step(env.action_space.sample())
            samples.append([float(x) for x in obs])
            if terminated or truncated:
                break
    return {"observations": samples, "obs_dim": int(getattr(env, "obs_dim", len(samples[0]) if samples else 0))}


def run_parity(job: dict[str, Any]) -> dict[str, Any]:
    """Compares the exported ONNX graph with the SB3 policy over whole episodes, in the sandbox."""
    from stable_baselines3 import PPO

    from rl_ide.export.exporter import PolicyExporter
    from rl_ide.synthesizer.env_builder import instantiate_env, load_synthesized_env_class

    env_cls = load_synthesized_env_class(job["env_module"])
    env = instantiate_env(env_cls, dataset_records=_load_records(job.get("records_path")))
    model = PPO.load(job["model_path"], env=env, device="cpu")
    return PolicyExporter.verify_behavioural_parity(
        model=model,
        onnx_path=job["onnx_path"],
        env=env,
        n_episodes=int(job.get("episodes") or 10),
    )


def run_train(job: dict[str, Any]) -> dict[str, Any]:
    import pandas as pd

    from rl_ide.architect.models import AgentBlueprintSpec
    from rl_ide.autopilot.loop_controller import AutonomousLoopController, SessionStateStore
    from rl_ide.ingest.loader import IngestedDataset
    from rl_ide.synthesizer.env_builder import load_synthesized_env_class

    blueprint = AgentBlueprintSpec.model_validate(job["blueprint"])
    records = _load_records(job.get("records_path"))
    env_cls = load_synthesized_env_class(job["env_module"])

    dataset = None
    if records is not None:
        dataset = IngestedDataset(
            name="staged_dataset",
            source_type="file",
            format="json",
            df=pd.DataFrame(records),
            **({"dataset_id": blueprint.dataset_id} if blueprint.dataset_id else {}),
        )

    store = SessionStateStore(base_dir=job["base_dir"])
    controller = AutonomousLoopController(store)
    session = controller.run_loop(
        session_id=job["session_id"],
        blueprint=blueprint,
        env_cls=env_cls,
        dataset=dataset,
        on_event_callback=_emit,
        max_rounds=job.get("max_rounds"),
        episodes_per_burst=int(job.get("episodes_per_burst") or 25),
        additional_rounds=job.get("additional_rounds"),
    )

    summary = session.model_dump()
    # The API already holds the code and the arena; shipping them back would double the payload.
    summary.pop("env_module_code", None)
    summary.pop("arena_html", None)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sandboxed environment worker")
    parser.add_argument("--mode", required=True, choices=["smoke", "train", "observations", "parity"])
    parser.add_argument("--job", required=True)
    args = parser.parse_args(argv)

    job = json.loads(Path(args.job).read_text(encoding="utf-8"))
    out_path = job["out_path"]
    try:
        if args.mode == "smoke":
            result = run_smoke(job)
        elif args.mode == "train":
            result = run_train(job)
        elif args.mode == "parity":
            result = run_parity(job)
        else:
            result = run_observations(job)
        _write_result(out_path, {"ok": True, "result": result})
        return 0
    except Exception as exc:  # reported to the parent, never swallowed
        import traceback

        _write_result(
            out_path,
            {"ok": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()},
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
