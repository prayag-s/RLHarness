"""FastAPI Application and WebSocket Telemetry Server."""

from __future__ import annotations
import asyncio
import math
import os
import tempfile
from pathlib import Path
from typing import Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rl_ide.contracts.adapter import EpisodeLog
from rl_ide.adapters.stream_adapter import CSVStreamAdapter
from rl_ide.adapters.dynamic_sim_adapter import DynamicSimAdapter
from rl_ide.adapters.auto_data_adapter import AutoDataAdapter
from rl_ide.compiler.models import TaskConfig, RewardDiffProposal
from rl_ide.compiler.nl_to_config import NLToConfigCompiler, ADAPTER_REGISTRY
from rl_ide.compiler.critique_engine import CritiqueEngine
from rl_ide.compiler.llm_client import LLMConfig, validate_connection, create_llm_client
from rl_ide.reward.dsl import RewardDSL
from rl_ide.state.store import TaskStateStore
from rl_ide.trainer.runner import TrainingRunner
from rl_ide.export.exporter import PolicyExporter
from rl_ide.export.registry import RunBundleRegistry, BundleMetadata, BundleDetails
from rl_ide.ingest.loader import load_file, IngestedDataset
from rl_ide.ingest.profiler import SchemaProfiler
from rl_ide.ingest.sql_connector import validate_read_only_sql, load_sql_dataset
from rl_ide.vibe.orchestrator import (
    VibeOrchestrator,
    VibeSessionResponse,
    VibeCoachResponse,
    compute_skill_summary,
)
from rl_ide.vibe.translator import dsl_to_plain_rules


class ValidateSettingsRequest(BaseModel):
    llm_config: LLMConfig


class CompileTaskRequest(BaseModel):
    prompt: str
    adapter_name: str = "CSVStreamAdapter"
    task_id: str | None = None
    llm_config: LLMConfig | None = None


class TrainBurstRequest(BaseModel):
    task_id: str = "default_task"
    n_episodes: int = 10


class CritiquePreviewRequest(BaseModel):
    task_id: str = "default_task"
    target_episode_id: int
    critique_text: str
    llm_config: LLMConfig | None = None


class CritiqueApplyRequest(BaseModel):
    task_id: str = "default_task"
    new_dsl: dict[str, Any]
    critique_text: str | None = None
    run_burst_after: bool = False
    burst_episodes: int = 10


class RollbackRequest(BaseModel):
    task_id: str = "default_task"
    target_version: int


class ExportRequest(BaseModel):
    task_id: str = "default_task"
    output_dir: str = "exported_bundles"


class ImportBundleRequest(BaseModel):
    bundle_path: str
    load_active: bool = False


class SQLIngestRequest(BaseModel):
    connection_uri: str
    query_or_table: str
    limit: int = 2000


class VibeCreateRequest(BaseModel):
    prompt: str
    dataset_id: str | None = None
    task_id: str | None = None
    llm_config: LLMConfig | None = None
    burst_episodes: int = 25


class VibeCoachRequest(BaseModel):
    session_id: str
    critique_text: str | None = None
    feedback_type: str | None = None  # "thumbs_up" | "thumbs_down" | "text" | "rule_boost" | "rule_reduce"
    target_episode_id: int | None = None
    burst_episodes: int = 20
    term_name: str | None = None
    new_importance: str | None = None
    llm_config: LLMConfig | None = None


class SimInteractRequest(BaseModel):
    session_id: str | None = None
    adapter_name: str | None = None
    board_state: list[int] | None = None
    human_cell_idx: int | None = None
    robot_state: dict[str, float] | None = None
    target_xy: tuple[float, float] | list[float] | None = None
    obstacles_list: list[dict[str, float]] | None = None
    n_frames: int = 4


class AppState:
    def __init__(self):
        self.state_store = TaskStateStore(base_dir="runs")
        self.compiler = NLToConfigCompiler()
        self.registry = RunBundleRegistry()
        self.vibe_orchestrator = VibeOrchestrator(compiler=self.compiler)
        self.dataset_registry: dict[str, IngestedDataset] = {}
        self.runner: TrainingRunner | None = None
        self.active_task_config: TaskConfig | None = None
        self.locked_episodes: set[int] = set()
        self.active_websockets: list[WebSocket] = []

    async def broadcast(self, message: dict[str, Any]) -> None:
        disconnected: list[WebSocket] = []
        for ws in self.active_websockets:
            try:
                await ws.send_json(message)
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
            if ws in self.active_websockets:
                self.active_websockets.remove(ws)


def create_app() -> FastAPI:
    app = FastAPI(title="RL-Native Agent IDE Backend", version="2.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    state = AppState()
    app.state.app_state = state

    @app.get("/api/adapters")
    def get_adapters():
        """Lists registered adapters, event schemas, observation and action specifications."""
        adapters_data = []
        for name, cls in ADAPTER_REGISTRY.items():
            instance = cls()
            adapters_data.append({
                "name": name,
                "event_schema": instance.event_schema(),
                "observations": instance.observations().model_dump(),
                "actions": instance.actions().model_dump(),
            })
        return {"adapters": adapters_data}

    @app.post("/api/settings/validate")
    def validate_llm_settings(req: ValidateSettingsRequest):
        """Validates BYOK provider connectivity and API credentials."""
        valid, message = validate_connection(req.llm_config)
        return {"valid": valid, "message": message, "provider": req.llm_config.provider}

    @app.post("/api/tasks/compile")
    def compile_task(req: CompileTaskRequest):
        """Compiles user prose into validated TaskConfig, warms up policy, and stores v1 snapshot."""
        task_id = req.task_id or f"task_{req.adapter_name.lower()}"
        adapter_cls = ADAPTER_REGISTRY.get(req.adapter_name)
        if not adapter_cls:
            raise HTTPException(status_code=400, detail=f"Unknown adapter {req.adapter_name}")

        adapter = adapter_cls()

        # Support user BYOK compiler client
        compiler = state.compiler
        if req.llm_config:
            compiler = NLToConfigCompiler(llm_client=create_llm_client(req.llm_config))

        task_config = compiler.compile(
            user_prompt=req.prompt,
            adapter_name=req.adapter_name,
            adapter=adapter,
            task_id=task_id,
        )

        # Initialize runner & execute warm-start pass
        runner = TrainingRunner(
            adapter=adapter,
            reward_dsl=task_config.reward_dsl,
            state_store=state.state_store,
            task_id=task_id,
            hyperparameters=task_config.hyperparameters,
        )
        warmstart_metrics = runner.warmstart_policy(epochs=12, n_samples=100)

        # Save v1
        v1 = state.state_store.save_version(
            task_id=task_id,
            task_config=task_config,
            reward_dsl=task_config.reward_dsl,
            applied_critique="Initial compilation from natural language",
        )

        state.runner = runner
        state.active_task_config = task_config

        return {
            "status": "compiled",
            "task_id": task_id,
            "version": v1.version,
            "version_tag": v1.version_tag,
            "task_config": task_config.model_dump(),
            "warmstart_metrics": warmstart_metrics,
        }

    @app.post("/api/train/burst")
    async def train_burst(req: TrainBurstRequest):
        """Runs a training micro-burst, streaming episode metrics and frames."""
        if not state.runner:
            raise HTTPException(status_code=400, detail="No active training task. Compile a task first.")

        loop = asyncio.get_event_loop()

        def _on_episode(ep_log: EpisodeLog):
            msg = {
                "type": "episode",
                "episode_id": ep_log.episode_id,
                "total_reward": ep_log.total_reward,
                "breakdown": ep_log.breakdown,
                "steps_count": len(ep_log.steps),
                "last_step": ep_log.steps[-1].events if ep_log.steps else {},
            }
            asyncio.run_coroutine_threadsafe(state.broadcast(msg), loop)

        episodes = state.runner.run_burst(
            n_episodes=req.n_episodes,
            on_episode_callback=_on_episode,
        )

        return {
            "status": "burst_completed",
            "episodes_count": len(episodes),
            "episodes": [ep.model_dump() for ep in episodes],
        }

    @app.post("/api/episodes/{ep_id}/lock")
    def toggle_lock_episode(ep_id: int):
        """Locks or unlocks a golden episode to protect against behavioral regressions."""
        if ep_id in state.locked_episodes:
            state.locked_episodes.remove(ep_id)
            is_locked = False
        else:
            state.locked_episodes.add(ep_id)
            is_locked = True
        return {
            "episode_id": ep_id,
            "locked": is_locked,
            "locked_episodes": sorted(list(state.locked_episodes)),
        }

    @app.post("/api/critique/preview")
    def preview_critique(req: CritiquePreviewRequest):
        """Evaluates critique and runs Stage 1 (counterfactual) and Stage 2 (speculative) previews."""
        if not state.runner:
            raise HTTPException(status_code=400, detail="No active runner.")

        target_ep = state.runner.buffer.get_by_id(req.target_episode_id)
        if not target_ep:
            recent = state.runner.buffer.get_recent(1)
            target_ep = recent[0] if recent else None

        if not target_ep:
            raise HTTPException(status_code=404, detail=f"Episode #{req.target_episode_id} not found in buffer")

        llm_client = create_llm_client(req.llm_config) if req.llm_config else None
        engine = CritiqueEngine(
            schema=state.runner.adapter.event_schema(),
            action_labels=state.runner.adapter.actions().labels,
            llm_client=llm_client,
        )

        proposal = engine.propose_diff(
            current_dsl=state.runner.reward_dsl,
            target_episode=target_ep,
            recent_episodes=state.runner.buffer.get_recent(20),
            locked_episodes=list(state.locked_episodes),
            critique_text=req.critique_text,
            policy=state.runner.model,
        )

        return proposal.model_dump()

    @app.post("/api/critique/apply")
    def apply_critique(req: CritiqueApplyRequest):
        """Applies validated diff, updates state store, and performs Critic Warmup."""
        if not state.runner:
            raise HTTPException(status_code=400, detail="No active runner.")

        new_dsl = RewardDSL.from_dict(req.new_dsl)
        version_snap = state.runner.apply_reward_diff(
            new_dsl=new_dsl,
            critique_text=req.critique_text,
        )

        burst_episodes: list[EpisodeLog] = []
        if req.run_burst_after:
            burst_episodes = state.runner.run_burst(n_episodes=req.burst_episodes)

        return {
            "status": "applied",
            "version": version_snap.version if version_snap else None,
            "version_tag": version_snap.version_tag if version_snap else "updated",
            "episodes_after": len(burst_episodes),
        }

    @app.post("/api/tasks/rollback")
    def rollback_task(req: RollbackRequest):
        """Reverts active task configuration and weights to a prior version."""
        restored = state.state_store.rollback(
            task_id=req.task_id,
            target_version=req.target_version,
        )

        # Re-initialize runner with restored configuration
        if state.runner:
            state.runner.reward_dsl = restored.reward_dsl
            state.runner.evaluator = state.runner.gym_env.evaluator = None
            from rl_ide.reward.evaluator import RewardEvaluator
            new_ev = RewardEvaluator(restored.reward_dsl, state.runner.adapter.event_schema())
            state.runner.evaluator = new_ev
            state.runner.gym_env.set_evaluator(new_ev)

        return {
            "status": "rolled_back",
            "restored_version": req.target_version,
            "new_version_tag": restored.version_tag,
        }

    @app.post("/api/export")
    def export_policy(req: ExportRequest):
        """Exports standalone ONNX bundle with parity check and drift baseline."""
        if not state.runner or not state.active_task_config:
            raise HTTPException(status_code=400, detail="No policy available for export.")

        bundle_dir, zip_path, max_diff = PolicyExporter.create_runbundle(
            task_config=state.active_task_config,
            reward_dsl=state.runner.reward_dsl,
            model=state.runner.model,
            episodes=state.runner.buffer.all(),
            output_dir=req.output_dir,
            bundle_name=req.task_id,
        )

        return {
            "status": "exported",
            "bundle_dir": str(bundle_dir),
            "zip_path": str(zip_path),
            "parity_max_diff": max_diff,
            "parity_passed": (max_diff < 1e-5),
        }

    # RunBundle Policy Registry Endpoints
    @app.get("/api/bundles")
    def list_bundles():
        """Lists all exported and registered runbundles available across teams."""
        bundles = state.registry.list_bundles()
        return {"bundles": [b.model_dump() for b in bundles]}

    @app.get("/api/bundles/{bundle_id}")
    def get_bundle_details(bundle_id: str):
        """Retrieves full configuration, ONNX details, and drift baseline for a specific bundle."""
        try:
            details = state.registry.get_bundle(bundle_id)
            return details.model_dump()
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"Bundle '{bundle_id}' not found.")

    @app.post("/api/bundles/import")
    def import_bundle(req: ImportBundleRequest):
        """Imports an external bundle archive into the registry and optionally activates it."""
        try:
            installed_dir, task_config = state.registry.import_bundle(req.bundle_path)
            
            if req.load_active:
                adapter_cls = ADAPTER_REGISTRY.get(task_config.adapter_name)
                if not adapter_cls:
                    raise HTTPException(status_code=400, detail=f"Unknown adapter '{task_config.adapter_name}' in bundle.")
                adapter = adapter_cls()
                runner = TrainingRunner(
                    adapter=adapter,
                    reward_dsl=task_config.reward_dsl,
                    state_store=state.state_store,
                    task_id=task_config.task_id,
                    hyperparameters=task_config.hyperparameters,
                )
                runner.warmstart_policy(epochs=5, n_samples=50)
                state.runner = runner
                state.active_task_config = task_config

            return {
                "status": "imported",
                "bundle_dir": str(installed_dir),
                "task_config": task_config.model_dump(),
                "is_active": req.load_active,
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Import failed: {str(e)}")

    @app.delete("/api/bundles/{bundle_id}")
    def delete_bundle(bundle_id: str):
        """Deletes a bundle from the registry."""
        success = state.registry.delete_bundle(bundle_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Bundle '{bundle_id}' not found.")
        return {"status": "deleted", "bundle_id": bundle_id}

    # ==========================================================================
    # Phase 3C: Universal Ingestion & Vibe Autopilot Endpoints
    # ==========================================================================

    @app.post("/api/ingest/upload")
    async def ingest_upload(
        file: UploadFile = File(...),
        role: str = Form("tabular"),
        target_col: str | None = Form(None),
    ):
        """Universal file uploader for .csv, .xlsx, .json, .sqlite, .pdf, .docx, .txt, .md."""
        filename = file.filename or "uploaded_dataset"
        contents = await file.read()

        suffix = os.path.splitext(filename)[1]
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(contents)
            tmp_path = Path(tmp.name)

        try:
            dataset = load_file(
                tmp_path,
                filename=filename,
                role=role,
                target_col=target_col,
            )
            state.dataset_registry[dataset.dataset_id] = dataset

            sample_records = dataset.df.head(10).to_dict(orient="records")
            for row in sample_records:
                for k, v in row.items():
                    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                        row[k] = None

            cols = list(dataset.df.columns)
            col_types = {col: str(dataset.df[col].dtype) for col in cols}
            if dataset.profile:
                for col_name, c_type in dataset.profile.column_types.items():
                    col_types[col_name] = c_type

            extracted_rules_summary = [r.model_dump() for r in dataset.extracted_rules]

            return {
                "dataset_id": dataset.dataset_id,
                "name": dataset.name,
                "format": dataset.format,
                "role": dataset.role,
                "total_rows": len(dataset.df),
                "columns": cols,
                "column_types": col_types,
                "sample_records": sample_records,
                "extracted_rules": extracted_rules_summary,
                "profile_summary": dataset.profile.model_dump() if dataset.profile else None,
            }
        finally:
            if tmp_path.exists():
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    @app.post("/api/ingest/sql")
    def ingest_sql(req: SQLIngestRequest):
        """Connects to live SQL DB, enforces read-only query guardrails, and loads rows."""
        try:
            if req.query_or_table.strip().upper().startswith(("SELECT", "WITH")) or " " in req.query_or_table.strip():
                validate_read_only_sql(req.query_or_table)
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))

        try:
            df = load_sql_dataset(req.connection_uri, req.query_or_table, limit=req.limit)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Database query failed: {str(e)}")

        profiler = SchemaProfiler()
        profile = profiler.profile(df)

        dataset = IngestedDataset(
            name=req.query_or_table if not req.query_or_table.strip().upper().startswith("SELECT") else "sql_query_result",
            source_type="sql",
            format="sql",
            df=df,
            profile=profile,
            role="tabular",
            metadata={"connection_uri": req.connection_uri, "query": req.query_or_table},
        )
        state.dataset_registry[dataset.dataset_id] = dataset

        sample_records = df.head(10).to_dict(orient="records")
        for row in sample_records:
            for k, v in row.items():
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    row[k] = None

        return {
            "dataset_id": dataset.dataset_id,
            "name": dataset.name,
            "format": dataset.format,
            "role": dataset.role,
            "total_rows": len(df),
            "columns": list(df.columns),
            "column_types": profile.column_types,
            "sample_records": sample_records,
            "extracted_rules": [],
            "profile_summary": profile.model_dump(),
        }

    @app.post("/api/vibe/create")
    def vibe_create(req: VibeCreateRequest):
        """One-click Vibe autopilot initialization."""
        dataset = state.dataset_registry.get(req.dataset_id) if req.dataset_id else None

        try:
            session, response = state.vibe_orchestrator.create_session(
                prompt=req.prompt,
                dataset_id=req.dataset_id,
                dataset=dataset,
                task_id=req.task_id,
                llm_config=req.llm_config,
                burst_episodes=req.burst_episodes,
            )
            state.runner = session.runner
            state.active_task_config = session.task_config

            return response.model_dump()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Vibe creation failed: {str(e)}")

    @app.post("/api/vibe/coach")
    def vibe_coach(req: VibeCoachRequest):
        """Conversational coaching or rule card priority adjustment."""
        try:
            response = state.vibe_orchestrator.coach_session(
                session_id=req.session_id,
                critique_text=req.critique_text,
                feedback_type=req.feedback_type,
                target_episode_id=req.target_episode_id,
                burst_episodes=req.burst_episodes,
                term_name=req.term_name,
                new_importance=req.new_importance,
                llm_config=req.llm_config,
            )
            session = state.vibe_orchestrator.sessions.get(req.session_id)
            if session:
                state.runner = session.runner
                state.active_task_config = session.task_config

            return response.model_dump()
        except KeyError as ke:
            raise HTTPException(status_code=404, detail=str(ke))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Coaching failed: {str(e)}")

    @app.post("/api/sim/interact")
    def sim_interact(req: SimInteractRequest):
        """Interactive live testing API for Tic-Tac-Toe moves and 2D Robot waypoint/obstacle dragging."""
        target_adapter = None
        target_policy = None

        if req.session_id and req.session_id in state.vibe_orchestrator.sessions:
            session = state.vibe_orchestrator.sessions[req.session_id]
            target_adapter = session.adapter
            target_policy = session.runner.model.policy if session.runner and session.runner.model else None
        elif state.runner:
            target_adapter = state.runner.adapter
            target_policy = state.runner.model.policy if state.runner.model else None
        elif req.adapter_name:
            adapter_cls = ADAPTER_REGISTRY.get(req.adapter_name)
            if adapter_cls:
                target_adapter = adapter_cls()

        if not target_adapter or not hasattr(target_adapter, "play_interactive_step"):
            target_adapter = DynamicSimAdapter(mode="tictactoe")

        target_coords = None
        if req.target_xy:
            target_coords = (float(req.target_xy[0]), float(req.target_xy[1]))

        try:
            res = target_adapter.play_interactive_step(
                policy=target_policy,
                board_state=req.board_state,
                human_cell_idx=req.human_cell_idx,
                robot_state=req.robot_state,
                target_xy=target_coords,
                obstacles_list=req.obstacles_list,
                n_frames=req.n_frames,
            )
            return res
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Simulation interaction failed: {str(e)}")

    @app.websocket("/ws/stream")
    async def websocket_stream(websocket: WebSocket):
        """Streams real-time episode frames and reward telemetry."""
        await websocket.accept()
        state.active_websockets.append(websocket)
        try:
            while True:
                data = await websocket.receive_text()
                if data == "ping":
                    await websocket.send_text("pong")
        except WebSocketDisconnect:
            if websocket in state.active_websockets:
                state.active_websockets.remove(websocket)

    return app


app = create_app()
