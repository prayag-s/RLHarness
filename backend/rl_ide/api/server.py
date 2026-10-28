"""Clean v4.0 FastAPI Server for RL-Native Agent IDE.

Provides REST and WebSocket endpoints for Conversational Architect intake,
JIT code synthesis, live training streaming, sandboxed arena interaction,
mid-flight feedback nudges, and ONNX .runbundle packaging.
"""

from __future__ import annotations
import asyncio
import json
import os
from pathlib import Path
from typing import Any
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field

from rl_ide.architect.models import AgentBlueprintSpec, IntakeResponse
from rl_ide.architect.intake import IntakeArchitectAgent
from rl_ide.autopilot.loop_controller import (
    AutonomousLoopController,
    SessionStateStore,
    SessionState,
    RoundSnapshot,
)
from rl_ide.autopilot.auto_tuner import TuningResult
from rl_ide.export.exporter import PolicyExporter
from rl_ide.export.registry import RunBundleRegistry, BundleMetadata, BundleDetails
from rl_ide.ingest.loader import IngestedDataset, load_bytes
from rl_ide.ingest.sql_connector import test_sql_connection, load_sql_dataset
from rl_ide.llm.client import (
    LLMConfig,
    validate_llm_connection,
    create_llm_client,
    get_llm_status,
    verify_and_set_llm_config,
    is_llm_ready,
    init_llm_state,
)
from rl_ide.synthesizer.env_builder import JITEnvironmentSynthesizer, load_synthesized_env_class
from rl_ide.synthesizer.verifier import SandboxSmokeTester
from rl_ide.synthesizer.arena_builder import JITArenaSynthesizer

app = FastAPI(
    title="RL-Native Agent Studio v4.0 API",
    description="Autonomous JIT Agent Synthesis, Conversational Architect & Closed-Loop Training",
    version="4.0.0",
)

def create_app() -> FastAPI:
    """Returns the FastAPI application instance."""
    return app

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r".*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State
global_store = SessionStateStore()
global_controller = AutonomousLoopController(global_store)
global_registry = RunBundleRegistry()
ingested_datasets_store: dict[str, IngestedDataset] = {}

# Initialize LLM state (from persistent storage or environment variables)
init_llm_state()


def require_verified_llm() -> None:
    """Enforces that a working, verified LLM key is active before allowing task execution."""
    if not is_llm_ready():
        raise HTTPException(
            status_code=403,
            detail=(
                "A verified LLM API key is mandatory before using the studio. "
                "Please configure and test your LLM API credentials in Settings."
            ),
        )


# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, list[WebSocket]] = {}

    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        if session_id not in self.active_connections:
            self.active_connections[session_id] = []
        self.active_connections[session_id].append(websocket)

    def disconnect(self, session_id: str, websocket: WebSocket):
        if session_id in self.active_connections:
            if websocket in self.active_connections[session_id]:
                self.active_connections[session_id].remove(websocket)

    async def broadcast(self, session_id: str, message: dict[str, Any]):
        if session_id in self.active_connections:
            dead_sockets = []
            for connection in self.active_connections[session_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    dead_sockets.append(connection)
            for dead in dead_sockets:
                self.disconnect(session_id, dead)


manager = ConnectionManager()


# -----------------------------------------------------------------------------
# 1. Ingestion Endpoints
# -----------------------------------------------------------------------------

@app.post("/api/v4/ingest/upload")
async def upload_file(
    file: UploadFile = File(...),
    role: str = "tabular",
):
    """Universal file ingestion (.csv, .xlsx, .pdf, .docx, .json, .txt)."""
    try:
        content = await file.read()
        dataset = load_bytes(
            filename=file.filename or "uploaded_data.csv",
            content=content,
            role=role,
        )
        ingested_datasets_store[dataset.dataset_id] = dataset

        return {
            "dataset_id": dataset.dataset_id,
            "name": dataset.name,
            "format": dataset.format,
            "source_type": dataset.source_type,
            "role": dataset.role,
            "columns": list(dataset.df.columns) if hasattr(dataset, "df") and dataset.df is not None else [],
            "num_rows": len(dataset.df) if hasattr(dataset, "df") and dataset.df is not None else 0,
            "extracted_rules_count": len(dataset.extracted_rules),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


class SQLIngestRequest(BaseModel):
    connection_uri: str
    query: str
    name: str = "sql_dataset"


@app.post("/api/v4/ingest/sql")
def ingest_sql(req: SQLIngestRequest):
    """Ingests read-only query results from live SQLite / PostgreSQL / MySQL."""
    try:
        dataset = load_sql_dataset(
            connection_uri=req.connection_uri,
            query=req.query,
            name=req.name,
        )
        ingested_datasets_store[dataset.dataset_id] = dataset

        return {
            "dataset_id": dataset.dataset_id,
            "name": dataset.name,
            "format": dataset.format,
            "columns": list(dataset.df.columns),
            "num_rows": len(dataset.df),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# -----------------------------------------------------------------------------
# 2. Conversational Intake Architect Endpoints
# -----------------------------------------------------------------------------

class ArchitectChatRequest(BaseModel):
    user_prompt: str
    dataset_id: str | None = None
    user_answers: dict[str, str] = Field(default_factory=dict)
    force_smart_defaults: bool = False
    conversation_history: list[dict[str, str]] = Field(default_factory=list)


@app.post("/api/v4/architect/chat", response_model=IntakeResponse)
def architect_chat(req: ArchitectChatRequest):
    """Conversational Intake interview evaluating user intent across the 5 Pillars."""
    require_verified_llm()
    dataset = ingested_datasets_store.get(req.dataset_id) if req.dataset_id else None
    architect = IntakeArchitectAgent()
    resp = architect.interview(
        user_prompt=req.user_prompt,
        dataset=dataset,
        user_answers=req.user_answers,
        force_smart_defaults=req.force_smart_defaults,
        conversation_history=req.conversation_history,
    )
    return resp


# -----------------------------------------------------------------------------
# 3. Session Synthesis & Autonomous Loop Endpoints
# -----------------------------------------------------------------------------

class SessionStartRequest(BaseModel):
    blueprint: AgentBlueprintSpec
    session_id: str | None = None
    dataset_id: str | None = None
    run_sync: bool = False


@app.post("/api/v4/sessions/start")
async def start_session(req: SessionStartRequest):
    """Synthesizes env_module.py + arena.html, verifies via smoke tester, and runs autonomous loop."""
    require_verified_llm()
    import uuid
    sid = req.session_id or f"sess_{uuid.uuid4().hex[:8]}"
    ws_dir = global_store.base_dir / sid
    ws_dir.mkdir(parents=True, exist_ok=True)

    dataset = ingested_datasets_store.get(req.dataset_id) if req.dataset_id else None

    # 1. Synthesize Environment & Arena
    env_builder = JITEnvironmentSynthesizer()
    code_str, env_path, env_cls = env_builder.synthesize(
        blueprint=req.blueprint,
        dataset=dataset,
        workspace_dir=ws_dir,
    )

    verifier = SandboxSmokeTester()
    records = dataset.df.to_dict(orient="records") if (dataset and hasattr(dataset, "df")) else None
    verified, healed_code, logs = verifier.verify_and_heal(
        code_str=code_str,
        blueprint=req.blueprint,
        dataset=dataset,
        workspace_dir=ws_dir,
    )

    if not verified:
        raise HTTPException(status_code=500, detail="Environment code failed sandbox verification checks.")

    # Synthesize Arena HTML
    dummy_env = env_cls(dataset_records=records)
    dummy_obs, dummy_info = dummy_env.reset()
    arena_builder = JITArenaSynthesizer()
    html_str, arena_path = arena_builder.synthesize(
        blueprint=req.blueprint,
        sample_arena_state=dummy_info.get("arena_state", {}),
        workspace_dir=ws_dir,
    )

    # Initialize Session in store
    session = SessionState(
        session_id=sid,
        blueprint=req.blueprint,
        dataset_id=req.dataset_id,
        status="synthesized",
        max_rounds=req.blueprint.max_auto_tune_rounds or 4,
        target_threshold=req.blueprint.target_metric_threshold,
        env_module_code=healed_code,
        arena_html=html_str,
    )
    global_store.set_session(session)

    # Event emitter helper
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.get_event_loop()

    def sync_event_emitter(event_data: dict[str, Any]):
        asyncio.run_coroutine_threadsafe(manager.broadcast(sid, event_data), loop)

    if req.run_sync:
        # Run synchronously for test suites
        final_session = global_controller.run_loop(
            session_id=sid,
            blueprint=req.blueprint,
            env_cls=env_cls,
            dataset=dataset,
            on_event_callback=sync_event_emitter,
            max_rounds=req.blueprint.max_auto_tune_rounds,
        )
        return final_session
    else:
        # Launch background task in a worker thread so event loop is never blocked
        async def _run_background():
            await asyncio.sleep(0.3)  # Brief pause to ensure client WebSocket connects
            try:
                await asyncio.to_thread(
                    global_controller.run_loop,
                    session_id=sid,
                    blueprint=req.blueprint,
                    env_cls=env_cls,
                    dataset=dataset,
                    on_event_callback=sync_event_emitter,
                    max_rounds=req.blueprint.max_auto_tune_rounds,
                )
            except Exception as loop_err:
                print(f"[server._run_background] Session {sid} loop error: {loop_err}")
                sess = global_store.get_session(sid)
                if sess:
                    sess.status = "error"
                sync_event_emitter({
                    "event": "training_error",
                    "session_id": sid,
                    "message": f"Training loop error: {loop_err}",
                })

        asyncio.create_task(_run_background())
        return {
            "session_id": sid,
            "status": "training",
            "message": "Autonomous JIT Synthesis & Closed-Loop Training initiated in background.",
        }


@app.get("/api/v4/sessions/{session_id}")
def get_session(session_id: str):
    """Retrieves current session state, round history, metrics, and synthesized code."""
    session = global_store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return session


@app.get("/api/v4/sessions/{session_id}/arena", response_class=HTMLResponse)
def get_session_arena(session_id: str):
    """Serves the raw synthesized arena.html document for the sandboxed iframe."""
    session = global_store.get_session(session_id)
    if not session or not session.arena_html:
        ws_arena = global_store.base_dir / session_id / "arena.html"
        if ws_arena.exists():
            return HTMLResponse(content=ws_arena.read_text(encoding="utf-8"))
        raise HTTPException(status_code=404, detail="Arena HTML not found.")
    return HTMLResponse(content=session.arena_html)


class InteractRequest(BaseModel):
    payload: dict[str, Any]


@app.post("/api/v4/sessions/{session_id}/interact")
def session_interact(session_id: str, req: InteractRequest):
    """Executes live human play/inspection against the agent inside arena.html."""
    env = global_store.get_env(session_id)
    model = global_store.get_model(session_id)
    if not env:
        raise HTTPException(status_code=404, detail=f"Active environment not found for session '{session_id}'.")

    def policy_predict(obs):
        if model:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                features = model.policy.extract_features(obs_t)
                latent_pi, _ = model.policy.mlp_extractor(features)
                logits = model.policy.action_net(latent_pi)
                probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()
                action = int(np.argmax(probs))
                return action, probs.tolist()
        else:
            n_acts = env.action_space.n
            return 0, [1.0 / n_acts] * n_acts

    res = env.interactive_step(req.payload, policy_predict)
    return res


class NudgeRequest(BaseModel):
    feedback: str


@app.post("/api/v4/sessions/{session_id}/nudge")
def session_nudge(session_id: str, req: NudgeRequest):
    """Applies human feedback prose and runs an extra targeted tuning round."""
    require_verified_llm()
    try:
        tuning_res = global_controller.nudge_session(session_id, req.feedback)
        return tuning_res
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


class RollbackRequest(BaseModel):
    round_num: int


@app.post("/api/v4/sessions/{session_id}/rollback")
def session_rollback(session_id: str, req: RollbackRequest):
    """Rolls back model weights and env_module.py to a previous round snapshot."""
    try:
        snapshot = global_store.rollback_to_round(session_id, req.round_num)
        return {
            "session_id": session_id,
            "status": "rolled_back",
            "restored_round": snapshot.round_num,
            "restored_metric": snapshot.metric,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# -----------------------------------------------------------------------------
# 4. Export & Registry Endpoints
# -----------------------------------------------------------------------------

@app.post("/api/v4/sessions/{session_id}/export")
def session_export(session_id: str):
    """Exports SB3 policy to ONNX with parity check and packages standalone .runbundle."""
    session = global_store.get_session(session_id)
    model = global_store.get_model(session_id)
    env = global_store.get_env(session_id)

    if not session or not model:
        raise HTTPException(status_code=404, detail="Session or trained policy model not found.")

    ws_dir = global_store.base_dir / session_id
    env_path = ws_dir / "env_module.py"
    arena_path = ws_dir / "arena.html"

    bundle_dir, zip_path, max_diff = PolicyExporter.package_runbundle(
        session_id=session_id,
        blueprint=session.blueprint,
        model=model,
        env_module_path=env_path,
        arena_html_path=arena_path,
    )

    return {
        "session_id": session_id,
        "bundle_dir": str(bundle_dir),
        "zip_path": str(zip_path),
        "onnx_parity_diff": max_diff,
        "download_url": f"/api/v4/sessions/{session_id}/download",
    }


@app.get("/api/v4/sessions/{session_id}/download")
def download_runbundle(session_id: str):
    """Downloads the packaged .runbundle.zip archive."""
    import re
    safe_sid = re.sub(r"[^a-zA-Z0-9_\-]", "", session_id)
    zip_path = Path("exported_bundles") / f"{safe_sid}.runbundle.zip"
    if not zip_path.exists():
        raise HTTPException(status_code=404, detail="Runbundle zip archive not found. Call export first.")
    return FileResponse(
        path=str(zip_path),
        filename=f"{safe_sid}.runbundle.zip",
        media_type="application/zip",
    )


@app.get("/api/v4/bundles", response_model=list[BundleMetadata])
def list_bundles():
    """Lists cataloged exported runbundles."""
    return global_registry.list_bundles()


# -----------------------------------------------------------------------------
# 5. Settings Validation, LLM Activation & WebSocket Telemetry
# -----------------------------------------------------------------------------

@app.get("/api/v4/settings/llm")
def get_llm_settings_endpoint():
    """Returns current active LLM provider, verification state, and connectivity metrics."""
    return get_llm_status()


@app.post("/api/v4/settings/llm")
def configure_llm_settings(config: LLMConfig):
    """Tests and activates BYOK LLM credentials globally across all agents and synthesizers."""
    success, msg, result = verify_and_set_llm_config(config, save_to_disk=True)
    return result


@app.post("/api/v4/settings/validate")
def validate_settings(config: LLMConfig):
    """Tests candidate credentials connectivity without altering global active client."""
    res = validate_llm_connection(config)
    res["valid"] = res.get("success", False)
    return res


@app.websocket("/ws/v4/{session_id}")
async def websocket_telemetry(websocket: WebSocket, session_id: str):
    """Streams live training frames, round metrics, and auto-tuner events."""
    await manager.connect(session_id, websocket)
    try:
        while True:
            # Keep alive and listen for client commands
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                if msg.get("action") == "ping":
                    await websocket.send_json({"event": "pong", "session_id": session_id})
            except Exception:
                pass
    except WebSocketDisconnect:
        manager.disconnect(session_id, websocket)
