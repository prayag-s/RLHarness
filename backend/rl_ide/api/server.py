"""Clean v4.0 FastAPI Server for RL-Native Agent IDE.

Provides REST and WebSocket endpoints for Conversational Architect intake,
JIT code synthesis, live training streaming, sandboxed arena interaction,
mid-flight feedback nudges, and ONNX .runbundle packaging.
"""

from __future__ import annotations
import asyncio
from contextvars import ContextVar
import json
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from pathlib import Path
from typing import Any
import numpy as np
import torch
from stable_baselines3 import PPO
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field

from rl_ide.architect.models import AgentBlueprintSpec, IntakeResponse
from rl_ide.architect.intake import IntakeArchitectAgent
from rl_ide.architect.omni_agent import (
    StudioOmniAgent,
    StudioActionType,
    StudioChatRequest,
    StudioChatResponse,
)
from rl_ide.state.conversation_store import (
    ConversationStore,
    get_conversation_store,
    generate_title_from_prompt,
    utc_now_iso,
)
from rl_ide.autopilot.loop_controller import (
    AutonomousLoopController,
    SessionStateStore,
    SessionState,
    RoundSnapshot,
    TrainingInProgressError,
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
from rl_ide.synthesizer.env_builder import JITEnvironmentSynthesizer, load_synthesized_env_class, instantiate_env
from rl_ide.state.dataset_store import get_dataset_store
from rl_ide.state.records_store import load_records, save_records
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
    # Local studio only: arbitrary origins must not be able to act with user credentials.
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_owner_ctx: ContextVar[str] = ContextVar("rl_owner_id", default="local")


def current_owner() -> str:
    """Tenant key for this request.

    The studio has no identity layer yet (G4 in COMMERCIAL_READINESS_GAPS.md), so ownership comes
    from ``x-owner-id`` and defaults to the single local user. Everything below is already scoped by
    it, which means swapping in real authentication only changes this function.
    """
    try:
        return _owner_ctx.get() or "local"
    except LookupError:
        return "local"


# Global State
global_store = SessionStateStore()
global_controller = AutonomousLoopController(global_store)
global_registry = RunBundleRegistry()
dataset_store = get_dataset_store()
global_omni_agent = StudioOmniAgent(
    conv_store=get_conversation_store(),
    session_store=global_store,
    loop_controller=global_controller,
)

# Initialize LLM state (from persistent storage or environment variables)
init_llm_state()


def load_session_scorecard(session_id: str) -> dict[str, Any] | None:
    """Returns the persisted held-out scorecard for a session, if one has been produced."""
    session = global_store.get_session(session_id)
    if session is not None and session.scorecard:
        return session.scorecard
    path = global_store.base_dir / session_id / "scorecard.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


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


# Optional shared-secret gate (enabled only when RL_IDE_API_TOKEN is set).
_API_TOKEN = os.environ.get("RL_IDE_API_TOKEN", "").strip()

# Strong references to background tasks so they are not garbage-collected mid-flight.
_background_tasks: set[asyncio.Task] = set()


def _spawn_background(coro) -> None:
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


@app.middleware("http")
async def _optional_api_token(request, call_next):
    if _API_TOKEN and request.url.path.startswith("/api/"):
        supplied = request.headers.get("x-api-token", "")
        if supplied != _API_TOKEN:
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": "Invalid or missing API token."}, status_code=401)
    _owner_ctx.set((request.headers.get("x-owner-id") or "local").strip() or "local")
    return await call_next(request)


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
        dataset_store.save(dataset, owner_id=current_owner())

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
        dataset_store.save(dataset, owner_id=current_owner())

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
    conversation_id: str | None = None
    dataset_id: str | None = None
    user_answers: dict[str, Any] = Field(default_factory=dict)
    force_smart_defaults: bool = False
    conversation_history: list[dict[str, str]] = Field(default_factory=list)
    training_params: dict[str, Any] | None = None


@app.post("/api/v4/architect/chat", response_model=IntakeResponse)
def architect_chat(req: ArchitectChatRequest):
    """Conversational Intake interview evaluating user intent across the 5 Pillars."""
    require_verified_llm()
    dataset = dataset_store.get(req.dataset_id, current_owner()) if req.dataset_id else None
    architect = IntakeArchitectAgent()
    resp = architect.interview(
        user_prompt=req.user_prompt,
        dataset=dataset,
        user_answers=req.user_answers,
        force_smart_defaults=req.force_smart_defaults,
        conversation_history=req.conversation_history,
        training_params=req.training_params,
    )

    import uuid
    cid = req.conversation_id or f"conv_{uuid.uuid4().hex[:8]}"
    resp.conversation_id = cid

    conv_store = get_conversation_store()
    existing_conv = conv_store.get_conversation(cid)

    chat_history = list(existing_conv["chat_history"]) if existing_conv else []
    chat_history.append({
        "role": "user",
        "content": req.user_prompt,
        "timestamp": utc_now_iso(),
    })
    if req.user_answers:
        chat_history.append({
            "role": "user_answers",
            "answers": req.user_answers,
            "timestamp": utc_now_iso(),
        })

    title = existing_conv["title"] if existing_conv else generate_title_from_prompt(req.user_prompt)

    if resp.status == "needs_clarification":
        chat_history.append({
            "role": "assistant",
            "content": resp.assistant_message,
            "questions": [q.model_dump() for q in resp.questions],
            "timestamp": utc_now_iso(),
        })
        conv_store.save_or_update(
            conversation_id=cid,
            title=title,
            status="clarifying",
            chat_history=chat_history,
            dataset_id=req.dataset_id,
            training_params=req.training_params or resp.recommended_training_params,
        )
    else:
        chat_history.append({
            "role": "assistant",
            "content": resp.assistant_message,
            "blueprint": resp.blueprint_spec.model_dump() if resp.blueprint_spec else None,
            "timestamp": utc_now_iso(),
        })
        t_params = req.training_params or (
            {
                "max_auto_tune_rounds": resp.blueprint_spec.max_auto_tune_rounds,
                "episodes_per_round": resp.blueprint_spec.episodes_per_round,
                "target_metric_threshold": resp.blueprint_spec.target_metric_threshold,
            }
            if resp.blueprint_spec
            else resp.recommended_training_params
        )
        conv_store.save_or_update(
            conversation_id=cid,
            title=title,
            status="ready_to_build",
            chat_history=chat_history,
            dataset_id=req.dataset_id,
            blueprint_spec=resp.blueprint_spec.model_dump() if resp.blueprint_spec else None,
            training_params=t_params,
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

    dataset = dataset_store.get(req.dataset_id, current_owner()) if req.dataset_id else None

    try:
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

        # Reload verified canonical environment class
        from rl_ide.synthesizer.env_builder import load_synthesized_env_class, instantiate_env
        canonical_env_file = ws_dir / "env_module.py"
        if canonical_env_file.exists():
            env_cls = load_synthesized_env_class(canonical_env_file)

        # Synthesize Arena HTML
        dummy_env = instantiate_env(env_cls, dataset_records=records)
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

        # Update ConversationStore
        conv_store = get_conversation_store()
        t_params = {
            "max_auto_tune_rounds": req.blueprint.max_auto_tune_rounds,
            "episodes_per_round": req.blueprint.episodes_per_round,
            "target_metric_threshold": req.blueprint.target_metric_threshold,
        }
        conv_store.save_or_update(
            conversation_id=sid,
            title=req.blueprint.task_title or generate_title_from_prompt(req.blueprint.domain_summary),
            status="training",
            blueprint_spec=req.blueprint.model_dump(),
            dataset_id=req.dataset_id,
            env_code=healed_code,
            arena_html=html_str,
            training_params=t_params,
        )
    except HTTPException:
        raise
    except Exception as exc:
        import traceback
        traceback.print_exc()
        if ws_dir.exists() and not list(ws_dir.iterdir()):
            try:
                ws_dir.rmdir()
            except Exception:
                pass
        raise HTTPException(
            status_code=500,
            detail=f"Synthesis failed: {exc}",
        )

    # Event emitter helper
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.get_event_loop()

    def sync_event_emitter(event_data: dict[str, Any]):
        asyncio.run_coroutine_threadsafe(manager.broadcast(sid, event_data), loop)
        if event_data.get("event") in ("round_evaluated", "target_achieved", "budget_exhausted", "auto_tuned"):
            sess = global_store.get_session(sid)
            if sess:
                conv_store.save_or_update(
                    conversation_id=sid,
                    status=sess.status,
                    current_round=sess.current_round,
                    best_metric=sess.current_metric,
                    rounds_history=[r.model_dump() for r in sess.rounds_history],
                    episodes=sess.episodes,
                    env_code=sess.env_module_code,
                    latest_model_zip_path=sess.latest_model_zip_path,
                )

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
        conv_store.save_or_update(
            conversation_id=sid,
            status=final_session.status,
            current_round=final_session.current_round,
            best_metric=final_session.current_metric,
            rounds_history=[r.model_dump() for r in final_session.rounds_history],
            episodes=final_session.episodes,
            env_code=final_session.env_module_code,
            latest_model_zip_path=final_session.latest_model_zip_path,
        )
        return final_session
    else:
        # Launch background task in a worker thread so event loop is never blocked
        async def _run_background():
            await asyncio.sleep(0.3)  # Brief pause to ensure client WebSocket connects
            try:
                final_sess = await asyncio.to_thread(
                    global_controller.run_loop,
                    session_id=sid,
                    blueprint=req.blueprint,
                    env_cls=env_cls,
                    dataset=dataset,
                    on_event_callback=sync_event_emitter,
                    max_rounds=req.blueprint.max_auto_tune_rounds,
                )
                conv_store.save_or_update(
                    conversation_id=sid,
                    status=final_sess.status,
                    current_round=final_sess.current_round,
                    best_metric=final_sess.current_metric,
                    rounds_history=[r.model_dump() for r in final_sess.rounds_history],
                    episodes=final_sess.episodes,
                    env_code=final_sess.env_module_code,
                    latest_model_zip_path=final_sess.latest_model_zip_path,
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

        _spawn_background(_run_background())
        return {
            "session_id": sid,
            "status": "training",
            "message": "Autonomous JIT Synthesis & Closed-Loop Training initiated in background.",
        }


class CreateConversationRequest(BaseModel):
    title: str = "New Agent"
    dataset_id: str | None = None
    training_params: dict[str, Any] | None = None


class UpdateConversationParamsRequest(BaseModel):
    max_auto_tune_rounds: int | None = None
    episodes_per_round: int | None = None
    target_metric_threshold: float | None = None


@app.get("/api/v4/conversations")
def list_conversations():
    """Returns the caller's conversations sorted by updated_at DESC."""
    conv_store = get_conversation_store()
    return conv_store.list_conversations(owner_id=current_owner())


@app.post("/api/v4/conversations")
def create_conversation(req: CreateConversationRequest = CreateConversationRequest()):
    """Creates a new blank conversation and returns its conversation_id."""
    conv_store = get_conversation_store()
    return conv_store.create_conversation(
        title=req.title,
        dataset_id=req.dataset_id,
        training_params=req.training_params,
        owner_id=current_owner(),
    )


@app.get("/api/v4/conversations/{conversation_id}")
def get_conversation(conversation_id: str):
    """Loads full conversation state from disk/SQLite, re-instantiates SynthesizedGymEnv and reloads saved SB3 model."""
    conv_store = get_conversation_store()
    conv = conv_store.get_conversation(conversation_id, owner_id=current_owner())
    if not conv:
        raise HTTPException(status_code=404, detail=f"Conversation '{conversation_id}' not found.")

    # Re-instantiate SynthesizedGymEnv and reload saved SB3 model into global_store
    ws_dir = global_store.base_dir / conversation_id
    env_file = ws_dir / "env_module.py"
    if env_file.exists():
        try:
            from rl_ide.synthesizer.env_builder import load_synthesized_env_class, instantiate_env
            env_cls = load_synthesized_env_class(env_file)
            dataset = (
                dataset_store.get(conv.get("dataset_id"), conv.get("owner_id") or "local")
                if conv.get("dataset_id")
                else None
            )
            records = dataset.df.to_dict(orient="records") if (dataset and hasattr(dataset, "df")) else None
            if records is None:
                records = load_records(ws_dir)
            else:
                save_records(ws_dir, records)
            env = instantiate_env(env_cls, dataset_records=records)
            global_store.set_env(conversation_id, env)

            # Check if model zip or weights exist
            model_zip = conv.get("latest_model_zip_path") or (Path("runs") / conversation_id / "model.zip")
            if model_zip and Path(str(model_zip)).exists():
                try:
                    loaded_model = PPO.load(str(model_zip), env=env, device="cpu")
                    global_store.set_model(conversation_id, loaded_model)
                except Exception as m_err:
                    print(f"Warning: Failed to load PPO model from {model_zip}: {m_err}")
            if global_store.get_model(conversation_id) is None and conv.get("rounds_history"):
                latest_round = conv["rounds_history"][-1]
                weights_path = Path(latest_round.get("weights_path", ""))
                if weights_path.exists():
                    try:
                        temp_model = PPO("MlpPolicy", env, device="cpu", verbose=0)
                        state_dict = torch.load(str(weights_path), map_location="cpu", weights_only=True)
                        temp_model.policy.load_state_dict(state_dict)
                        global_store.set_model(conversation_id, temp_model)
                    except Exception as w_err:
                        print(f"Warning: Failed to load weights from {weights_path}: {w_err}")
        except Exception as e:
            print(f"Warning: Failed to re-instantiate environment for conversation {conversation_id}: {e}")

    # Reconstitute SessionState in global_store if not present
    existing_session = global_store.get_session(conversation_id)
    if not existing_session:
        try:
            bp_dict = conv.get("blueprint_spec")
            if bp_dict:
                bp = AgentBlueprintSpec.model_validate(bp_dict)
            else:
                bp = AgentBlueprintSpec(
                    task_title=conv.get("title") or "Synthesized Agent",
                    observation_features=["f0", "f1"],
                    action_labels=["Action 0", "Action 1"],
                    target_metric_name="Success Rate",
                    target_metric_threshold=85.0,
                    max_auto_tune_rounds=5,
                )
            t_params = conv.get("training_params", {}) or {}
            env_code_val = conv.get("env_code") or conv.get("env_module_code") or ""
            if not env_code_val and env_file.exists():
                try:
                    env_code_val = env_file.read_text(encoding="utf-8")
                except Exception:
                    env_code_val = ""
            env_code_val = env_code_val or ""
            arena_html_val = conv.get("arena_html") or ""

            session = SessionState(
                session_id=conversation_id,
                blueprint=bp,
                dataset_id=conv.get("dataset_id"),
                status=conv.get("status", "created"),
                current_round=conv.get("current_round", 1),
                max_rounds=t_params.get("max_auto_tune_rounds", bp.max_auto_tune_rounds or 5),
                current_metric=conv.get("best_metric", 0.0),
                target_threshold=t_params.get("target_metric_threshold", bp.target_metric_threshold or 90.0),
                rounds_history=[RoundSnapshot.model_validate(r) for r in conv.get("rounds_history", [])],
                episodes=conv.get("episodes", []),
                env_module_code=env_code_val,
                arena_html=arena_html_val,
                latest_model_zip_path=conv.get("latest_model_zip_path"),
                scorecard=load_session_scorecard(conversation_id),
            )
            global_store.set_session(session)
        except Exception as s_err:
            print(f"Warning: Failed to reconstruct SessionState: {s_err}")

    conv["scorecard"] = load_session_scorecard(conversation_id)
    return conv


@app.patch("/api/v4/conversations/{conversation_id}/params")
def update_conversation_params(conversation_id: str, req: UpdateConversationParamsRequest):
    """Updates max_auto_tune_rounds, episodes_per_round, and target_metric_threshold."""
    conv_store = get_conversation_store()
    params = {}
    if req.max_auto_tune_rounds is not None:
        params["max_auto_tune_rounds"] = req.max_auto_tune_rounds
    if req.episodes_per_round is not None:
        params["episodes_per_round"] = req.episodes_per_round
    if req.target_metric_threshold is not None:
        params["target_metric_threshold"] = req.target_metric_threshold

    try:
        updated = conv_store.update_params(conversation_id, params)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Conversation '{conversation_id}' not found.")

    sess = global_store.get_session(conversation_id)
    if sess:
        if req.max_auto_tune_rounds is not None:
            sess.max_rounds = req.max_auto_tune_rounds
        if req.target_metric_threshold is not None:
            sess.target_threshold = req.target_metric_threshold

    return updated


@app.delete("/api/v4/conversations")
def clear_all_conversations():
    """Deletes the caller's conversations, their workspaces, bundles and datasets."""
    conv_store = get_conversation_store()
    owner = current_owner()
    conversations = conv_store.list_conversations(owner_id=owner)
    for summary in conversations:
        cid = summary["conversation_id"]
        conv_store.delete(cid, owner_id=owner)
        global_store.delete_session(cid)
    for dataset in dataset_store.list(owner_id=owner):
        dataset_store.delete(dataset["dataset_id"], owner_id=owner)
    return {"success": True, "deleted_count": len(conversations)}


@app.delete("/api/v4/conversations/{conversation_id}")
def delete_conversation(conversation_id: str):
    """Deletes one of the caller's conversations and everything it owns."""
    conv_store = get_conversation_store()
    owner = current_owner()
    conv = conv_store.get_conversation(conversation_id, owner_id=owner)
    if not conv:
        raise HTTPException(status_code=404, detail=f"Conversation '{conversation_id}' not found.")
    conv_store.delete(conversation_id, owner_id=owner)
    global_store.delete_session(conversation_id)
    ds_id = conv.get("dataset_id")
    if ds_id:
        # Only drop the dataset when no other conversation of this owner still references it.
        still_used = any(
            c.get("dataset_id") == ds_id
            for c in conv_store.list_conversations(owner_id=owner)
            if c.get("conversation_id") != conversation_id
        )
        if not still_used:
            dataset_store.delete(ds_id, owner_id=owner)
    return {"success": True, "conversation_id": conversation_id}


@app.get("/api/v4/datasets")
def list_datasets():
    """Lists the caller's uploaded datasets (durable, unlike the previous in-memory dict)."""
    return dataset_store.list(owner_id=current_owner())


@app.delete("/api/v4/datasets/{dataset_id}")
def delete_dataset(dataset_id: str):
    """Deletes a dataset and its stored rows."""
    if not dataset_store.delete(dataset_id, owner_id=current_owner()):
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
    return {"success": True, "dataset_id": dataset_id}


class UnifiedConversationChatRequest(BaseModel):
    user_message: str | None = None
    message: str | None = None
    prompt: str | None = None
    user_prompt: str | None = None
    dataset_id: str | None = None
    selected_options: dict[str, Any] = Field(default_factory=dict)
    user_answers: dict[str, Any] = Field(default_factory=dict)
    training_params: dict[str, Any] | None = None
    force_action: StudioActionType | None = None
    run_sync: bool = False

    def get_effective_message(self) -> str:
        return self.user_message or self.message or self.prompt or self.user_prompt or ""

    def get_effective_answers(self) -> dict[str, Any]:
        return self.selected_options or self.user_answers or {}


class ArenaUpdateRequest(BaseModel):
    user_visual_request: str | None = None
    visual_prompt: str | None = None
    theme: str | None = None


class ArenaLockRequest(BaseModel):
    locked: bool = True


@app.post("/api/v4/conversations/{conversation_id}/chat", response_model=StudioChatResponse)
async def conversation_omni_chat(conversation_id: str, req: UnifiedConversationChatRequest):
    """Unified Codex-style omni-agent chat endpoint for continuous conversation at any stage."""
    require_verified_llm()
    msg = req.get_effective_message()
    if not msg.strip():
        raise HTTPException(status_code=400, detail="User message cannot be empty.")

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.get_event_loop()

    def sync_broadcast(event_data: dict[str, Any]):
        asyncio.run_coroutine_threadsafe(manager.broadcast(conversation_id, event_data), loop)

    conv_store = get_conversation_store()
    existing_conv = conv_store.get_conversation(conversation_id)
    effective_ds_id = req.dataset_id or (existing_conv.get("dataset_id") if existing_conv else None)
    active_dataset = (
        dataset_store.get(effective_ds_id, existing_conv.get("owner_id") if existing_conv else current_owner())
        if effective_ds_id
        else None
    )

    try:
        response = await asyncio.to_thread(
            global_omni_agent.process_message,
            conversation_id=conversation_id,
            user_message=msg,
            dataset=active_dataset,
            dataset_id=effective_ds_id,
            selected_options=req.get_effective_answers(),
            training_params=req.training_params,
            force_action=req.force_action,
            broadcast_callback=sync_broadcast,
            run_sync=req.run_sync,
        )
    except TrainingInProgressError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    # Launch asynchronous background training if training action was triggered
    if StudioActionType.RUN_OR_CONTINUE_TRAINING in response.action_types and not req.run_sync:
        conv = conv_store.get_conversation(conversation_id)
        if conv and conv.get("blueprint_spec"):
            bp = AgentBlueprintSpec.model_validate(conv["blueprint_spec"])
            ws_dir = global_store.base_dir / conversation_id
            env_file = ws_dir / "env_module.py"
            if env_file.exists():
                env_cls = load_synthesized_env_class(env_file)
                eps_per_burst = int(response.training_params.get("episodes_per_round") or bp.episodes_per_round or 200)
                add_rounds = int(response.training_params.get("max_auto_tune_rounds") or bp.max_auto_tune_rounds or 5)
                training_dataset = active_dataset or (
                    dataset_store.get(bp.dataset_id, current_owner()) if bp.dataset_id else None
                )

                prior_rounds = len(global_store.get_session(conversation_id).rounds_history) if global_store.get_session(conversation_id) else 0

                async def _run_training_background():
                    try:
                        trained_session = await asyncio.to_thread(
                            global_controller.run_loop,
                            session_id=conversation_id,
                            blueprint=bp,
                            env_cls=env_cls,
                            dataset=training_dataset,
                            on_event_callback=sync_broadcast,
                            max_rounds=add_rounds,
                            additional_rounds=add_rounds if prior_rounds else None,
                            episodes_per_burst=eps_per_burst,
                        )
                        c_store = get_conversation_store()
                        c_store.save_or_update(
                            conversation_id=conversation_id,
                            status=trained_session.status,
                            current_round=trained_session.current_round,
                            best_metric=trained_session.current_metric,
                            rounds_history=[r.model_dump() for r in trained_session.rounds_history],
                            episodes=trained_session.episodes,
                            latest_model_zip_path=trained_session.latest_model_zip_path,
                        )
                    except TrainingInProgressError as e:
                        sync_broadcast({"event": "training_conflict", "message": str(e)})
                    except Exception as e:
                        print(f"Error in continuous background training: {e}")
                        sync_broadcast({"event": "error", "message": str(e)})

                _spawn_background(_run_training_background())

    return response


@app.post("/api/v4/conversations/{conversation_id}/arena/update")
async def update_conversation_arena(conversation_id: str, req: ArenaUpdateRequest):
    """Mid-flight or post-training hot-reloading of visual arena HTML."""
    require_verified_llm()
    req_text = req.user_visual_request or req.visual_prompt or req.theme or "Update visual styling"
    conv_store = get_conversation_store()
    conv = conv_store.get_conversation(conversation_id)
    if not conv or not conv.get("blueprint_spec"):
        raise HTTPException(status_code=404, detail=f"Conversation '{conversation_id}' not found or has no blueprint.")

    blueprint = AgentBlueprintSpec.model_validate(conv["blueprint_spec"])
    current_html = conv.get("arena_html") or ""
    ws_dir = global_store.base_dir / conversation_id

    arena_builder = JITArenaSynthesizer()
    new_html = arena_builder.update_arena(
        blueprint=blueprint,
        current_arena_html=current_html,
        user_visual_request=req_text,
        workspace_dir=ws_dir,
    )

    conv_store.save_or_update(conversation_id, arena_html=new_html)
    sess = global_store.get_session(conversation_id)
    if sess:
        sess.arena_html = new_html

    await manager.broadcast(conversation_id, {"event": "arena_updated", "arena_html": new_html})
    return {"conversation_id": conversation_id, "arena_html": new_html, "status": "arena_updated"}


@app.post("/api/v4/conversations/{conversation_id}/arena/lock")
async def lock_conversation_arena(conversation_id: str, req: ArenaLockRequest = ArenaLockRequest()):
    """Locks or unlocks the Arena visual layout."""
    conv_store = get_conversation_store()
    try:
        updated = conv_store.lock_arena(conversation_id, locked=req.locked)
        return {"conversation_id": conversation_id, "arena_locked": updated.get("arena_locked", True)}
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Conversation '{conversation_id}' not found.")


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
    model_config = {"extra": "allow"}
    payload: dict[str, Any] = Field(default_factory=dict)

    def get_effective_payload(self) -> dict[str, Any]:
        if self.payload and len(self.payload) > 0:
            return self.payload
        extra = getattr(self, "__pydantic_extra__", None) or {}
        base = {k: v for k, v in self.__dict__.items() if k != "payload" and not k.startswith("_")}
        base.update(extra)
        return base or self.payload


@app.post("/api/v4/sessions/{session_id}/interact")
def session_interact(session_id: str, req: InteractRequest):
    """Executes live human play/inspection against the agent inside arena.html."""
    if global_controller.is_busy(session_id):
        raise HTTPException(
            status_code=409,
            detail="Training is currently mutating this session's policy; retry once the round completes.",
        )
    env = global_store.get_env(session_id)
    model = global_store.get_model(session_id)
    if not env:
        ws_dir = global_store.base_dir / session_id
        ws_env_file = ws_dir / "env_module.py"
        if not ws_env_file.exists():
            conv = get_conversation_store().get_conversation(session_id)
            if conv:
                if conv.get("env_code"):
                    ws_dir.mkdir(parents=True, exist_ok=True)
                    ws_env_file.write_text(conv["env_code"], encoding="utf-8")
                elif conv.get("blueprint_spec"):
                    bp = AgentBlueprintSpec.model_validate(conv["blueprint_spec"])
                    code_str, _, _ = global_omni_agent.env_synthesizer.synthesize(blueprint=bp, workspace_dir=ws_dir)
                    ws_dir.mkdir(parents=True, exist_ok=True)
                    ws_env_file.write_text(code_str, encoding="utf-8")

        if ws_env_file.exists():
            try:
                env_cls = load_synthesized_env_class(ws_env_file)
                env = instantiate_env(env_cls, dataset_records=load_records(ws_dir))
                global_store.set_env(session_id, env)
            except Exception as e:
                print(f"Warning: could not instantiate env from {ws_env_file}: {e}")
        if not env:
            raise HTTPException(status_code=404, detail=f"Active environment not found for session '{session_id}'.")

    if not model:
        conv = get_conversation_store().get_conversation(session_id)
        model_candidates = [
            conv.get("latest_model_zip_path") if conv else None,
            Path("runs") / session_id / "model.zip",
            global_store.base_dir / session_id / "model.zip",
        ]
        for mc in model_candidates:
            if mc and Path(str(mc)).exists():
                try:
                    from stable_baselines3 import PPO
                    model = PPO.load(str(mc), env=env, device="cpu")
                    global_store.set_model(session_id, model)
                    break
                except Exception as e:
                    print(f"Warning: could not load model from {mc}: {e}")

        if not model and conv and conv.get("rounds_history"):
            latest_round = conv["rounds_history"][-1]
            weights_path = Path(latest_round.get("weights_path", ""))
            if weights_path.exists():
                try:
                    from stable_baselines3 import PPO
                    temp_model = PPO("MlpPolicy", env, device="cpu", verbose=0)
                    state_dict = torch.load(str(weights_path), map_location="cpu", weights_only=True)
                    temp_model.policy.load_state_dict(state_dict)
                    model = temp_model
                    global_store.set_model(session_id, model)
                except Exception as w_err:
                    print(f"Warning: could not load weights from {weights_path}: {w_err}")

    # A changed environment must not be paired with an incompatible policy.
    if model is not None and (
        getattr(model, "observation_space", None) != env.observation_space
        or getattr(model, "action_space", None) != env.action_space
    ):
        global_store.delete_model(session_id)
        model = None

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

    try:
        return env.interactive_step(req.get_effective_payload(), policy_predict)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Interactive step failed: {exc}")


class NudgeRequest(BaseModel):
    feedback: str


@app.post("/api/v4/sessions/{session_id}/nudge")
def session_nudge(session_id: str, req: NudgeRequest):
    """Applies human feedback prose and runs an extra targeted tuning round."""
    require_verified_llm()
    try:
        tuning_res = global_controller.nudge_session(session_id, req.feedback)
        return tuning_res
    except TrainingInProgressError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
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

class ExportRequest(BaseModel):
    # Opt-in: bundles otherwise ship without any customer rows (G3).
    include_data: bool = False


@app.post("/api/v4/sessions/{session_id}/export")
def session_export(session_id: str, req: ExportRequest | None = None):
    """Exports SB3 policy to ONNX with parity check and packages standalone .runbundle."""
    session = global_store.get_session(session_id)
    model = global_store.get_model(session_id)
    env = global_store.get_env(session_id)

    if not session:
        conv = get_conversation_store().get_conversation(session_id)
        if conv and conv.get("blueprint_spec"):
            try:
                bp = AgentBlueprintSpec.model_validate(conv["blueprint_spec"])
                from rl_ide.autopilot.loop_controller import RoundSnapshot
                session = SessionState(
                    session_id=session_id,
                    blueprint=bp,
                    status=conv.get("status", "completed"),
                    current_round=int(conv.get("current_round", 1)),
                    current_metric=float(conv.get("best_metric", 0.0)),
                    rounds_history=[RoundSnapshot.model_validate(r) for r in conv.get("rounds_history", [])],
                    episodes=conv.get("episodes", []),
                    latest_model_zip_path=conv.get("latest_model_zip_path"),
                )
                global_store.set_session(session_id, session)
            except Exception as e:
                print(f"Warning: could not reconstruct session for export {session_id}: {e}")

    if not model:
        conv = get_conversation_store().get_conversation(session_id)
        model_candidates = [
            conv.get("latest_model_zip_path") if conv else None,
            Path("runs") / session_id / "model.zip",
            global_store.base_dir / session_id / "model.zip",
        ]
        for mc in model_candidates:
            if mc and Path(str(mc)).exists():
                try:
                    from stable_baselines3 import PPO
                    model = PPO.load(str(mc), device="cpu")
                    global_store.set_model(session_id, model)
                    break
                except Exception as e:
                    print(f"Warning: could not load model for export from {mc}: {e}")

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
        scorecard=load_session_scorecard(session_id),
        include_data=bool(req.include_data) if req else False,
    )

    exported_scorecard: dict[str, Any] | None = None
    scorecard_file = Path(bundle_dir) / "scorecard.json"
    if scorecard_file.exists():
        try:
            exported_scorecard = json.loads(scorecard_file.read_text(encoding="utf-8"))
        except Exception:
            exported_scorecard = None

    return {
        "session_id": session_id,
        "bundle_dir": str(bundle_dir),
        "bundle_path": str(bundle_dir),
        "zip_path": str(zip_path),
        "onnx_parity_diff": max_diff,
        "scorecard": exported_scorecard,
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
