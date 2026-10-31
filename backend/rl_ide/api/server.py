"""Clean v4.0 FastAPI Server for RL-Native Agent IDE.

Provides REST and WebSocket endpoints for Conversational Architect intake,
JIT code synthesis, live training streaming, sandboxed arena interaction,
mid-flight feedback nudges, and ONNX .runbundle packaging.
"""

from __future__ import annotations

import logging
import asyncio
import atexit
import hmac
from contextvars import ContextVar
import threading
import time
import json
import os
import shutil
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from pathlib import Path
from typing import Any
import numpy as np
import torch
from stable_baselines3 import PPO
from fastapi import FastAPI, File, Request, UploadFile, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
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
from rl_ide.autopilot.crossfire_protocol import CrossfireProtocolStore
from rl_ide.autopilot.eval_corpus import EvalCorpusStore, GoldenEvalCase
from rl_ide.ingest.interaction_log import detect_interaction_log, harvest_into_corpus
from rl_ide.export.exporter import PolicyExporter
from rl_ide.export.registry import RunBundleRegistry, BundleMetadata, BundleDetails
from rl_ide.ingest.loader import IngestedDataset, load_bytes
from rl_ide.ingest.sql_connector import test_sql_connection, load_sql_dataset
from rl_ide.llm import metering
from rl_ide.llm.client import (
    LLMConfig,
    validate_llm_connection,
    create_llm_client,
    get_llm_status,
    verify_and_set_llm_config,
    is_llm_ready,
    init_llm_state,
    is_test_mode,
)
from rl_ide.synthesizer.env_builder import JITEnvironmentSynthesizer, load_synthesized_env_class, instantiate_env
from rl_ide.security import enforce_isolation_requirement, isolation_available
from rl_ide.state.auth_store import get_token_store
from rl_ide.state.dataset_store import get_dataset_store
from rl_ide.state.job_store import (
    CANCELLED,
    FAILED,
    QUEUED,
    RUNNING,
    SUCCEEDED,
    JobCancelled,
    get_job_store,
)
from rl_ide.state.records_store import load_records, save_records
from rl_ide.synthesizer.verifier import SandboxSmokeTester
from rl_ide.synthesizer.arena_builder import JITArenaSynthesizer

# Marker the UI switches on when a model-backed route is called without a verified provider.
LLM_KEY_REQUIRED = "llm-key-required"

logger = logging.getLogger(__name__)

from contextlib import asynccontextmanager


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Start background work that must outlive a request but not a process."""
    _start_retention()
    yield


app = FastAPI(
    lifespan=_lifespan,
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

# A deployment that requires isolation must not start on a host that cannot provide it.
enforce_isolation_requirement()


def mount_frontend(dist_dir: str | Path | None = None) -> bool:
    """Serves the built SPA from the API origin when one is shipped next to it (container image).

    Mounted last so every `/api` and `/ws` route keeps precedence; a dev setup without a build
    simply keeps the Vite server.
    """
    from fastapi.staticfiles import StaticFiles

    candidate = Path(dist_dir or os.environ.get("RL_IDE_FRONTEND_DIST") or "frontend_dist")
    if not (candidate / "index.html").exists():
        return False
    app.mount("/", StaticFiles(directory=str(candidate), html=True), name="ui")
    return True


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
    """Blocks the model-backed actions until a working provider is verified.

    Only synthesis-time work needs this. Uploading a dataset, listing conversations, reading an
    existing arena and the deterministic preview are all local and stay available, so a prospective
    user can evaluate the studio before they have a key (G13).
    """
    if not is_llm_ready():
        raise HTTPException(
            status_code=403,
            detail=(
                "This step needs a model. Connect and verify an LLM provider in Settings to "
                "synthesize environments, chat with the architect or train. Uploading data, browsing "
                "existing agents and the arena preview work without a key."
            ),
            headers={"X-RL-Reason": LLM_KEY_REQUIRED},
        )


def bind_usage_context(conversation_id: str | None = None) -> None:
    """Attributes LLM spend made while answering this request to a conversation and tenant."""
    metering.current_conversation_id.set(conversation_id)
    metering.current_owner_id.set(current_owner())


# Deployment (not tenant) secret: may mint and revoke tenant tokens. Never scopes data itself.
_ADMIN_TOKEN = (os.environ.get("RL_IDE_ADMIN_TOKEN") or os.environ.get("RL_IDE_API_TOKEN") or "").strip()
_ADMIN_OWNER = "admin"


def _auth_required() -> bool:
    """True when this deployment must authenticate callers (G4).

    An explicit `RL_IDE_REQUIRE_AUTH=1` is honoured immediately; otherwise the presence of issued
    tenant tokens is itself a signal that the deployment is multi-tenant, so silently accepting an
    owner header would be a downgrade. Callers are re-read on every request because a token may be
    issued while the server runs.
    """
    if _AUTH_REQUIRED_ENV:
        return True
    # A test run must be hermetic: tokens issued by a developer's live instance must not make the
    # suite demand credentials.
    if is_test_mode():
        return False
    try:
        return get_token_store().count() > 0
    except Exception:
        return False


_AUTH_REQUIRED_ENV = (os.environ.get("RL_IDE_REQUIRE_AUTH", "") or "").strip().lower() in ("1", "true", "yes")

# Strong references to background tasks so they are not garbage-collected mid-flight.
_background_tasks: set[asyncio.Task] = set()


def _spawn_background(coro) -> None:
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


def _bearer_token(request) -> str:
    header = request.headers.get("authorization") or ""
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    # Retained for existing scripts; a bearer token is the documented form.
    return (request.headers.get("x-api-token") or "").strip()


def _is_admin(token: str) -> bool:
    return bool(_ADMIN_TOKEN) and hmac.compare_digest(token.encode(), _ADMIN_TOKEN.encode())


@app.middleware("http")
async def _authenticate(request, call_next):
    """Resolves the tenant for every API request.

    Two modes, decided once at startup:

    * **Hosted** (`RL_IDE_REQUIRE_AUTH=1`, or any tenant token exists): a bearer token is mandatory
      and it alone decides the owner. `x-owner-id` is ignored, because a caller-supplied tenant label
      is not a boundary.
    * **Local single-user**: no token store is in use, so the studio stays usable exactly as before.

    A deployment secret (`RL_IDE_ADMIN_TOKEN`) may create and revoke tokens; it is never itself a
    tenant.
    """
    from fastapi.responses import JSONResponse

    path = request.url.path
    if not path.startswith("/api/"):
        _owner_ctx.set("local")
        return await call_next(request)

    supplied = _bearer_token(request)
    owner = None
    if supplied:
        if _is_admin(supplied):
            owner = _ADMIN_OWNER
        elif _auth_required():
            owner = get_token_store().verify(supplied)
            if owner is None:
                return JSONResponse(
                    {"detail": "Invalid, revoked or unknown API token."},
                    status_code=401,
                    headers={"WWW-Authenticate": "Bearer"},
                )
        else:
            # A token was offered in local mode: honour it if it is real, ignore it otherwise.
            owner = get_token_store().verify(supplied)

    if owner is None:
        if _auth_required():
            return JSONResponse(
                {"detail": "Authentication required: send `Authorization: Bearer <token>`."},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
        owner = (request.headers.get("x-owner-id") or "local").strip() or "local"

    _owner_ctx.set(owner)
    metering.current_owner_id.set(owner)
    return await call_next(request)


# WebSocket Connection Manager
class ConnectionManager:
    """Delivers events to a session's sockets without the producer needing an event loop.

    Training and synthesis run in worker threads; the loop that served the request that started
    them may already be gone (a reload, a shutdown, or TestClient's per-request loop). So producers
    publish into a bounded queue and the websocket drains it, which also covers a client that
    reconnects mid-run.
    """

    _MAX_PENDING = 200

    def __init__(self):
        self.active_connections: dict[str, list[WebSocket]] = {}
        self._pending: dict[str, list[dict[str, Any]]] = {}
        self._lock = threading.Lock()

    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        with self._lock:
            self.active_connections.setdefault(session_id, []).append(websocket)

    def disconnect(self, session_id: str, websocket: WebSocket):
        with self._lock:
            sockets = self.active_connections.get(session_id)
            if sockets and websocket in sockets:
                sockets.remove(websocket)

    def publish(self, session_id: str, message: dict[str, Any]) -> None:
        """Thread-safe, loop-free emit; safe to call from any worker."""
        with self._lock:
            pending = self._pending.setdefault(session_id, [])
            pending.append(message)
            if len(pending) > self._MAX_PENDING:
                del pending[: len(pending) - self._MAX_PENDING]

    def take_pending(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            pending = self._pending.pop(session_id, [])
        return pending

    async def broadcast(self, session_id: str, message: dict[str, Any]):
        """Sends immediately to connected sockets; used when the caller is already on the loop."""
        with self._lock:
            sockets = list(self.active_connections.get(session_id, []))
        dead_sockets = []
        for connection in sockets:
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
    bind_usage_context(req.conversation_id)
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
    bind_usage_context(req.session_id)
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
    def sync_event_emitter(event_data: dict[str, Any]):
        manager.publish(sid, event_data)
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

    # Re-instantiate SynthesizedGymEnv and reload saved SB3 model into global_store.
    # Generated code is only executed in-process for the local single-user studio, and only when the
    # sandboxed serve worker is unavailable; otherwise `/interact` goes through that worker
    # (see SANDBOXED_INTERACT_AVAILABLE).
    ws_dir = global_store.base_dir / conversation_id
    env_file = ws_dir / "env_module.py"
    if env_file.exists() and not SANDBOXED_INTERACT_AVAILABLE and current_owner() == "local":
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
    # The header used to hardcode "unmonitored"; report what live scoring actually knows.
    conv["drift"] = drift_snapshot(conversation_id)
    conv["drift_status"] = conv["drift"].get("status", "unmonitored")
    # A reload mid-synthesis must still say "the bespoke canvas is on its way": the pending flag is
    # derived from the durable job row, so it can never go stale after a crash or restart.
    active = get_job_store().active_job(conversation_id)
    conv["active_job"] = (
        {"job_id": active["job_id"], "kind": active["kind"], "stage": active["stage"], "status": active["status"]}
        if active
        else None
    )
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
        close_interact_session(cid)
        forget_drift_state(cid)
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
    close_interact_session(conversation_id)
    forget_drift_state(conversation_id)
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


def run_training_task(
    *,
    conversation_id: str,
    blueprint: AgentBlueprintSpec,
    env_file: Path,
    dataset: Any,
    max_rounds: int,
    additional_rounds: int | None,
    episodes_per_burst: int,
    on_event,
) -> SessionState:
    """Trains a session, preferring the sandbox and never silently ignoring its failure.

    A missing isolation mechanism (no bwrap/unshare, e.g. macOS or a locked-down CI) falls back to
    in-process execution with a loud event; an isolation *failure* does not, because falling back
    would re-open the boundary the sandbox exists to close.
    """
    from rl_ide.autopilot.sandbox_runner import SandboxedTrainingError, run_rounds_in_sandbox
    from rl_ide.security import SandboxUnavailable
    from rl_ide.llm.client import get_active_llm_client

    # A retrained or rewritten environment invalidates any cached interactive session.
    close_interact_session(conversation_id)

    records = dataset.df.to_dict(orient="records") if (dataset is not None and hasattr(dataset, "df")) else None

    if isolation_available():
        try:
            summary = run_rounds_in_sandbox(
                session_id=conversation_id,
                blueprint=blueprint.model_dump(),
                env_module=env_file,
                records=records,
                base_dir=global_store.base_dir,
                max_rounds=max_rounds,
                episodes_per_burst=episodes_per_burst,
                additional_rounds=additional_rounds,
                on_event=on_event,
                llm_client=get_active_llm_client(),
            )
            summary.pop("scorecard", None)      # re-read from the copied workspace file
            session = SessionState.model_validate(summary)
            workspace = global_store.base_dir / conversation_id
            env_path = workspace / "env_module.py"
            arena_path = workspace / "arena.html"
            if env_path.exists():
                session.env_module_code = env_path.read_text(encoding="utf-8")
            if arena_path.exists():
                session.arena_html = arena_path.read_text(encoding="utf-8")
            session.scorecard = load_session_scorecard(conversation_id)
            global_store.set_session(session)
            _refresh_drift_baseline(conversation_id, session, blueprint)
            return session
        except SandboxUnavailable as exc:
            on_event({"event": "sandbox_unavailable", "conversation_id": conversation_id,
                      "message": f"Isolation unavailable ({exc}); falling back to in-process training."})
        except SandboxedTrainingError as exc:
            on_event({"event": "sandbox_failed", "conversation_id": conversation_id,
                      "message": f"Sandboxed training failed: {exc}"})
            raise
        except Exception as exc:  # noqa: BLE001 - surface, do not silently degrade
            on_event({"event": "sandbox_failed", "conversation_id": conversation_id,
                      "message": f"Sandboxed training error: {exc}"})
            raise

    on_event({"event": "sandbox_bypassed", "conversation_id": conversation_id,
              "message": "Training runs in the API process (single-user mode, no isolation)."})
    env_cls = load_synthesized_env_class(env_file)
    trained = global_controller.run_loop(
        session_id=conversation_id,
        blueprint=blueprint,
        env_cls=env_cls,
        dataset=dataset,
        on_event_callback=on_event,
        max_rounds=max_rounds,
        additional_rounds=additional_rounds,
        episodes_per_burst=episodes_per_burst,
    )
    _refresh_drift_baseline(conversation_id, trained, blueprint)
    return trained


# ---------------------------------------------------------------------------
# Durable jobs (G5/G6)
#
# A studio turn is 2-3 minutes of synthesis. Before this, it existed only as an in-process coroutine:
# no identity to poll, no cancel, and a restart erased it silently. Every turn is now a row in
# `jobs`, training is claimed from the same table by a worker, and a restart turns whatever was
# running into an explicit, retryable failure.
# ---------------------------------------------------------------------------
jobs = get_job_store()

_jobs_lock = threading.RLock()
_training_consumer: threading.Thread | None = None


def _remember_loop() -> None:
    """Fails work a previous process could not finish, so a restart never leaves a silent spinner."""
    if not is_test_mode():
        interrupted = jobs.reconcile_interrupted()
        if interrupted:
            print(f"[jobs] marked {interrupted} interrupted job(s) as failed after restart")


def _broadcast_threadsafe(conversation_id: str, event: dict[str, Any]) -> None:
    """Publishes an event for the session's sockets; callable from any thread, no loop required."""
    if conversation_id:
        manager.publish(conversation_id, event)


def job_public(job: dict[str, Any]) -> dict[str, Any]:
    """Client-facing view of a job (no internal payload echoed back)."""
    return {
        "job_id": job["job_id"],
        "conversation_id": job.get("conversation_id"),
        "kind": job.get("kind"),
        "status": job.get("status"),
        "stage": job.get("stage"),
        "detail": job.get("detail"),
        "error": job.get("error"),
        "retryable": job.get("retryable", False),
        "resume_hint": _resume_hint(job),
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),
    }


def _resume_hint(job: dict[str, Any]) -> str | None:
    """Tells the client how to recover, from durable state rather than from memory.

    A failed chat turn is simply re-sent. A failed *training* job can usually continue: the completed
    rounds and the saved policy survived, so the remaining rounds can be queued instead of redoing
    the work.
    """
    if job.get("status") != FAILED:
        return None
    if job.get("kind") != "training":
        return "Re-send the message; nothing was charged for the failed step."
    conv_id = job.get("conversation_id")
    if not conv_id:
        return None
    conv = get_conversation_store().get_conversation(conv_id)
    if not conv:
        return None
    done = len(conv.get("rounds_history") or [])
    if done == 0:
        return "No round completed; start training again from the arena."
    return (
        f"{done} round(s) were saved. Resume with "
        f"POST /api/v4/conversations/{conv_id}/training/resume to continue from the saved policy."
    )


def _run_chat_job(job: dict[str, Any]) -> None:
    """Executes one studio turn and records its outcome. Runs off the request thread."""
    job_id = job["job_id"]
    conversation_id = job.get("conversation_id") or ""
    payload = job.get("payload") or {}
    store = get_job_store()

    # A fresh thread does not inherit context vars: attribute spend to the right tenant.
    metering.current_owner_id.set(job.get("owner_id") or "local")
    metering.current_conversation_id.set(conversation_id or None)

    def progress(stage: str, detail: str | None = None) -> None:
        store.set_stage(job_id, stage, detail)
        _broadcast_threadsafe(
            conversation_id,
            {"event": "job_stage", "job_id": job_id, "stage": stage, "detail": detail},
        )

    def cancelled() -> bool:
        return store.is_cancel_requested(job_id)

    try:
        progress("intake", "Reading the request and planning changes")
        dataset_id = payload.get("dataset_id")
        dataset = dataset_store.get(dataset_id, job.get("owner_id") or "local") if dataset_id else None
        force_action = payload.get("force_action")

        response = global_omni_agent.process_message(
            conversation_id=conversation_id,
            user_message=payload.get("message") or "",
            dataset=dataset,
            dataset_id=dataset_id,
            selected_options=payload.get("answers") or {},
            training_params=payload.get("training_params"),
            force_action=StudioActionType(force_action) if force_action else None,
            broadcast_callback=lambda event: _broadcast_threadsafe(conversation_id, event),
            run_sync=bool(payload.get("run_sync")),
            progress_callback=progress,
            should_cancel=cancelled,
            owner_id=job.get("owner_id") or "local",
        )

        result = response.model_dump(mode="json")
        store.finish(job_id, status=SUCCEEDED, result=result)
        _broadcast_threadsafe(
            conversation_id,
            {
                "event": "job_completed",
                "job_id": job_id,
                "status": response.status,
                "action_types": [a.value for a in response.action_types],
            },
        )

        if not payload.get("run_sync"):
            _enqueue_training_if_requested(
                conversation_id=conversation_id,
                response=response,
                owner_id=job.get("owner_id") or "local",
                dataset=dataset,
            )
    except JobCancelled:
        store.finish(job_id, status=CANCELLED, error="Cancelled by the user.")
        _broadcast_threadsafe(conversation_id, {"event": "job_cancelled", "job_id": job_id})
    except TrainingInProgressError as exc:
        store.finish(job_id, status=FAILED, error=f"TrainingInProgressError: {exc}")
    except Exception as exc:  # noqa: BLE001 - surfaced to the client, never silently swallowed
        store.finish(job_id, status=FAILED, error=f"{type(exc).__name__}: {exc}")
        _broadcast_threadsafe(
            conversation_id, {"event": "job_failed", "job_id": job_id, "message": str(exc)}
        )


def _enqueue_training_if_requested(
    *,
    conversation_id: str,
    response: Any,
    owner_id: str,
    dataset: Any,
) -> dict[str, Any] | None:
    """Queues a training job instead of holding a coroutine in this process's memory."""
    if StudioActionType.RUN_OR_CONTINUE_TRAINING not in (response.action_types or []):
        return None

    conv = get_conversation_store().get_conversation(conversation_id)
    if not conv or not conv.get("blueprint_spec"):
        return None
    if not (global_store.base_dir / conversation_id / "env_module.py").exists():
        return None

    bp = AgentBlueprintSpec.model_validate(conv["blueprint_spec"])
    session = global_store.get_session(conversation_id)
    prior_rounds = len(session.rounds_history) if session else 0
    add_rounds = int(
        (response.training_params or {}).get("max_auto_tune_rounds") or bp.max_auto_tune_rounds or 5
    )

    job = jobs.create(
        kind="training",
        conversation_id=conversation_id,
        owner_id=owner_id,
        payload={
            "dataset_id": bp.dataset_id or getattr(dataset, "dataset_id", None),
            "max_rounds": add_rounds,
            "additional_rounds": add_rounds if prior_rounds else None,
            "episodes_per_burst": int(
                (response.training_params or {}).get("episodes_per_round") or bp.episodes_per_round or 200
            ),
        },
    )
    _ensure_training_consumer()
    _broadcast_threadsafe(
        conversation_id,
        {"event": "job_queued", "job_id": job["job_id"], "kind": "training", "stage": "queued"},
    )
    return job


class ResumeTrainingRequest(BaseModel):
    additional_rounds: int | None = None
    episodes_per_burst: int | None = None


@app.post("/api/v4/conversations/{conversation_id}/training/resume")
def resume_training(conversation_id: str, req: ResumeTrainingRequest = ResumeTrainingRequest()):
    """Continues an interrupted training run instead of restarting it (G6 residual).

    A restart used to lose whatever a training round had produced. The completed rounds and the
    saved policy are already durable, and the loop supports warm-starting from a saved model, so a
    resume only has to enqueue the *remaining* rounds. Which rounds remain comes from the
    conversation, not from the client's memory of what it asked for.
    """
    conv_store = get_conversation_store()
    conv = conv_store.get_conversation(conversation_id, owner_id=current_owner())
    if not conv:
        raise HTTPException(status_code=404, detail=f"Conversation '{conversation_id}' not found.")
    if not conv.get("blueprint_spec"):
        raise HTTPException(status_code=409, detail="This conversation has no trained blueprint yet.")
    if not (global_store.base_dir / conversation_id / "env_module.py").exists():
        raise HTTPException(status_code=409, detail="The generated environment is missing; retrain from scratch.")

    active = jobs.active_job(conversation_id, kind="training")
    if active:
        raise HTTPException(
            status_code=409,
            detail=f"Training job '{active['job_id']}' is already {active['status']}.",
        )

    blueprint = AgentBlueprintSpec.model_validate(conv["blueprint_spec"])
    done = len(conv.get("rounds_history") or [])
    target = int(blueprint.max_auto_tune_rounds or 5)
    remaining = req.additional_rounds or max(1, target - done)

    payload = {
        "dataset_id": conv.get("dataset_id"),
        "max_rounds": target,
        "additional_rounds": remaining,
        "episodes_per_burst": int(req.episodes_per_burst or blueprint.episodes_per_round or 200),
        "resume_of_rounds": done,
    }
    job = jobs.create(
        kind="training", conversation_id=conversation_id, owner_id=current_owner(), payload=payload
    )
    _ensure_training_consumer()
    _broadcast_threadsafe(
        conversation_id,
        {
            "event": "job_queued",
            "job_id": job["job_id"],
            "kind": "training",
            "stage": "queued",
            "resume_of_rounds": done,
        },
    )
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "stage": job["stage"],
        "completed_rounds": done,
        "additional_rounds": remaining,
        "warm_start": bool(conv.get("latest_model_zip_path")),
    }


def _run_training_job(job: dict[str, Any]) -> None:
    """Trains one conversation from its queued job record."""
    job_id = job["job_id"]
    conversation_id = job.get("conversation_id") or ""
    payload = job.get("payload") or {}
    store = get_job_store()

    metering.current_owner_id.set(job.get("owner_id") or "local")
    metering.current_conversation_id.set(conversation_id or None)

    if store.is_cancel_requested(job_id):
        store.finish(job_id, status=CANCELLED, error="Cancelled before it started.")
        return

    conv = get_conversation_store().get_conversation(conversation_id)
    if not conv or not conv.get("blueprint_spec"):
        store.finish(job_id, status=FAILED, error="Conversation or blueprint disappeared before training.")
        return

    try:
        store.set_stage(job_id, "training", "Training in the sandbox")
        blueprint = AgentBlueprintSpec.model_validate(conv["blueprint_spec"])
        dataset_id = payload.get("dataset_id")
        trained = run_training_task(
            conversation_id=conversation_id,
            blueprint=blueprint,
            env_file=global_store.base_dir / conversation_id / "env_module.py",
            dataset=dataset_store.get(dataset_id, job.get("owner_id") or "local") if dataset_id else None,
            max_rounds=int(payload.get("max_rounds") or 1),
            additional_rounds=payload.get("additional_rounds"),
            episodes_per_burst=int(payload.get("episodes_per_burst") or 200),
            on_event=lambda event: _broadcast_threadsafe(conversation_id, event),
        )
        get_conversation_store().save_or_update(
            conversation_id=conversation_id,
            status=trained.status,
            current_round=trained.current_round,
            best_metric=trained.current_metric,
            rounds_history=[r.model_dump() for r in trained.rounds_history],
            episodes=trained.episodes,
            latest_model_zip_path=trained.latest_model_zip_path,
        )
        store.finish(
            job_id,
            status=SUCCEEDED,
            result={
                "status": trained.status,
                "current_round": trained.current_round,
                "best_metric": trained.current_metric,
            },
        )
        _broadcast_threadsafe(
            conversation_id,
            {"event": "job_completed", "job_id": job_id, "status": trained.status},
        )
    except TrainingInProgressError as exc:
        store.finish(job_id, status=FAILED, error=f"TrainingInProgressError: {exc}")
    except Exception as exc:  # noqa: BLE001
        store.finish(job_id, status=FAILED, error=f"{type(exc).__name__}: {exc}")
        _broadcast_threadsafe(
            conversation_id, {"event": "job_failed", "job_id": job_id, "message": str(exc)}
        )


def _drain_training_jobs() -> None:
    """Consumes queued training jobs, then exits: the queue lives in SQLite, not in this thread."""
    store = get_job_store()
    idle_polls = 0
    while idle_polls < 3:
        job = store.claim_next("training")
        if job is None:
            idle_polls += 1
            time.sleep(0.2)
            continue
        idle_polls = 0
        _run_training_job(job)


# ---------------------------------------------------------------------------
# Retention (G9): data nobody has touched is erased on a schedule, not on request.
# ---------------------------------------------------------------------------
_retention_thread: threading.Thread | None = None
_retention_lock = threading.Lock()


def _retention_sweeper() -> Any:
    from rl_ide.export.registry import RunBundleRegistry
    from rl_ide.state.retention import RetentionPolicy, RetentionSweeper

    return RetentionSweeper(
        RetentionPolicy.from_env(),
        conversation_store_factory=get_conversation_store,
        dataset_store_factory=lambda: dataset_store,
        registry_factory=lambda: _global_registry(),
    )


def _global_registry():
    return global_registry


def _retention_loop() -> None:
    from rl_ide.state.retention import RetentionPolicy

    while True:
        policy = RetentionPolicy.from_env()
        interval = max(60, int(policy.sweep_interval_seconds))
        try:
            if policy.enabled:
                report = _retention_sweeper().sweep()
                if report.total or report.errors:
                    logger.info("retention sweep: %s", report.as_dict())
        except Exception as exc:  # a sweep must never kill the thread
            logger.warning("retention sweep failed: %s", exc)
        time.sleep(interval)


def _ensure_retention_worker() -> None:
    """Starts the scheduled sweep once, and only when a policy is configured."""
    from rl_ide.state.retention import RetentionPolicy

    if is_test_mode() or not RetentionPolicy.from_env().enabled:
        return
    global _retention_thread
    with _retention_lock:
        if _retention_thread is not None and _retention_thread.is_alive():
            return
        _retention_thread = threading.Thread(
            target=_retention_loop, name="rl-retention", daemon=True
        )
        _retention_thread.start()


def _start_retention() -> None:
    """Called from the lifespan handler below, once the app is starting."""
    _ensure_retention_worker()


@app.get("/api/v4/admin/retention")
def get_retention_policy():
    """The active retention policy, and what a sweep would remove right now."""
    from rl_ide.state.retention import RetentionPolicy

    policy = RetentionPolicy.from_env()
    preview = _retention_sweeper().sweep(dry_run=True) if policy.enabled else None
    return {
        "policy": policy.describe(),
        "would_delete": preview.as_dict() if preview else None,
    }


@app.post("/api/v4/admin/retention/sweep")
def run_retention_sweep(dry_run: bool = Query(default=True)):
    """Runs the policy now. Defaults to a dry run so an armed policy can be inspected first."""
    from rl_ide.state.retention import RetentionPolicy

    policy = RetentionPolicy.from_env()
    if not policy.enabled:
        raise HTTPException(
            status_code=409,
            detail="No retention policy configured (set RL_IDE_RETENTION_DAYS or the per-artefact vars).",
        )
    return _retention_sweeper().sweep(dry_run=dry_run).as_dict()


def _ensure_training_consumer() -> None:
    global _training_consumer
    with _jobs_lock:
        if _training_consumer is not None and _training_consumer.is_alive():
            return
        _training_consumer = threading.Thread(
            target=_drain_training_jobs, name="rl-training-jobs", daemon=True
        )
        _training_consumer.start()


@app.post("/api/v4/conversations/{conversation_id}/chat", response_model=StudioChatResponse)
async def conversation_omni_chat(
    conversation_id: str,
    req: UnifiedConversationChatRequest,
    request: Request,
):
    """Unified Codex-style omni-agent chat endpoint for continuous conversation at any stage.

    Send ``Prefer: respond-async`` to get a ``202 {job_id, stage}`` back immediately and poll
    ``GET /api/v4/jobs/{job_id}``. Without it the call blocks until the turn finishes, which is what
    simple API clients and the test suite expect.
    """
    require_verified_llm()
    bind_usage_context(conversation_id)
    msg = req.get_effective_message()
    if not msg.strip():
        raise HTTPException(status_code=400, detail="User message cannot be empty.")

    _remember_loop()

    conv_store = get_conversation_store()
    existing_conv = conv_store.get_conversation(conversation_id)
    payload = {
        "message": msg,
        "dataset_id": req.dataset_id or (existing_conv.get("dataset_id") if existing_conv else None),
        "answers": req.get_effective_answers(),
        "training_params": req.training_params,
        "force_action": req.force_action.value if req.force_action else None,
        "run_sync": req.run_sync,
    }
    job = jobs.create(
        kind="chat",
        conversation_id=conversation_id,
        owner_id=current_owner(),
        payload=payload,
    )

    if "respond-async" in (request.headers.get("prefer") or "").lower():
        threading.Thread(target=_run_chat_job, args=(job,), name=f"rl-chat-{job['job_id']}", daemon=True).start()
        return JSONResponse(status_code=202, content=job_public(jobs.get(job["job_id"])))

    await asyncio.to_thread(_run_chat_job, job)
    finished = jobs.get(job["job_id"]) or job

    if finished["status"] == CANCELLED:
        raise HTTPException(status_code=409, detail=finished.get("error") or "Job cancelled.")
    if finished["status"] == FAILED:
        error = finished.get("error") or "Studio chat failed."
        if error.startswith("TrainingInProgressError: "):
            raise HTTPException(status_code=409, detail=error.split(": ", 1)[1])
        raise HTTPException(status_code=500, detail=error)

    result = finished.get("result")
    if not result:
        raise HTTPException(status_code=500, detail="Studio chat produced no result.")
    return StudioChatResponse.model_validate(result)


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

    arena_source = "bespoke" if arena_builder.last_source == "llm" else "fallback"
    arena_error = arena_builder.last_error
    conv_store.save_or_update(
        conversation_id,
        arena_html=new_html,
        arena_source=arena_source if not arena_error else conv.get("arena_source"),
    )
    sess = global_store.get_session(conversation_id)
    if sess:
        sess.arena_html = new_html

    await manager.broadcast(
        conversation_id,
        {
            "event": "arena_updated",
            "arena_html": new_html,
            "arena_kind": arena_source,
            "pending_bespoke": False,
            "error": arena_error,
        },
    )
    return {
        "conversation_id": conversation_id,
        "arena_html": new_html,
        "status": "arena_updated" if not arena_error else "arena_update_failed",
        "arena_source": arena_source,
        "error": arena_error,
    }


@app.post("/api/v4/conversations/{conversation_id}/arena/lock")
async def lock_conversation_arena(conversation_id: str, req: ArenaLockRequest = ArenaLockRequest()):
    """Locks or unlocks the Arena visual layout."""
    conv_store = get_conversation_store()
    try:
        updated = conv_store.lock_arena(conversation_id, locked=req.locked)
        return {"conversation_id": conversation_id, "arena_locked": updated.get("arena_locked", True)}
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Conversation '{conversation_id}' not found.")


class CorpusCaseRequest(BaseModel):
    obs_vector: list[float] = Field(default_factory=list)
    expected_actions: list[int] = Field(default_factory=list)
    forbidden_actions: list[int] = Field(default_factory=list)
    rationale: str = ""
    source: str = "user_locked_step"
    waive_case_id: str | None = None
    waive_reason: str = ""


class PassiveHarvestRequest(BaseModel):
    dataset_id: str
    min_usable_rows: int = 100
    max_cases: int = 50


def _corpus_store() -> EvalCorpusStore:
    return EvalCorpusStore(global_store.base_dir)


@app.get("/api/v4/conversations/{conversation_id}/corpus")
def get_eval_corpus(conversation_id: str):
    """The behavioural regression suite: what this policy must keep getting right."""
    store = _corpus_store()
    cases = store.load(conversation_id)
    report = None
    session = global_store.get_session(conversation_id)
    if session and session.eval_corpus:
        report = session.eval_corpus
    elif cases:
        try:
            model = global_store.get_model(conversation_id)
            if model is not None:
                from rl_ide.autopilot.eval_corpus import greedy_predictor

                report = store.evaluate_policy(conversation_id, greedy_predictor(model)).model_dump()
        except Exception:
            report = None
    return {
        "conversation_id": conversation_id,
        "report": report,
        "cases": [c.model_dump() for c in cases],
        "count": len(cases),
    }


@app.post("/api/v4/conversations/{conversation_id}/corpus/cases")
def add_eval_corpus_case(conversation_id: str, req: CorpusCaseRequest):
    """Locks a step (or waives an earlier assertion) into the regression suite."""
    store = _corpus_store()
    if req.waive_case_id:
        waived = store.waive_case(conversation_id, req.waive_case_id, req.waive_reason)
        if not waived:
            raise HTTPException(status_code=404, detail="Case not found.")
        return {"waived": req.waive_case_id}

    case = store.add_case(
        conversation_id,
        req.obs_vector,
        expected_actions=req.expected_actions,
        forbidden_actions=req.forbidden_actions,
        source=req.source,
        rationale=req.rationale,
    )
    if case is None:
        raise HTTPException(
            status_code=400, detail="A case needs an observation and at least one expectation."
        )
    return {"case": case.model_dump()}


@app.post("/api/v4/conversations/{conversation_id}/corpus/harvest-passive")
def harvest_passive_log(conversation_id: str, req: PassiveHarvestRequest):
    """Turns an uploaded frontier interaction log into corpus assertions."""
    try:
        dataset = dataset_store.get(req.dataset_id, owner_id=current_owner())
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {exc}")
    if dataset is None or getattr(dataset, "df", None) is None:
        raise HTTPException(status_code=400, detail="Dataset has no tabular rows.")

    detection = detect_interaction_log(dataset.df, min_usable_rows=req.min_usable_rows)
    if not detection.detected:
        return {"detected": False, "detection": detection.model_dump(), "harvested": None}

    env = global_store.get_env(conversation_id)
    labels = {}
    if env is not None:
        labels = {
            str(label): idx for idx, label in enumerate(getattr(env, "action_labels", []) or [])
        }
    harvested = harvest_into_corpus(
        _corpus_store(),
        conversation_id,
        dataset.df,
        detection,
        env=env,
        label_to_action=labels,
        max_cases=req.max_cases,
    )
    return {"detected": True, "detection": detection.model_dump(), "harvested": harvested}


@app.get("/api/v4/conversations/{conversation_id}/crossfire")
def get_crossfire_closure(conversation_id: str):
    """The pre-registered verdict, or the reasons a verdict cannot be claimed."""
    return {
        "conversation_id": conversation_id,
        "closure": CrossfireProtocolStore(global_store.base_dir).closure_report(conversation_id),
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
    model_config = {"extra": "allow"}
    payload: dict[str, Any] = Field(default_factory=dict)

    def get_effective_payload(self) -> dict[str, Any]:
        if self.payload and len(self.payload) > 0:
            return self.payload
        extra = getattr(self, "__pydantic_extra__", None) or {}
        base = {k: v for k, v in self.__dict__.items() if k != "payload" and not k.startswith("_")}
        base.update(extra)
        return base or self.payload


# Interactive play runs in a persistent sandboxed worker whenever bwrap is available. Without it,
# a hosted tenant must not be able to trigger in-process execution, so only the local studio may use
# the legacy path.
SANDBOXED_INTERACT_AVAILABLE = isolation_available()

_interact_sessions: dict[str, Any] = {}
_interact_sessions_lock = threading.RLock()
# A serve worker holds a whole interpreter plus torch: never let them accumulate per conversation.
_INTERACT_IDLE_SECONDS = float(os.environ.get("RL_IDE_INTERACT_IDLE_SECONDS", "900"))
_MAX_INTERACT_SESSIONS = int(os.environ.get("RL_IDE_MAX_INTERACT_SESSIONS", "8"))


# -----------------------------------------------------------------------------
# Live drift scoring: the scorecard proves the policy was good, this proves it still is.
# -----------------------------------------------------------------------------
_drift_monitors: dict[str, Any] = {}
_drift_lock = threading.RLock()
# Calibrated in rl_ide/drift/live.py: 50 observations is the smallest window whose PSI is stable,
# and two consecutive crossings are required before an alert. Do not lower these without re-running
# the calibration test, or unchanged traffic will be reported as drifting.
_DRIFT_WINDOW = int(os.environ.get("RL_IDE_DRIFT_WINDOW", "50"))
_DRIFT_STRIDE = int(os.environ.get("RL_IDE_DRIFT_STRIDE", "5"))


def _drift_workspace(session_id: str) -> Path:
    return global_store.base_dir / session_id


def _baseline_for(session_id: str):
    """Returns this conversation's drift baseline, fitting one from its episodes when absent."""
    from rl_ide.drift import DriftBaseline, fit_baseline_from_episodes

    ws = _drift_workspace(session_id)
    path = ws / "drift_baseline.json"
    if path.exists():
        try:
            return DriftBaseline.load(path)
        except Exception:
            pass

    conv = get_conversation_store().get_conversation(session_id) or {}
    blueprint = conv.get("blueprint_spec") or {}
    baseline = fit_baseline_from_episodes(conv.get("episodes") or [], blueprint.get("observation_features") or [])
    if baseline is not None:
        try:
            ws.mkdir(parents=True, exist_ok=True)
            baseline.save(path)
        except Exception:
            pass
    return baseline


def _drift_monitor(session_id: str):
    """Lazily builds the monitor so a conversation that never trains costs nothing."""
    from rl_ide.drift import DriftMonitor

    with _drift_lock:
        monitor = _drift_monitors.get(session_id)
        if monitor is not None:
            return monitor
        baseline = _baseline_for(session_id)
        if baseline is None:
            return None
        monitor = DriftMonitor(baseline, window=_DRIFT_WINDOW, stride=_DRIFT_STRIDE)
        _drift_monitors[session_id] = monitor
        return monitor


def drift_snapshot(session_id: str) -> dict[str, Any]:
    """Current drift state for a conversation, as the API and the header report it."""
    from rl_ide.drift import DriftStatusStore, UNMONITORED

    store = DriftStatusStore()
    with _drift_lock:
        monitor = _drift_monitors.get(session_id)
    if monitor is not None:
        return monitor.snapshot()
    persisted = store.load(_drift_workspace(session_id))
    if persisted.get("status") and persisted.get("status") != UNMONITORED:
        return persisted
    # Fitting from the stored episodes writes the baseline once, so later reads are a file stat.
    # Without this the header said "unmonitored" for a trained policy until someone clicked.
    if _baseline_for(session_id) is None:
        return {"status": UNMONITORED, "samples_seen": 0, "window": _DRIFT_WINDOW}
    return {"status": "warming_up", "samples_seen": 0, "window": _DRIFT_WINDOW}


def score_live_drift(session_id: str, result: dict[str, Any]) -> dict[str, Any] | None:
    """Scores one live interaction. Returns the report-shaped payload when a window was judged."""
    monitor = _drift_monitor(session_id)
    if monitor is None or not isinstance(result, dict):
        return None
    try:
        report = monitor.observe(
            result.get("obs_vector"),
            result.get("action_probs"),
            len(result.get("action_probs") or []) or None,
        )
    except Exception as exc:  # monitoring must never break play
        logger.warning("drift scoring failed for %s: %s", session_id, exc)
        return None

    # A warming-up window still reports its progress: the UI must be able to say "collecting
    # samples" instead of showing nothing until the first verdict.
    snapshot = monitor.snapshot()
    if report is None:
        return snapshot
    from rl_ide.drift import DriftStatusStore

    DriftStatusStore().save(_drift_workspace(session_id), snapshot)
    return snapshot


def _refresh_drift_baseline(conversation_id: str, session: Any, blueprint: AgentBlueprintSpec) -> None:
    """Refits the live-drift baseline from the episodes training just produced.

    A retrained policy distributes its actions differently, so keeping the old baseline would flag
    the new policy as drifting against the old one.
    """
    from rl_ide.drift import fit_baseline_from_episodes

    forget_drift_state(conversation_id)
    # A baseline is advisory: nothing here may fail a training run that already succeeded.
    try:
        episodes = [
            ep.model_dump(mode="json") if hasattr(ep, "model_dump") else ep
            for ep in getattr(session, "episodes", []) or []
        ]
        baseline = fit_baseline_from_episodes(episodes, list(blueprint.observation_features or []))
        if baseline is None:
            return
        ws = _drift_workspace(conversation_id)
        ws.mkdir(parents=True, exist_ok=True)
        baseline.save(ws / "drift_baseline.json")
    except Exception as exc:
        logger.warning("could not build drift baseline for %s: %s", conversation_id, exc)


def forget_drift_state(session_id: str) -> None:
    """Drops in-memory drift state for a conversation that is being deleted or retrained."""
    with _drift_lock:
        _drift_monitors.pop(session_id, None)


def _resolve_model_zip(session_id: str) -> Path | None:
    conv = get_conversation_store().get_conversation(session_id)
    candidates = [
        conv.get("latest_model_zip_path") if conv else None,
        Path("runs") / session_id / "model.zip",
        global_store.base_dir / session_id / "model.zip",
    ]
    for candidate in candidates:
        if candidate and Path(str(candidate)).exists():
            return Path(str(candidate))
    return None


def _purge_orphan_serve_dirs(max_age_seconds: float = 3600.0) -> None:
    """Deletes case dirs a dead process left behind: each can still hold staged customer rows.

    Only dirs untouched for ``max_age_seconds`` are removed so a second instance sharing this
    working directory (a test run beside a dev server) never deletes a live session's case.
    """
    root = Path("runs") / "sandbox"
    if not root.is_dir():
        return
    now = time.time()
    with _interact_sessions_lock:
        live = {session.case_dir for session in _interact_sessions.values() if session.case_dir}
    for case in root.glob("serve_*"):
        if case in live:
            continue
        try:
            if now - case.stat().st_mtime < max_age_seconds:
                continue
        except OSError:
            continue
        shutil.rmtree(case, ignore_errors=True)


def _sweep_interact_sessions() -> None:
    """Closes idle and surplus sandboxed environments (bounded memory, no orphan processes)."""
    now = time.monotonic()
    stale: list[str] = []
    with _interact_sessions_lock:
        for sid, session in list(_interact_sessions.items()):
            if now - getattr(session, "last_used", now) > _INTERACT_IDLE_SECONDS:
                stale.append(sid)
        while len(_interact_sessions) - len(stale) > _MAX_INTERACT_SESSIONS:
            candidates = [sid for sid in _interact_sessions if sid not in stale]
            if not candidates:
                break
            oldest = min(candidates, key=lambda sid: getattr(_interact_sessions[sid], "last_used", 0.0))
            stale.append(oldest)
        evicted = [_interact_sessions.pop(sid) for sid in stale if sid in _interact_sessions]
    for session in evicted:
        try:
            session.close()
        except Exception:
            pass
    _purge_orphan_serve_dirs()


def close_all_interact_sessions() -> None:
    with _interact_sessions_lock:
        sessions = list(_interact_sessions.values())
        _interact_sessions.clear()
    for session in sessions:
        try:
            session.close()
        except Exception:
            pass


atexit.register(close_all_interact_sessions)

# A previous process may have died without closing its sessions; their case dirs hold staged rows.
# Skipped under pytest, where an API instance may be serving a developer's live session in parallel.
if not is_test_mode():
    _purge_orphan_serve_dirs()


def get_interact_session(session_id: str):
    """Returns this session's sandboxed environment, starting one on first use."""
    from rl_ide.security.serve_client import SandboxedEnvSession

    _sweep_interact_sessions()
    with _interact_sessions_lock:
        existing = _interact_sessions.get(session_id)
        if existing is not None:
            return existing

        workspace = global_store.base_dir / session_id
        env_file = workspace / "env_module.py"
        if not env_file.exists():
            conv = get_conversation_store().get_conversation(session_id)
            if conv and conv.get("env_code"):
                workspace.mkdir(parents=True, exist_ok=True)
                env_file.write_text(conv["env_code"], encoding="utf-8")
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f"Active environment not found for session '{session_id}'.",
                )

        session = SandboxedEnvSession()
        session.start(
            env_module=env_file,
            records=load_records(workspace),
            model_path=_resolve_model_zip(session_id),
        )
        _interact_sessions[session_id] = session
        return session


def close_interact_session(session_id: str) -> None:
    """Drops the sandboxed environment after training or deletion so stale code is never served."""
    with _interact_sessions_lock:
        session = _interact_sessions.pop(session_id, None)
    if session is not None:
        try:
            session.close()
        except Exception:
            pass


def _with_drift(session_id: str, result: Any) -> Any:
    """Attaches the live drift state to an interaction and warns the UI when it crosses over."""
    snapshot = score_live_drift(session_id, result if isinstance(result, dict) else {})
    if snapshot is None or not isinstance(result, dict):
        return result
    if snapshot.get("is_drifted"):
        _broadcast_threadsafe(
            session_id,
            {
                "event": "drift_warning",
                "message": snapshot.get("recommendation"),
                "drift": snapshot,
            },
        )
    return {**result, "drift": snapshot}


@app.post("/api/v4/sessions/{session_id}/interact")
def session_interact(session_id: str, req: InteractRequest):
    """Executes live human play/inspection against the agent inside arena.html."""
    if global_controller.is_busy(session_id):
        raise HTTPException(
            status_code=409,
            detail="Training is currently mutating this session's policy; retry once the round completes.",
        )
    if SANDBOXED_INTERACT_AVAILABLE:
        try:
            result = get_interact_session(session_id).interact(req.get_effective_payload())
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Interactive step failed: {exc}")
        return _with_drift(session_id, result)

    if current_owner() != "local":
        raise HTTPException(
            status_code=501,
            detail=(
                "Interactive play requires sandboxed execution, and bwrap is not available on this "
                "host. Scoring, export and the held-out scorecard are unaffected."
            ),
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

    # Captured from the environment, because a human click specifies the action and the policy is
    # not always consulted.
    from rl_ide.drift.live import capture_observations

    seen_obs = capture_observations(env)

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
        result = env.interactive_step(req.get_effective_payload(), policy_predict)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Interactive step failed: {exc}")
    if isinstance(result, dict) and seen_obs:
        result = {**result, "obs_vector": list(seen_obs)}
    return _with_drift(session_id, result)


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


class TokenIssueRequest(BaseModel):
    owner_id: str
    label: str = ""


def _require_admin(request: Request) -> None:
    """Token management is a deployment action, not a tenant action."""
    token = _bearer_token(request)
    if not _is_admin(token):
        raise HTTPException(
            status_code=403,
            detail="Token administration requires the deployment admin token (RL_IDE_ADMIN_TOKEN).",
        )


@app.post("/api/v4/admin/tokens")
def issue_token(req: TokenIssueRequest, request: Request):
    """Mints a tenant token. The plaintext is returned once and never stored."""
    _require_admin(request)
    try:
        issued = get_token_store().create(owner_id=req.owner_id, label=req.label)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return issued


@app.get("/api/v4/admin/tokens")
def list_tokens(request: Request, owner_id: str | None = Query(default=None)):
    """Metadata for issued tokens; never the token or its hash."""
    _require_admin(request)
    return get_token_store().list(owner_id=owner_id)


@app.delete("/api/v4/admin/tokens/{token_id}")
def revoke_token(token_id: str, request: Request):
    _require_admin(request)
    if not get_token_store().revoke(token_id):
        raise HTTPException(status_code=404, detail=f"Token '{token_id}' not found.")
    return {"token_id": token_id, "revoked": True}


@app.get("/api/v4/whoami")
def whoami():
    """The tenant this request is scoped to — what a client should display and log."""
    return {"owner_id": current_owner(), "authenticated": _auth_required()}


@app.get("/api/v4/capabilities")
def get_capabilities():
    """What this deployment can do right now, and what still needs a model.

    The studio is usable without a provider: upload, profiling, browsing existing agents, the
    deterministic arena preview and ONNX export are all local. Synthesis, architect chat, training
    and restyling need one. Reporting that explicitly is what lets the UI offer exploration instead
    of a hard lockout (G13).
    """
    status = get_llm_status()
    return {
        "llm": {
            "verified": bool(status.get("is_verified")),
            "provider": status.get("provider"),
            "model_name": status.get("model_name"),
            "offline_test_mode": bool(is_test_mode()),
        },
        "needs_llm": [
            "synthesize_environment",
            "architect_chat",
            "train",
            "restyle_arena",
        ],
        "works_without_llm": [
            "upload_dataset",
            "profile_dataset",
            "list_conversations",
            "read_arena",
            "arena_preview",
            "export_onnx",
        ],
        "auth": {
            "required": _auth_required(),
            "tenant_tokens": (get_token_store().count() if _auth_required() else 0),
            "admin_token_configured": bool(_ADMIN_TOKEN),
        },
        "isolation": {
            "sandboxed_execution": bool(isolation_available()),
            "required": os.environ.get("RL_IDE_REQUIRE_ISOLATION", "").lower() in ("1", "true", "yes"),
        },
    }


@app.get("/api/v4/jobs")
def list_jobs(
    conversation_id: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
):
    """Recent jobs for this tenant (optionally one conversation), newest first."""
    return [
        job_public(job)
        for job in jobs.list_jobs(
            conversation_id=conversation_id, owner_id=current_owner(), limit=limit
        )
    ]


@app.get("/api/v4/jobs/{job_id}")
def get_job(job_id: str):
    """Poll a job: its stage while it runs, its result once it succeeds, its retry hint on failure."""
    job = jobs.get(job_id)
    if not job or job.get("owner_id") != current_owner():
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    view = job_public(job)
    if job.get("status") == SUCCEEDED:
        view["result"] = job.get("result")
    return view


@app.post("/api/v4/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    """Cooperative cancel: the job stops at the next stage boundary, never mid-write."""
    job = jobs.get(job_id)
    if not job or job.get("owner_id") != current_owner():
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if job.get("status") in (SUCCEEDED, FAILED, CANCELLED):
        raise HTTPException(status_code=409, detail=f"Job already {job['status']}.")
    accepted = jobs.request_cancel(job_id)
    return {
        "job_id": job_id,
        "cancel_requested": True,
        "accepted": accepted,
        "detail": (
            "Cancellation requested; the job stops at the next stage boundary."
            if accepted
            else "The job already finished before the request arrived."
        ),
    }


@app.get("/api/v4/usage")
def get_usage(
    conversation_id: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
):
    """What this conversation, or this tenant, has cost: tokens, priced spend and the budget cap."""
    if conversation_id:
        conv = get_conversation_store().get_conversation(conversation_id, owner_id=current_owner())
        if not conv:
            raise HTTPException(status_code=404, detail=f"Conversation '{conversation_id}' not found.")
    return metering.usage_report(
        conversation_id=conversation_id, owner_id=current_owner(), limit=limit
    )


@app.websocket("/ws/v4/{session_id}")
async def websocket_telemetry(websocket: WebSocket, session_id: str):
    """Streams live training frames, round metrics, job stages and auto-tuner events.

    A websocket carries no HTTP middleware, so the tenant boundary is enforced here too: in an
    authenticated deployment the token arrives as a query parameter (browsers cannot set headers on
    a socket) and the socket may only follow a session the caller owns.
    """
    owner = "local"
    if _auth_required():
        supplied = (websocket.query_params.get("token") or "").strip()
        owner = (_ADMIN_OWNER if _is_admin(supplied) else get_token_store().verify(supplied)) or ""
        if not owner:
            await websocket.close(code=4401)
            return
        conv = get_conversation_store().get_conversation(session_id, owner_id=owner)
        if conv is None:
            await websocket.close(code=4403)
            return
    _owner_ctx.set(owner)
    await manager.connect(session_id, websocket)
    try:
        while True:
            for event in manager.take_pending(session_id):
                await websocket.send_json(event)
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=0.2)
            except asyncio.TimeoutError:
                continue
            try:
                msg = json.loads(data)
                if msg.get("action") == "ping":
                    await websocket.send_json({"event": "pong", "session_id": session_id})
            except Exception:
                pass
    except (WebSocketDisconnect, RuntimeError):
        manager.disconnect(session_id, websocket)


# Mounted last on purpose: a mount at "/" catches every path registered after it, so it must come
# after all /api and /ws routes or the API disappears behind the SPA.
mount_frontend()
