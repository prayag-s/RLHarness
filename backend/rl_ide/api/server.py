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
global_omni_agent = StudioOmniAgent(
    conv_store=get_conversation_store(),
    session_store=global_store,
    loop_controller=global_controller,
)

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
    dataset = ingested_datasets_store.get(req.dataset_id) if req.dataset_id else None
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

    dataset = ingested_datasets_store.get(req.dataset_id) if req.dataset_id else None

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

        asyncio.create_task(_run_background())
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
    """Returns list of all saved conversations sorted by updated_at DESC."""
    conv_store = get_conversation_store()
    return conv_store.list_conversations()


@app.post("/api/v4/conversations")
def create_conversation(req: CreateConversationRequest = CreateConversationRequest()):
    """Creates a new blank conversation and returns its conversation_id."""
    conv_store = get_conversation_store()
    return conv_store.create_conversation(
        title=req.title,
        dataset_id=req.dataset_id,
        training_params=req.training_params,
    )


@app.get("/api/v4/conversations/{conversation_id}")
def get_conversation(conversation_id: str):
    """Loads full conversation state from disk/SQLite, re-instantiates SynthesizedGymEnv and reloads saved SB3 model."""
    conv_store = get_conversation_store()
    conv = conv_store.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail=f"Conversation '{conversation_id}' not found.")

    # Re-instantiate SynthesizedGymEnv and reload saved SB3 model into global_store
    ws_dir = global_store.base_dir / conversation_id
    env_file = ws_dir / "env_module.py"
    if env_file.exists():
        try:
            from rl_ide.synthesizer.env_builder import load_synthesized_env_class, instantiate_env
            env_cls = load_synthesized_env_class(env_file)
            dataset = ingested_datasets_store.get(conv.get("dataset_id")) if conv.get("dataset_id") else None
            records = dataset.df.to_dict(orient="records") if (dataset and hasattr(dataset, "df")) else None
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
            elif conv.get("rounds_history"):
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
    if not existing_session and conv.get("blueprint_spec"):
        try:
            bp = AgentBlueprintSpec.model_validate(conv["blueprint_spec"])
            t_params = conv.get("training_params", {}) or {}
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
                env_module_code=conv.get("env_code", ""),
                arena_html=conv.get("arena_html", ""),
                latest_model_zip_path=conv.get("latest_model_zip_path"),
            )
            global_store.set_session(session)
        except Exception as s_err:
            print(f"Warning: Failed to reconstruct SessionState: {s_err}")

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


@app.delete("/api/v4/conversations/{conversation_id}")
def delete_conversation(conversation_id: str):
    """Deletes a conversation and its workspace files."""
    conv_store = get_conversation_store()
    conv = conv_store.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail=f"Conversation '{conversation_id}' not found.")
    conv_store.delete(conversation_id)
    return {"success": True, "conversation_id": conversation_id}


class UnifiedConversationChatRequest(BaseModel):
    user_message: str | None = None
    message: str | None = None
    prompt: str | None = None
    user_prompt: str | None = None
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

    response = global_omni_agent.process_message(
        conversation_id=conversation_id,
        user_message=msg,
        selected_options=req.get_effective_answers(),
        training_params=req.training_params,
        force_action=req.force_action,
        broadcast_callback=sync_broadcast,
        run_sync=req.run_sync,
    )
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
