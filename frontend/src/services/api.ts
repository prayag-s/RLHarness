import {
  AgentBlueprintSpec,
  ClarifyingQuestion,
  IntakeResponse,
  SessionState,
  TuningResult,
  BundleMetadata,
  IngestedDatasetSummary,
  LLMSettings,
} from '../types'

const API_V4 = '/api/v4'

// -----------------------------------------------------------------------------
// 1. Data Ingestion
// -----------------------------------------------------------------------------

export async function uploadDatasetFile(
  file: File,
  role: string = 'tabular',
): Promise<IngestedDatasetSummary> {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('role', role)

  const res = await fetch(`${API_V4}/ingest/upload`, {
    method: 'POST',
    body: formData,
  })

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Upload failed' }))
    throw new Error(err.detail || 'Dataset file upload failed')
  }

  return res.json()
}

export async function ingestSQLDataset(
  connectionUri: string,
  query: string,
  name: string = 'sql_dataset',
): Promise<IngestedDatasetSummary> {
  const res = await fetch(`${API_V4}/ingest/sql`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      connection_uri: connectionUri,
      query,
      name,
    }),
  })

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'SQL query failed' }))
    throw new Error(err.detail || 'SQL ingestion failed')
  }

  return res.json()
}

// -----------------------------------------------------------------------------
// 2. Conversational Intake Architect
// -----------------------------------------------------------------------------

export interface ArchitectChatParams {
  user_prompt: string
  dataset_id?: string | null
  user_answers?: Record<string, string>
  force_smart_defaults?: boolean
  conversation_history?: Array<{ role: string; content: string }>
}

export async function chatWithArchitect(
  params: ArchitectChatParams,
): Promise<IntakeResponse> {
  const res = await fetch(`${API_V4}/architect/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  })

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Architect interview failed' }))
    throw new Error(err.detail || 'Architect interview failed')
  }

  const data = await res.json()
  const rawQuestions = data.questions || data.clarifying_questions || []
  const normalizedQuestions: ClarifyingQuestion[] = rawQuestions.map((q: any) => ({
    ...q,
    question_id: q.pillar || q.question_id,
  }))

  const rawBlueprint = data.blueprint_spec || data.blueprint || null
  const normalizedBlueprint: AgentBlueprintSpec | null = rawBlueprint
    ? {
        ...rawBlueprint,
        task_id: rawBlueprint.task_title || rawBlueprint.task_id,
        objective: rawBlueprint.domain_summary || rawBlueprint.objective,
        target_metric_name: rawBlueprint.success_metric_name || rawBlueprint.target_metric_name,
      }
    : null

  return {
    status: data.status,
    assistant_message: data.assistant_message || data.message || '',
    message: data.assistant_message || data.message || '',
    questions: normalizedQuestions,
    clarifying_questions: normalizedQuestions,
    blueprint_spec: normalizedBlueprint,
    blueprint: normalizedBlueprint,
  }
}

// -----------------------------------------------------------------------------
// 3. Autonomous Session Synthesis & Training
// -----------------------------------------------------------------------------

export interface StartSessionParams {
  blueprint: AgentBlueprintSpec
  session_id?: string
  dataset_id?: string | null
  run_sync?: boolean
}

export async function startSynthesisSession(
  params: StartSessionParams,
): Promise<{ session_id: string; status: string; message?: string }> {
  const res = await fetch(`${API_V4}/sessions/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  })

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to start synthesis session' }))
    throw new Error(err.detail || 'Failed to start synthesis session')
  }

  return res.json()
}

export async function fetchSession(sessionId: string): Promise<SessionState> {
  const res = await fetch(`${API_V4}/sessions/${sessionId}`)
  if (!res.ok) {
    throw new Error(`Failed to fetch session ${sessionId}`)
  }
  return res.json()
}

export function getSessionArenaUrl(sessionId: string): string {
  return `${API_V4}/sessions/${sessionId}/arena`
}

export async function interactSession(
  sessionId: string,
  payload: Record<string, any>,
): Promise<any> {
  const res = await fetch(`${API_V4}/sessions/${sessionId}/interact`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ payload }),
  })

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Interaction failed' }))
    throw new Error(err.detail || 'Interaction failed')
  }

  return res.json()
}

export async function nudgeSession(
  sessionId: string,
  feedback: string,
): Promise<TuningResult> {
  const res = await fetch(`${API_V4}/sessions/${sessionId}/nudge`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ feedback }),
  })

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Nudge feedback failed' }))
    throw new Error(err.detail || 'Nudge feedback failed')
  }

  return res.json()
}

export async function rollbackSession(
  sessionId: string,
  roundNum: number,
): Promise<{ session_id: string; status: string; restored_round: number; restored_metric: number }> {
  const res = await fetch(`${API_V4}/sessions/${sessionId}/rollback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ round_num: roundNum }),
  })

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Rollback failed' }))
    throw new Error(err.detail || 'Rollback failed')
  }

  return res.json()
}

// -----------------------------------------------------------------------------
// 4. Export & RunBundle Registry
// -----------------------------------------------------------------------------

export interface ExportResult {
  session_id: string
  bundle_dir: string
  zip_path: string
  onnx_parity_diff: number
  download_url: string
}

export async function exportSessionPolicy(sessionId: string): Promise<ExportResult> {
  const res = await fetch(`${API_V4}/sessions/${sessionId}/export`, {
    method: 'POST',
  })

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Export failed' }))
    throw new Error(err.detail || 'Export failed')
  }

  return res.json()
}

export async function fetchBundles(): Promise<BundleMetadata[]> {
  const res = await fetch(`${API_V4}/bundles`)
  if (!res.ok) {
    throw new Error('Failed to fetch bundles')
  }
  return res.json()
}

// -----------------------------------------------------------------------------
// 5. Settings Validation
// -----------------------------------------------------------------------------

export async function validateLLMSettings(
  settings: LLMSettings,
): Promise<{ valid: boolean; provider: string; message: string }> {
  const res = await fetch(`${API_V4}/settings/validate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      provider: settings.provider,
      api_key: settings.apiKey,
      base_url: settings.baseUrl,
      model_name: settings.modelName,
    }),
  })

  if (!res.ok) {
    throw new Error('LLM validation request failed')
  }

  return res.json()
}

// -----------------------------------------------------------------------------
// 6. WebSocket Telemetry Stream
// -----------------------------------------------------------------------------

export function connectTelemetryWebSocket(
  sessionId: string,
  onMessage: (data: any) => void,
  onError?: (err: Event) => void,
  onClose?: () => void,
): WebSocket {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const host = window.location.host
  const wsUrl = `${proto}//${host}/ws/v4/${sessionId}`

  const ws = new WebSocket(wsUrl)

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data)
      onMessage(data)
    } catch (e) {
      console.warn('Failed to parse websocket message', e)
    }
  }

  // Keepalive ping every 15s with guaranteed cleanup
  const pingInterval = setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action: 'ping' }))
    } else if (ws.readyState === WebSocket.CLOSED || ws.readyState === WebSocket.CLOSING) {
      clearInterval(pingInterval)
    }
  }, 15000)

  ws.onclose = () => {
    clearInterval(pingInterval)
    if (onClose) onClose()
  }

  return ws
}
