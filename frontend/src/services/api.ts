import {
  AgentBlueprintSpec,
  CrossfireClosure,
  EvalCorpusCase,
  EvalCorpusResponse,
  GateHistoryResponse,
  PassiveHarvestResult,
  ClarifyingQuestion,
  IntakeResponse,
  SessionState,
  TuningResult,
  BundleMetadata,
  IngestedDatasetSummary,
  LLMSettings,
  LLMStatusResponse,
  ConversationSummary,
  ConversationDetail,
  TrainingParams,
  StudioChatResponse,
  JobStatus,
  UsageReport,
  CoachListResponse,
  CoachReport,
  ThreadResponse,
  TurnStatus,
} from '../types'

const API_V4 = '/api/v4'

/** The backend marks model-backed routes with this reason so the UI need not parse prose. */
export const LLM_KEY_REQUIRED = 'llm-key-required'

const TOKEN_STORAGE_KEY = 'rl_ide_api_token'

/** The tenant token for this browser, or empty in the single-user local studio. */
export function getApiToken(): string {
  try {
    return localStorage.getItem(TOKEN_STORAGE_KEY) || ''
  } catch {
    return ''
  }
}

export function setApiToken(token: string): void {
  try {
    if (token) localStorage.setItem(TOKEN_STORAGE_KEY, token)
    else localStorage.removeItem(TOKEN_STORAGE_KEY)
  } catch {
    // A blocked storage API must not break the studio; requests will simply be unauthenticated.
  }
}

/** Every API call goes through here, so no request can forget the tenant token. */
async function apiFetch(path: string, init: RequestInit = {}): Promise<Response> {
  const merged = authHeaders((init.headers as Record<string, string>) || {})
  const res = await fetch(`${API_V4}${path}`, { ...init, headers: merged })
  if (res.status === 401) {
    // The app listens for this and asks for a tenant token instead of showing a broken studio.
    window.dispatchEvent(new CustomEvent('rl:auth-required'))
  }
  return res
}

/** Headers every request carries: the tenant token when one is configured. */
export function authHeaders(extra: Record<string, string> = {}): Record<string, string> {
  const token = getApiToken()
  return token ? { ...extra, Authorization: `Bearer ${token}` } : { ...extra }
}

export class ApiError extends Error {
  readonly status: number
  readonly reason: string | null
  /** Present when the Journey Coach blocked the action, so the UI can render the exact fix. */
  readonly guidance?: CoachReport

  constructor(
    message: string,
    status: number,
    reason: string | null,
    guidance?: CoachReport,
  ) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.reason = reason
    this.guidance = guidance
  }

  get needsModel(): boolean {
    return this.status === 403 && this.reason === LLM_KEY_REQUIRED
  }
}

async function failFrom(res: Response, fallback: string): Promise<never> {
  const body = await res.json().catch(() => ({ detail: fallback }))
  const rawDetail = (body as any)?.detail
  const message =
    typeof rawDetail === 'string'
      ? rawDetail
      : rawDetail?.message || fallback
  throw new ApiError(
    message,
    res.status,
    res.headers.get('X-RL-Reason'),
    (body as any)?.guidance,
  )
}

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

  const res = await apiFetch(`/ingest/upload`, {
    method: 'POST',
    body: formData,
  })

  if (!res.ok) {
    await failFrom(res, 'Dataset file upload failed')
  }

  return res.json()
}

export async function ingestSQLDataset(
  connectionUri: string,
  query: string,
  name: string = 'sql_dataset',
): Promise<IngestedDatasetSummary> {
  const res = await apiFetch(`/ingest/sql`, {
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
  conversation_id?: string | null
  dataset_id?: string | null
  user_answers?: Record<string, string>
  force_smart_defaults?: boolean
  conversation_history?: Array<{ role: string; content: string }>
  training_params?: TrainingParams
}

export async function chatWithArchitect(
  params: ArchitectChatParams,
): Promise<IntakeResponse> {
  const res = await apiFetch(`/architect/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  })

  if (!res.ok) {
    await failFrom(res, 'Architect interview failed')
  }

  const data = await res.json()
  const rawQuestions = data.questions || data.clarifying_questions || []
  const normalizedQuestions: ClarifyingQuestion[] = rawQuestions.map((q: any) => ({
    ...q,
    id: q.id || q.question_id || q.pillar,
    question_id: q.id || q.question_id || q.pillar,
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
    conversation_id: data.conversation_id,
    questions: normalizedQuestions,
    clarifying_questions: normalizedQuestions,
    blueprint_spec: normalizedBlueprint,
    blueprint: normalizedBlueprint,
    recommended_training_params: data.recommended_training_params,
  }
}

// -----------------------------------------------------------------------------
// 2a. Unified Continuous Conversational Studio (v4.2 StudioOmniAgent)
// -----------------------------------------------------------------------------

export interface StudioChatParams {
  conversation_id: string
  user_message: string
  dataset_id?: string
  selected_options?: Record<string, any>
  training_params?: TrainingParams
  run_sync?: boolean
}

export async function getJob(jobId: string): Promise<JobStatus> {
  const res = await apiFetch(`/jobs/${jobId}`)
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Job lookup failed' }))
    throw new Error(err.detail || 'Job lookup failed')
  }
  return res.json()
}

export async function cancelJob(jobId: string): Promise<void> {
  await apiFetch(`/jobs/${jobId}/cancel`, { method: 'POST' })
}

// Synthesis takes minutes: poll the job so the UI can show its stage instead of a dead spinner.
async function waitForJob(
  jobId: string,
  onStage?: (stage: string, detail?: string | null) => void,
): Promise<StudioChatResponse> {
  const deadline = Date.now() + 15 * 60 * 1000
  let lastStage = ''
  for (;;) {
    const job = await getJob(jobId)
    if (job.stage !== lastStage) {
      lastStage = job.stage
      onStage?.(job.stage, job.detail)
    }
    if (job.status === 'succeeded') {
      if (!job.result) throw new Error('Job finished without a result')
      return job.result
    }
    if (job.status === 'failed') throw new Error(job.error || 'Studio job failed')
    if (job.status === 'cancelled') throw new Error(job.error || 'Studio job cancelled')
    if (Date.now() > deadline) throw new Error('Studio job exceeded 15 minutes; check GET /api/v4/jobs')
    await new Promise((resolve) => setTimeout(resolve, 1200))
  }
}

export async function chatWithStudioOmni(
  params: StudioChatParams,
  onStage?: (stage: string, detail?: string | null) => void,
): Promise<StudioChatResponse> {
  const res = await apiFetch(`/conversations/${params.conversation_id}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Prefer: 'respond-async' },
    body: JSON.stringify({
      user_message: params.user_message,
      dataset_id: params.dataset_id,
      selected_options: params.selected_options || {},
      training_params: params.training_params,
      run_sync: params.run_sync ?? false,
    }),
  })

  if (res.status === 202) {
    const job = await res.json()
    onStage?.(job.stage, job.detail)
    return waitForJob(job.job_id, onStage)
  }

  if (!res.ok) {
    await failFrom(res, 'Studio chat failed')
  }

  const data = await res.json()
  return data
}

export interface ArenaUpdateResult {
  conversation_id: string
  arena_html: string
  status: string
  /** "bespoke" when the provider produced a canvas, "fallback" when it had to be previewed. */
  arena_source?: 'bespoke' | 'fallback'
  error?: string | null
}

export async function updateArenaLook(
  conversationId: string,
  userVisualRequest: string,
): Promise<ArenaUpdateResult> {
  const res = await apiFetch(`/conversations/${conversationId}/arena/update`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_visual_request: userVisualRequest }),
  })

  if (!res.ok) {
    await failFrom(res, 'Failed to update arena look')
  }

  return res.json()
}

export async function lockConversationArena(
  conversationId: string,
  locked: boolean = true,
): Promise<{ conversation_id: string; arena_locked: boolean }> {
  const res = await apiFetch(`/conversations/${conversationId}/arena/lock`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ locked }),
  })

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to lock arena' }))
    throw new Error(err.detail || 'Failed to lock arena')
  }

  return res.json()
}

// -----------------------------------------------------------------------------
// 2b. Conversation Management & Persistence
// -----------------------------------------------------------------------------

export async function fetchConversations(): Promise<ConversationSummary[]> {
  const res = await apiFetch(`/conversations`)
  if (!res.ok) {
    throw new Error('Failed to fetch conversations')
  }
  return res.json()
}

export async function createConversation(
  title: string = 'New Agent',
  datasetId?: string | null,
  trainingParams?: TrainingParams,
): Promise<ConversationDetail> {
  const res = await apiFetch(`/conversations`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      title,
      dataset_id: datasetId,
      training_params: trainingParams,
    }),
  })
  if (!res.ok) {
    throw new Error('Failed to create conversation')
  }
  return res.json()
}

export async function fetchConversation(conversationId: string): Promise<ConversationDetail> {
  const res = await apiFetch(`/conversations/${conversationId}`)
  if (!res.ok) {
    throw new Error(`Failed to load conversation ${conversationId}`)
  }
  return res.json()
}

/**
 * The single ordered transcript. `sinceSeq` makes this cheap to poll while a turn is running.
 */
export async function fetchThread(
  conversationId: string,
  sinceSeq = 0,
  limit = 500,
): Promise<ThreadResponse> {
  const params = new URLSearchParams()
  if (sinceSeq > 0) params.set('since_seq', String(sinceSeq))
  params.set('limit', String(limit))
  const res = await apiFetch(`/conversations/${conversationId}/thread?${params.toString()}`)
  if (!res.ok) {
    throw new Error(`Failed to load the thread for ${conversationId}`)
  }
  return res.json()
}

/** Stop is the only way to interrupt a turn, and it always hands the conversation back. */
export async function stopTurn(
  conversationId: string,
): Promise<{ conversation_id: string; cancelled_job: string | null; turn: TurnStatus }> {
  const res = await apiFetch(`/conversations/${conversationId}/turn/stop`, { method: 'POST' })
  if (!res.ok) {
    throw new Error('Could not stop the current turn')
  }
  return res.json()
}

export async function updateConversationParams(
  conversationId: string,
  params: TrainingParams,
): Promise<any> {
  const res = await apiFetch(`/conversations/${conversationId}/params`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  })
  if (!res.ok) {
    throw new Error('Failed to update conversation parameters')
  }
  return res.json()
}

export async function clearAllConversations(): Promise<{ success: boolean; deleted_count: number }> {
  const res = await apiFetch(`/conversations`, {
    method: 'DELETE',
  })
  if (!res.ok) {
    throw new Error('Failed to clear conversations')
  }
  return res.json()
}

export async function deleteConversation(conversationId: string): Promise<any> {
  const res = await apiFetch(`/conversations/${conversationId}`, {
    method: 'DELETE',
  })
  if (!res.ok) {
    throw new Error(`Failed to delete conversation ${conversationId}`)
  }
  return res.json()
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
  const res = await apiFetch(`/sessions/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  })

  if (!res.ok) {
    await failFrom(res, 'Failed to start synthesis session')
  }

  return res.json()
}

export async function fetchSession(sessionId: string): Promise<SessionState> {
  const res = await apiFetch(`/sessions/${sessionId}`)
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
  const res = await apiFetch(`/sessions/${sessionId}/interact`, {
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

// -----------------------------------------------------------------------------
// Trust surface: golden corpus, Crossfire protocol, passive harvest
// -----------------------------------------------------------------------------

export async function fetchEvalCorpus(
  conversationId: string,
): Promise<EvalCorpusResponse> {
  const res = await apiFetch(`/conversations/${conversationId}/corpus`)
  if (!res.ok) {
    await failFrom(res, 'Could not read the eval corpus')
  }
  return res.json()
}

export async function addCorpusCase(
  conversationId: string,
  payload: {
    obs_vector: number[]
    expected_actions?: number[]
    forbidden_actions?: number[]
    rationale?: string
    source?: string
  },
): Promise<{ case: EvalCorpusCase }> {
  const res = await apiFetch(`/conversations/${conversationId}/corpus/cases`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) {
    await failFrom(res, 'Could not lock this case')
  }
  return res.json()
}

export async function waiveCorpusCase(
  conversationId: string,
  caseId: string,
  reason: string,
): Promise<{ waived: string }> {
  const res = await apiFetch(`/conversations/${conversationId}/corpus/cases`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ waive_case_id: caseId, waive_reason: reason }),
  })
  if (!res.ok) {
    await failFrom(res, 'Could not waive this case')
  }
  return res.json()
}

export async function harvestPassiveLog(
  conversationId: string,
  datasetId: string,
  options: { min_usable_rows?: number; max_cases?: number } = {},
): Promise<PassiveHarvestResult> {
  const res = await apiFetch(`/conversations/${conversationId}/corpus/harvest-passive`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dataset_id: datasetId, ...options }),
  })
  if (!res.ok) {
    await failFrom(res, 'Could not harvest this log')
  }
  return res.json()
}

export async function fetchGateHistory(
  conversationId: string,
): Promise<GateHistoryResponse> {
  const res = await apiFetch(`/conversations/${conversationId}/gates`)
  if (!res.ok) {
    await failFrom(res, 'Could not read the gate history')
  }
  return res.json()
}

export async function fetchCrossfireClosure(
  conversationId: string,
): Promise<{ conversation_id: string; closure: CrossfireClosure }> {
  const res = await apiFetch(`/conversations/${conversationId}/crossfire`)
  if (!res.ok) {
    await failFrom(res, 'Could not read the Crossfire protocol')
  }
  return res.json()
}

export async function nudgeSession(
  sessionId: string,
  feedback: string,
): Promise<TuningResult> {
  const res = await apiFetch(`/sessions/${sessionId}/nudge`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ feedback }),
  })

  if (!res.ok) {
    await failFrom(res, 'Nudge feedback failed')
  }

  return res.json()
}

export async function rollbackSession(
  sessionId: string,
  roundNum: number,
): Promise<{ session_id: string; status: string; restored_round: number; restored_metric: number }> {
  const res = await apiFetch(`/sessions/${sessionId}/rollback`, {
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
  const res = await apiFetch(`/sessions/${sessionId}/export`, {
    method: 'POST',
  })

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Export failed' }))
    throw new Error(err.detail || 'Export failed')
  }

  return res.json()
}

export async function fetchBundles(): Promise<BundleMetadata[]> {
  const res = await apiFetch(`/bundles`)
  if (!res.ok) {
    throw new Error('Failed to fetch bundles')
  }
  return res.json()
}

// -----------------------------------------------------------------------------
// 5. Settings, LLM Status & Verification
// -----------------------------------------------------------------------------

export async function getLLMStatus(): Promise<LLMStatusResponse> {
  const res = await apiFetch(`/settings/llm`)
  if (!res.ok) {
    throw new Error('Failed to retrieve LLM status')
  }
  return res.json()
}

export async function saveAndVerifyLLMSettings(
  settings: LLMSettings,
): Promise<{
  success: boolean
  valid: boolean
  is_verified: boolean
  provider?: string
  model_name?: string
  message?: string
  error?: string
  latency_ms?: number
}> {
  const res = await apiFetch(`/settings/llm`, {
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
    const err = await res.json().catch(() => ({ detail: 'Failed to configure LLM' }))
    throw new Error(err.detail || 'Failed to configure LLM')
  }

  const data = await res.json()
  return {
    ...data,
    valid: Boolean(data.valid || data.success),
    success: Boolean(data.success || data.valid),
    is_verified: Boolean(data.is_verified),
    message: data.message || (data.success ? 'LLM connection verified' : data.error),
  }
}

export async function validateLLMSettings(
  settings: LLMSettings,
): Promise<{ valid: boolean; success: boolean; provider: string; message: string; error?: string }> {
  const res = await apiFetch(`/settings/validate`, {
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

  const data = await res.json()
  return {
    ...data,
    valid: Boolean(data.valid || data.success),
    success: Boolean(data.success || data.valid),
    message: data.message || data.error || (data.success ? 'Connection validated' : 'Validation failed'),
  }
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
  const token = getApiToken()
  const wsUrl = `${proto}//${host}/ws/v4/${sessionId}${token ? `?token=${encodeURIComponent(token)}` : ''}`

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

// -----------------------------------------------------------------------------
// Usage & cost accounting (G7)
// -----------------------------------------------------------------------------

export async function getUsage(conversationId?: string): Promise<UsageReport> {
  const query = conversationId ? `?conversation_id=${encodeURIComponent(conversationId)}` : ''
  const res = await apiFetch(`/usage${query}`)
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Usage lookup failed' }))
    throw new Error(err.detail || 'Usage lookup failed')
  }
  return res.json()
}

// -----------------------------------------------------------------------------
// Journey Coach
// -----------------------------------------------------------------------------

export async function fetchGuidance(
  conversationId: string,
  stage?: string,
): Promise<CoachListResponse> {
  const query = stage ? `?stage=${encodeURIComponent(stage)}` : ''
  const res = await apiFetch(`/conversations/${encodeURIComponent(conversationId)}/guidance${query}`)
  if (!res.ok) {
    await failFrom(res, 'Could not load coach findings')
  }
  return res.json()
}

export async function evaluateGuidance(
  conversationId: string,
  stage?: string,
  useLlm?: boolean,
): Promise<CoachReport> {
  const res = await apiFetch(
    `/conversations/${encodeURIComponent(conversationId)}/guidance/evaluate`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        stage: stage ?? null,
        use_llm: useLlm ?? null,
      }),
    },
  )
  if (!res.ok) {
    await failFrom(res, 'Could not evaluate coach findings')
  }
  return res.json()
}

export async function acknowledgeGuidance(
  conversationId: string,
  ruleId: string,
  fingerprint?: string,
): Promise<CoachListResponse> {
  const res = await apiFetch(
    `/conversations/${encodeURIComponent(conversationId)}/guidance/${encodeURIComponent(ruleId)}/ack`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ fingerprint: fingerprint ?? null }),
    },
  )
  if (!res.ok) {
    await failFrom(res, 'Could not acknowledge the finding')
  }
  return res.json()
}
