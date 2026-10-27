import {
  AdapterMeta,
  TaskConfig,
  EpisodeLog,
  RewardDiffProposal,
  ExportResponse,
  LLMSettings,
  BundleMetadata,
  BundleDetails,
} from '../types'

const API_BASE = '/api'

function formatLLMConfig(settings?: LLMSettings) {
  if (!settings) return undefined
  let apiKey: string | undefined
  let baseUrl: string | undefined
  if (settings.provider === 'deepseek') {
    apiKey = settings.deepseekApiKey || settings.openaiApiKey
    baseUrl = settings.deepseekBaseUrl || 'https://api.deepseek.com'
  } else if (settings.provider === 'openai') {
    apiKey = settings.openaiApiKey
  } else if (settings.provider === 'anthropic') {
    apiKey = settings.anthropicApiKey
  } else if (settings.provider === 'ollama') {
    baseUrl = settings.ollamaBaseUrl
  }
  return {
    provider: settings.provider,
    api_key: apiKey,
    base_url: baseUrl,
    model_name: settings.modelName || (settings.provider === 'deepseek' ? 'deepseek-chat' : undefined),
  }
}

export async function fetchAdapters(): Promise<AdapterMeta[]> {
  const res = await fetch(`${API_BASE}/adapters`)
  if (!res.ok) throw new Error('Failed to fetch adapters')
  const data = await res.json()
  return data.adapters
}

export async function validateLLMSettings(
  settings: LLMSettings,
): Promise<{ valid: boolean; message: string; provider: string }> {
  const llmConfig = formatLLMConfig(settings)
  const res = await fetch(`${API_BASE}/settings/validate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ llm_config: llmConfig }),
  })
  if (!res.ok) throw new Error('Validation request failed')
  return res.json()
}

export async function compileTask(
  prompt: string,
  adapterName: string,
  taskId?: string,
  llmSettings?: LLMSettings,
): Promise<{ status: string; task_config: TaskConfig; version: number }> {
  const payload: any = {
    prompt,
    adapter_name: adapterName,
    task_id: taskId,
  }
  const config = formatLLMConfig(llmSettings)
  if (config) payload.llm_config = config

  const res = await fetch(`${API_BASE}/tasks/compile`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) throw new Error('Task compilation failed')
  return res.json()
}

export async function runTrainingBurst(
  taskId: string,
  nEpisodes = 15,
): Promise<{ status: string; episodes: EpisodeLog[] }> {
  const res = await fetch(`${API_BASE}/train/burst`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ task_id: taskId, n_episodes: nEpisodes }),
  })
  if (!res.ok) throw new Error('Training burst failed')
  return res.json()
}

export async function toggleLockEpisode(
  episodeId: number,
): Promise<{ episode_id: number; locked: boolean; locked_episodes: number[] }> {
  const res = await fetch(`${API_BASE}/episodes/${episodeId}/lock`, {
    method: 'POST',
  })
  if (!res.ok) throw new Error('Toggle lock failed')
  return res.json()
}

export async function previewCritique(
  taskId: string,
  targetEpisodeId: number,
  critiqueText: string,
  llmSettings?: LLMSettings,
): Promise<RewardDiffProposal> {
  const payload: any = {
    task_id: taskId,
    target_episode_id: targetEpisodeId,
    critique_text: critiqueText,
  }
  const config = formatLLMConfig(llmSettings)
  if (config) payload.llm_config = config

  const res = await fetch(`${API_BASE}/critique/preview`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) throw new Error('Critique preview failed')
  return res.json()
}

export async function applyCritiqueDiff(
  taskId: string,
  newDsl: any,
  critiqueText: string,
  runBurstAfter = true,
): Promise<{ status: string; version?: number; version_tag: string; episodes_after?: number }> {
  const res = await fetch(`${API_BASE}/critique/apply`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      task_id: taskId,
      new_dsl: newDsl,
      critique_text: critiqueText,
      run_burst_after: runBurstAfter,
      burst_episodes: 15,
    }),
  })
  if (!res.ok) throw new Error('Failed to apply diff')
  return res.json()
}

export async function rollbackTaskVersion(
  taskId: string,
  targetVersion: number,
): Promise<{ status: string; restored_version: number; new_version_tag: string }> {
  const res = await fetch(`${API_BASE}/tasks/rollback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ task_id: taskId, target_version: targetVersion }),
  })
  if (!res.ok) throw new Error('Rollback failed')
  return res.json()
}

export async function exportPolicyBundle(
  taskId: string,
): Promise<ExportResponse> {
  const res = await fetch(`${API_BASE}/export`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ task_id: taskId }),
  })
  if (!res.ok) throw new Error('Export failed')
  return res.json()
}

// Bundle Registry API
export async function fetchBundles(): Promise<BundleMetadata[]> {
  const res = await fetch(`${API_BASE}/bundles`)
  if (!res.ok) throw new Error('Failed to fetch bundles')
  const data = await res.json()
  return data.bundles
}

export async function fetchBundleDetails(bundleId: string): Promise<BundleDetails> {
  const res = await fetch(`${API_BASE}/bundles/${bundleId}`)
  if (!res.ok) throw new Error(`Failed to fetch bundle details for ${bundleId}`)
  return res.json()
}

export async function importBundle(
  bundlePath: string,
  loadActive = false,
): Promise<{ status: string; task_config: TaskConfig; bundle_dir: string; is_active: boolean }> {
  const res = await fetch(`${API_BASE}/bundles/import`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ bundle_path: bundlePath, load_active: loadActive }),
  })
  if (!res.ok) throw new Error('Failed to import bundle')
  return res.json()
}

export async function deleteBundle(
  bundleId: string,
): Promise<{ status: string; bundle_id: string }> {
  const res = await fetch(`${API_BASE}/bundles/${bundleId}`, {
    method: 'DELETE',
  })
  if (!res.ok) throw new Error('Failed to delete bundle')
  return res.json()
}

// =============================================================================
// Phase 3D: Vibe Autopilot & Ingestion API
// =============================================================================

export async function createVibeSession(
  prompt: string,
  datasetId?: string,
  burstEpisodes = 15,
  llmSettings?: LLMSettings,
): Promise<any> {
  const payload: any = {
    prompt,
    dataset_id: datasetId,
    burst_episodes: burstEpisodes,
  }
  const config = formatLLMConfig(llmSettings)
  if (config) payload.llm_config = config

  const res = await fetch(`${API_BASE}/vibe/create`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || 'Failed to create Vibe session')
  }
  return res.json()
}

export async function coachVibeSession(
  sessionId: string,
  params: {
    critiqueText?: string
    feedbackType?: 'thumbs_up' | 'thumbs_down' | 'text' | 'rule_boost' | 'rule_reduce'
    targetEpisodeId?: number
    termName?: string
    newImportance?: string
    burstEpisodes?: number
    llmSettings?: LLMSettings
  },
): Promise<any> {
  const payload: any = {
    session_id: sessionId,
    critique_text: params.critiqueText,
    feedback_type: params.feedbackType,
    target_episode_id: params.targetEpisodeId,
    term_name: params.termName,
    new_importance: params.newImportance,
    burst_episodes: params.burstEpisodes || 10,
  }
  const config = formatLLMConfig(params.llmSettings)
  if (config) payload.llm_config = config

  const res = await fetch(`${API_BASE}/vibe/coach`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || 'Failed to apply coaching')
  }
  return res.json()
}

export async function interactSim(params: {
  sessionId?: string
  adapterName?: string
  boardState?: number[]
  humanCellIdx?: number
  robotState?: Record<string, number>
  targetXy?: [number, number]
  obstaclesList?: Array<{ x: number; y: number; radius: number }>
  nFrames?: number
}): Promise<any> {
  const payload: any = {
    session_id: params.sessionId,
    adapter_name: params.adapterName,
    board_state: params.boardState,
    human_cell_idx: params.humanCellIdx,
    robot_state: params.robotState,
    target_xy: params.targetXy,
    obstacles_list: params.obstaclesList,
    n_frames: params.nFrames || 4,
  }

  const res = await fetch(`${API_BASE}/sim/interact`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || 'Simulation interaction failed')
  }
  return res.json()
}

export async function uploadIngestFile(
  file: File,
  role = 'tabular',
  targetCol?: string,
): Promise<any> {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('role', role)
  if (targetCol) formData.append('target_col', targetCol)

  const res = await fetch(`${API_BASE}/ingest/upload`, {
    method: 'POST',
    body: formData,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || 'File ingestion failed')
  }
  return res.json()
}

export async function ingestSQL(
  connectionUri: string,
  queryOrTable: string,
  limit = 2000,
): Promise<any> {
  const res = await fetch(`${API_BASE}/ingest/sql`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      connection_uri: connectionUri,
      query_or_table: queryOrTable,
      limit,
    }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || 'SQL ingestion failed')
  }
  return res.json()
}
