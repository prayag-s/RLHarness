export interface AdapterMeta {
  name: string
  event_schema: Record<string, string>
  observations: {
    vector_dim: number
    feature_names: string[]
    horizon_type: string
    default_gamma: number
  }
  actions: {
    action_type: string
    labels: string[]
  }
}

export interface RewardTerm {
  name: string
  when: string | boolean
  value: number | string
}

export interface PinnedEpisode {
  episode_id: number
  forbid_action: string
  penalty: number
}

export interface RewardDSL {
  version: string
  horizon: {
    gamma: number
    critic_warmup_epochs_on_diff: number
    kl_coef: number
  }
  reward: {
    terms: RewardTerm[]
    pinned_episodes: PinnedEpisode[]
    aggregate: string
    clamp: [number, number]
  }
}

export interface TaskConfig {
  task_id: string
  name: string
  description: string
  adapter_name: string
  adapter_config: Record<string, any>
  hyperparameters: Record<string, any>
  reward_dsl: RewardDSL
}

export interface StepEvents {
  obs: Record<string, any>
  action: {
    action_idx: number
    label: string
  }
  state: Record<string, any>
  truth?: {
    has_label: boolean
    label: string
    true_anomaly?: boolean
  }
}

export interface StepResult {
  obs_vector: number[]
  events: StepEvents
  terminated: boolean
  truncated: boolean
  info: Record<string, any>
}

export interface EpisodeLog {
  episode_id: number
  steps: StepResult[]
  total_reward: number
  breakdown: Record<string, number>
  step_rewards: number[]
  step_breakdowns: Record<string, number>[]
  locked?: boolean
}

export interface TermDiff {
  action: 'added' | 'modified' | 'removed'
  term_name: string
  old_term?: Record<string, any>
  new_term?: Record<string, any>
  rationale: string
}

export interface ConflictWarning {
  term_name: string
  message: string
  severity: string
}

export interface Stage1Preview {
  target_episode_id: number
  before_score: number
  after_score: number
  score_delta: number
  recent_scores_before: number[]
  recent_scores_after: number[]
  summary: string
}

export interface Stage2Preview {
  target_episode_id: number
  target_action_before: string
  target_action_after: string
  target_flipped: boolean
  total_episodes_evaluated: number
  total_actions_flipped: number
  flip_rate: number
  locked_episodes_preserved: number
  summary: string
}

export interface RewardDiffProposal {
  new_dsl: RewardDSL
  diff_yaml: string
  term_diffs: TermDiff[]
  rationale: Record<string, string>
  conflicts: ConflictWarning[]
  stage1_counterfactual_preview?: Stage1Preview
  stage2_forked_behavior_preview?: Stage2Preview
}

export interface ExportResponse {
  status: string
  bundle_dir: string
  zip_path: string
  parity_max_diff: number
  parity_passed: boolean
}

export interface VersionRecord {
  version: number
  timestamp: string
  description?: string
  critique_text?: string
}

export interface DriftReport {
  is_drifted: boolean
  max_psi: number
  entropy_zscore: number
  action_shift_zscore: number
  recommendation: string
  feature_psis?: Record<string, number>
}

export interface LLMSettings {
  provider: 'mock' | 'deepseek' | 'openai' | 'anthropic' | 'ollama'
  deepseekApiKey?: string
  deepseekBaseUrl?: string
  openaiApiKey: string
  anthropicApiKey: string
  ollamaBaseUrl: string
  modelName: string
}

export interface BundleMetadata {
  bundle_id: string
  name: string
  task_id: string
  adapter_name: string
  archive_type: 'dir' | 'zip'
  bundle_path: string
  created_at: string
  onnx_size_bytes: number
  dsl_terms_count: number
  feature_count: number
  baseline_quantiles_count: number
  status: 'ready' | 'corrupt'
  error_message?: string
}

export interface BundleDetails {
  metadata: BundleMetadata
  task_config: Record<string, any>
  reward_dsl: Record<string, any>
  drift_baseline: Record<string, any>
  onnx_summary: Record<string, any>
  has_run_local_daemon: boolean
}


