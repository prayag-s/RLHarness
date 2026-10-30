// =============================================================================
// RL-Native Agent Studio v4.0 Type Definitions
// Autonomous JIT Agent Synthesis, Conversational Architect & Closed-Loop Training
// =============================================================================

export interface ObsSpec {
  vector_dim: number
  feature_names: string[]
  horizon_type?: string
  default_gamma?: number
}

export interface ActionSpec {
  action_type: string
  labels: string[]
}

export interface RewardTermSpec {
  name: string
  weight: number
  description: string
}

export interface RewardSpec {
  terms: RewardTermSpec[]
  target_metric_name: string
  target_metric_threshold: number
}

export interface ArenaElementSpec {
  name: string
  type: string
  description: string
}

export interface ArenaSpec {
  arena_type: string
  title: string
  theme: string
  elements: ArenaElementSpec[]
}

export interface InteractiveControlSpec {
  name: string
  type: string
  label: string
}

export interface InteractiveSpec {
  enabled: boolean
  controls: InteractiveControlSpec[]
}

export interface AgentBlueprintSpec {
  task_title: string
  domain_summary: string
  data_source_type: string
  dataset_id?: string | null
  observation_features: string[]
  action_labels: string[]
  arena_visual_description: string
  interactive_mode_description: string
  success_metric_name: string
  target_metric_threshold: number
  max_auto_tune_rounds?: number
  initial_reward_components?: Array<Record<string, any>>

  // Legacy / UI aliases
  task_id?: string
  objective?: string
  data_source?: string
  target_metric_name?: string
  obs_spec?: ObsSpec
  action_spec?: ActionSpec
  reward_spec?: RewardSpec
  arena_spec?: ArenaSpec
  interactive_spec?: InteractiveSpec
}

export interface ClarifyingQuestion {
  id?: string
  pillar: string
  question: string
  suggested_options: string[]
  default_option: string
  question_id: string
}

export interface TrainingParams {
  max_auto_tune_rounds?: number
  episodes_per_round?: number
  target_metric_threshold?: number
}

export interface IntakeResponse {
  status: 'needs_clarification' | 'ready_to_build'
  assistant_message: string
  message: string
  conversation_id?: string | null
  questions: ClarifyingQuestion[]
  clarifying_questions: ClarifyingQuestion[]
  blueprint_spec?: AgentBlueprintSpec | null
  blueprint?: AgentBlueprintSpec | null
  recommended_training_params?: TrainingParams
}

export interface EpisodeLog {
  episode_id: string | number
  round?: number
  total_reward: number
  reward_breakdown: Record<string, number>
  is_success: boolean
  steps?: any[]
}

export interface RoundSnapshot {
  round_num: number
  metric: number
  target_threshold: number
  target_achieved: boolean
  env_code_snapshot: string
  diagnosis?: string | null
  changes_summary?: string | null
  avg_return?: number
  term_rewards?: Record<string, number>
  episodes?: EpisodeLog[]
}

export interface TuningResult {
  round_num: number
  diagnosis: string
  changes_summary: string
  updated_code: string
  verified: boolean
}

export interface SessionState {
  session_id: string
  blueprint: AgentBlueprintSpec
  dataset_id?: string | null
  status: 'synthesized' | 'training' | 'completed' | 'error' | string
  current_round: number
  max_rounds: number
  target_threshold: number
  current_metric: number
  drift_status?: 'healthy' | 'drifted' | string
  env_module_code?: string
  arena_html?: string
  round_history?: RoundSnapshot[]
  rounds_history?: RoundSnapshot[]
  episodes?: EpisodeLog[]
  latest_frame?: Record<string, any>
  total_episodes_trained?: number
  scorecard?: Scorecard | null
}

export type StudioActionType =
  | 'clarify_or_discuss'
  | 'update_arena'
  | 'update_agent'
  | 'update_env_mechanics'
  | 'run_training'

export interface StudioChatResponse {
  conversation_id: string
  status: string
  assistant_message: string
  message: string
  chat_response?: string
  action_types: StudioActionType[]
  action_type?: StudioActionType
  blueprint_spec?: AgentBlueprintSpec | null
  blueprint?: AgentBlueprintSpec | null
  arena_html?: string | null
  env_code?: string | null
  arena_locked: boolean
  questions: ClarifyingQuestion[]
  clarifying_questions: ClarifyingQuestion[]
  quick_chips: string[]
  training_params: TrainingParams
  current_round: number
  best_metric: number
  current_metric?: number
}

export interface ScorecardBaseline {
  name: string
  value_pct: number
  detail?: string | null
  n?: number | null
}

export interface Scorecard {
  version: number
  task_title: string
  task_class: string
  metric_name: string
  held_out: boolean
  eval_split: string
  split_counts: Record<string, number>
  policy: {
    value_pct: number
    episodes: number
    ci95_pct: [number, number] | null
    illegal_actions_pct: number
  }
  baselines: ScorecardBaseline[]
  uplift_vs_best_baseline_pct: number
  beats_all_baselines: boolean
  rows: { total: number | null; embedded_in_module: number | null }
  notes: string[]
  deployment?: {
    action_agreement_pct?: number
    policy_success_pct?: number
    onnx_success_pct?: number
    onnx_numeric_parity_max_diff?: number
    error?: string
  }
}

export interface ConversationSummary {
  conversation_id: string
  title: string
  status: string
  best_metric: number
  target_metric_threshold: number
  current_round: number
  max_auto_tune_rounds: number
  arena_locked?: boolean
  updated_at: string
}

export interface ConversationDetail {
  conversation_id: string
  title: string
  status: string
  created_at: string
  updated_at: string
  chat_history: any[]
  dataset_id?: string | null
  blueprint_spec?: AgentBlueprintSpec | null
  training_params?: TrainingParams
  current_round: number
  best_metric: number
  rounds_history: RoundSnapshot[]
  episodes: EpisodeLog[]
  env_code?: string | null
  arena_html?: string | null
  latest_model_zip_path?: string | null
  arena_locked?: boolean
  scorecard?: Scorecard | null
}

export interface IngestedDatasetSummary {
  dataset_id: string
  name: string
  format: string
  source_type?: string
  role?: string
  columns: string[]
  num_rows: number
  extracted_rules_count?: number
}

export interface BundleMetadata {
  bundle_id: string
  name: string
  task_id: string
  adapter_name?: string
  archive_type: 'dir' | 'zip' | string
  bundle_path: string
  created_at: string
  onnx_size_bytes: number
  dsl_terms_count?: number
  feature_count?: number
  status: 'ready' | 'corrupt' | string
  error_message?: string
}

export interface LLMSettings {
  provider: 'deepseek' | 'openai' | 'anthropic' | 'ollama' | 'fallback'
  apiKey?: string
  baseUrl?: string
  modelName?: string
}

export interface LLMStatusResponse {
  is_verified: boolean
  provider: string
  model_name: string
  base_url?: string
  has_api_key: boolean
  masked_key?: string
  error?: string | null
  latency_ms?: number
  last_verified_at?: number | null
}

export interface ActivityTimelineItem {
  id: string
  type: 'synthesis' | 'smoke_test' | 'arena' | 'round' | 'auto_tune' | 'mastery' | 'nudge' | 'error' | 'info'
  title: string
  description: string
  timestamp: string
  status: 'pending' | 'success' | 'warning' | 'info'
  data?: any
}

export interface UsageReport {
  totals: {
    calls: number
    input_tokens: number
    output_tokens: number
    cost_usd: number
  }
  by_model: Array<{
    provider: string
    model: string
    calls: number
    input_tokens: number
    output_tokens: number
    cost_usd: number
  }>
  recent: Array<{
    conversation_id: string | null
    provider: string
    model: string
    input_tokens: number
    output_tokens: number
    cost_usd: number
    latency_ms: number
    created_at: string
  }>
  budget: {
    cap_usd: number | null
    spent_usd: number
    remaining_usd: number | null
  }
}
