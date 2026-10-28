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
  pillar: string
  question: string
  suggested_options: string[]
  default_option: string
  question_id: string
}

export interface IntakeResponse {
  status: 'needs_clarification' | 'ready_to_build'
  assistant_message: string
  message: string
  questions: ClarifyingQuestion[]
  clarifying_questions: ClarifyingQuestion[]
  blueprint_spec?: AgentBlueprintSpec | null
  blueprint?: AgentBlueprintSpec | null
}

export interface EpisodeLog {
  episode_id: number
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
  latest_frame?: Record<string, any>
  total_episodes_trained?: number
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

export interface ActivityTimelineItem {
  id: string
  type: 'synthesis' | 'smoke_test' | 'arena' | 'round' | 'auto_tune' | 'mastery' | 'nudge' | 'error' | 'info'
  title: string
  description: string
  timestamp: string
  status: 'pending' | 'success' | 'warning' | 'info'
  data?: any
}
