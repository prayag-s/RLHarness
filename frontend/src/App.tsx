import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Header } from './components/Header'
import { ScoreboardRail } from './components/ScoreboardRail'
import { ThreadView } from './components/ThreadView'
import { TurnComposer } from './components/TurnComposer'
import { ArenaStage } from './components/ArenaStage'
import { SettingsModal } from './components/SettingsModal'
import { LLMVerificationGateModal } from './components/LLMVerificationGateModal'
import { RegistryModal } from './components/RegistryModal'
import { CodeInspectorModal } from './components/CodeInspectorModal'
import { SQLIngestModal } from './components/SQLIngestModal'
import { ConversationSidebar } from './components/ConversationSidebar'
import { TrustPanel } from './components/TrustPanel'

import {
  SessionState,
  RoundSnapshot,
  EpisodeLog,
  LLMSettings,
  IngestedDatasetSummary,
  ConversationSummary,
  TrainingParams,
  StudioChatResponse,
  Scorecard,
  ArenaSource,
  DriftSnapshot,
  ObservedStep,
  CoachAction,
  CoachItem,
  ThreadEntry,
  TurnStatus,
} from './types'

import {
  ApiError,
  chatWithStudioOmni,
  getUsage,
  updateArenaLook,
  rollbackSession,
  exportSessionPolicy,
  fetchBundles,
  uploadDatasetFile,
  connectTelemetryWebSocket,
  getLLMStatus,
  saveAndVerifyLLMSettings,
  fetchConversations,
  fetchConversation,
  deleteConversation,
  clearAllConversations,
  acknowledgeGuidance,
  fetchThread,
  stopTurn,
} from './services/api'

// The field guide (Markdown + renderer) is only needed when opened, so keep it out of the
// initial bundle.
const GuideModal = React.lazy(() =>
  import('./components/GuideModal').then((m) => ({ default: m.GuideModal })),
)

/** True when the backend refused because no verified model is configured (G13). */
function isModelRequired(err: unknown): boolean {
  return err instanceof ApiError && err.needsModel
}

const THREAD_POLL_MS = 1500
const WS_SYNC_DEBOUNCE_MS = 300

/**
 * One chat, one clock.
 *
 * Every event — your instruction, the studio's plan, the arena, the coach, each training round,
 * every tuning decision — arrives in a single ordered thread, and only one party may append to it
 * at a time. Numbers live in the scoreboard beside it; see UNIFIED_CHAT_UX_SPEC_AND_PLAN.md.
 */
export const App: React.FC = () => {
  // Session & training state (the numbers the scoreboard renders)
  const [activeSession, setActiveSession] = useState<SessionState | null>(null)
  const [isTraining, setIsTraining] = useState(false)
  const [isExporting, setIsExporting] = useState(false)
  const [isRollingBack, setIsRollingBack] = useState(false)

  // Arena & code
  const [arenaHtml, setArenaHtml] = useState<string>('')
  const [arenaLocked, setArenaLocked] = useState<boolean>(false)
  const [arenaSource, setArenaSource] = useState<ArenaSource | null>(null)
  const [arenaPending, setArenaPending] = useState<boolean>(false)
  const [arenaError, setArenaError] = useState<string | null>(null)
  const [arenaStage, setArenaStage] = useState<string>('')
  const [arenaPulse, setArenaPulse] = useState<number>(0)
  const [envModuleCode, setEnvModuleCode] = useState<string>('')
  const [latestFrame, setLatestFrame] = useState<Record<string, any> | undefined>(undefined)
  const [episodes, setEpisodes] = useState<EpisodeLog[]>([])

  // Metrics
  const [currentMetric, setCurrentMetric] = useState<number>(0)
  const [targetThreshold, setTargetThreshold] = useState<number>(85)
  const [targetMetricName, setTargetMetricName] = useState<string>('Success / Win Rate')
  const [currentRound, setCurrentRound] = useState<number>(1)
  const [maxRounds, setMaxRounds] = useState<number>(4)
  const [roundHistory, setRoundHistory] = useState<RoundSnapshot[]>([])
  const [scorecard, setScorecard] = useState<Scorecard | null>(null)
  const [driftStatus, setDriftStatus] = useState<string>('unmonitored')
  const [driftDetail, setDriftDetail] = useState<string>('')
  const [driftSamples, setDriftSamples] = useState<number>(0)

  // Dataset
  const [attachedDataset, setAttachedDataset] = useState<IngestedDatasetSummary | null>(null)

  // The thread: the only place work is narrated
  const [threadEntries, setThreadEntries] = useState<ThreadEntry[]>([])
  const [turn, setTurn] = useState<TurnStatus | null>(null)
  const [pendingEntries, setPendingEntries] = useState<ThreadEntry[]>([])
  const [notices, setNotices] = useState<ThreadEntry[]>([])
  const [queuedMessage, setQueuedMessage] = useState<string | null>(null)
  const [isSending, setIsSending] = useState(false)
  const [isStopping, setIsStopping] = useState(false)
  const [focusedSeq, setFocusedSeq] = useState<number | null>(null)
  const [isLoadingThread, setIsLoadingThread] = useState(false)

  // Conversation persistence
  const [isConversationsOpen, setIsConversationsOpen] = useState(false)
  const [conversations, setConversations] = useState<ConversationSummary[]>([])
  const [activeConversationId, setActiveConversationId] = useState<string | undefined>(undefined)
  const [isLoadingConversations, setIsLoadingConversations] = useState(false)
  const [trainingParams, setTrainingParams] = useState<TrainingParams>({
    max_auto_tune_rounds: 5,
    episodes_per_round: 200,
    target_metric_threshold: 90,
  })

  // Drawers & modals
  const [isTrustOpen, setIsTrustOpen] = useState(false)
  const [isGuideOpen, setIsGuideOpen] = useState(false)
  const [guideSection, setGuideSection] = useState<string | null>(null)
  const [isSettingsOpen, setIsSettingsOpen] = useState(false)
  const [isRegistryOpen, setIsRegistryOpen] = useState(false)
  const [isCodeInspectorOpen, setIsCodeInspectorOpen] = useState(false)
  const [isSQLModalOpen, setIsSQLModalOpen] = useState(false)
  const [registryCount, setRegistryCount] = useState(0)
  const [usage, setUsage] = useState<{ calls: number; cost_usd: number }>({ calls: 0, cost_usd: 0 })
  const [lastStep, setLastStep] = useState<ObservedStep | null>(null)
  const [trustRefreshKey, setTrustRefreshKey] = useState(0)

  // Mandatory LLM gate
  const [isLLMVerified, setIsLLMVerified] = useState<boolean>(false)
  const [isCheckingLLM, setIsCheckingLLM] = useState<boolean>(true)
  const [isGateModalOpen, setIsGateModalOpen] = useState<boolean>(false)
  const [llmSettings, setLlmSettings] = useState<LLMSettings>(() => {
    const saved = localStorage.getItem('rl_ide_v4_llm_settings')
    if (saved) {
      try {
        return JSON.parse(saved)
      } catch {
        // fall through to defaults
      }
    }
    return {
      provider: 'deepseek',
      baseUrl: 'https://api.deepseek.com/v1',
      modelName: 'deepseek-chat',
      apiKey: '',
    }
  })

  const wsRef = useRef<WebSocket | null>(null)
  const seqRef = useRef(0)
  const activeConversationRef = useRef<string | undefined>(undefined)
  const syncTimerRef = useRef<number | null>(null)
  const chatInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    seqRef.current = threadEntries.length ? threadEntries[threadEntries.length - 1].seq : 0
  }, [threadEntries])

  useEffect(() => {
    activeConversationRef.current = activeConversationId
  }, [activeConversationId])

  const pushNotice = useCallback((title: string, body: string, status: ThreadEntry['status']) => {
    setNotices((prev) => [
      ...prev,
      {
        seq: Number.MAX_SAFE_INTEGER - prev.length,
        turn_id: 'notice',
        actor: 'system',
        kind: 'error',
        status,
        title,
        body,
        at: new Date().toISOString(),
      },
    ])
  }, [])

  const clearNotices = useCallback(() => setNotices([]), [])

  // ---------------------------------------------------------------- thread I/O

  const syncThread = useCallback(
    async (conversationId: string, opts: { reset?: boolean } = {}) => {
      const since = opts.reset ? 0 : seqRef.current
      const data = await fetchThread(conversationId, since)
      setThreadEntries((prev) => {
        if (opts.reset) return data.entries
        const seen = new Set(prev.map((e) => e.seq))
        const appended = data.entries.filter((e) => !seen.has(e.seq))
        return appended.length ? [...prev, ...appended] : prev
      })
      setTurn(data.turn)
      return data.turn
    },
    [],
  )

  const scheduleSync = useCallback(
    (conversationId: string) => {
      if (syncTimerRef.current) window.clearTimeout(syncTimerRef.current)
      syncTimerRef.current = window.setTimeout(() => {
        void syncThread(conversationId).catch(() => undefined)
      }, WS_SYNC_DEBOUNCE_MS)
    },
    [syncThread],
  )

  // ------------------------------------------------------------- conversations

  const loadConversationsList = useCallback(async () => {
    try {
      setIsLoadingConversations(true)
      setConversations(await fetchConversations())
    } catch (e) {
      console.warn('Failed to load conversations', e)
    } finally {
      setIsLoadingConversations(false)
    }
  }, [])

  const refreshUsage = useCallback(async (cid: string | null | undefined) => {
    try {
      const report = await getUsage(cid || undefined)
      setUsage({ calls: report.totals.calls, cost_usd: report.totals.cost_usd })
    } catch {
      // usage is informational; never block the studio on it
    }
  }, [])

  useEffect(() => {
    void loadConversationsList()
  }, [loadConversationsList])

  useEffect(() => {
    void refreshUsage(activeConversationId)
  }, [activeConversationId, refreshUsage])

  useEffect(() => {
    fetchBundles()
      .then((bundles) => setRegistryCount(bundles.length))
      .catch(() => undefined)
  }, [isRegistryOpen])

  // -------------------------------------------------------------- LLM startup

  useEffect(() => {
    let isMounted = true
    async function verifyStartupCredentials() {
      setIsCheckingLLM(true)
      try {
        const status = await getLLMStatus()
        if (status.is_verified) {
          if (isMounted) {
            setIsLLMVerified(true)
            setIsGateModalOpen(false)
            setLlmSettings((prev) => ({
              ...prev,
              provider: (status.provider as any) || prev.provider,
              modelName: status.model_name || prev.modelName,
              baseUrl: status.base_url || prev.baseUrl,
            }))
          }
          return
        }

        const saved = localStorage.getItem('rl_ide_v4_llm_settings')
        if (saved) {
          try {
            const parsed: LLMSettings = JSON.parse(saved)
            if (parsed.apiKey && parsed.provider !== 'fallback') {
              const verifyRes = await saveAndVerifyLLMSettings(parsed)
              if (verifyRes.is_verified || verifyRes.success) {
                if (isMounted) {
                  setIsLLMVerified(true)
                  setIsGateModalOpen(false)
                  setLlmSettings(parsed)
                }
                return
              }
            }
          } catch {
            // Credentials failed verification; stay in explore mode.
          }
        }

        if (isMounted) {
          setIsLLMVerified(false)
          setIsGateModalOpen(false)
        }
      } catch {
        if (isMounted) {
          setIsLLMVerified(false)
          setIsGateModalOpen(false)
        }
      } finally {
        if (isMounted) setIsCheckingLLM(false)
      }
    }

    void verifyStartupCredentials()
    return () => {
      isMounted = false
    }
  }, [])

  useEffect(() => {
    const onAuthRequired = () => {
      setIsSettingsOpen(true)
      pushNotice(
        'Authentication required',
        'This deployment needs a tenant token. Add it under Settings → Tenant token.',
        'warning',
      )
    }
    window.addEventListener('rl:auth-required', onAuthRequired)
    return () => window.removeEventListener('rl:auth-required', onAuthRequired)
  }, [pushNotice])

  // ------------------------------------------------------------------ metrics

  const applyDrift = useCallback((snapshot?: DriftSnapshot | null, status?: string) => {
    const next = status || snapshot?.status
    if (next) setDriftStatus(next)
    if (snapshot) {
      setDriftSamples(snapshot.samples_seen ?? 0)
      const detail =
        snapshot.reasons && snapshot.reasons.length > 0
          ? snapshot.reasons.join(' · ')
          : snapshot.recommendation || ''
      setDriftDetail(detail)
    }
  }, [])

  // ---------------------------------------------------------------- websocket

  const handleWebSocketEvent = useCallback(
    (event: any) => {
      const eventType = event.event
      const cid = activeConversationRef.current

      if (eventType === 'episode_step') {
        setLatestFrame(event)
        if (event.episode_id !== undefined) {
          setEpisodes((prev) => [
            ...prev.slice(-49),
            {
              episode_id: event.episode_id,
              total_reward: event.total_reward || 0,
              reward_breakdown: event.reward_breakdown || {},
              is_success: !!event.is_success,
            },
          ])
        }
        return
      }

      if (eventType === 'drift_warning') {
        applyDrift(event.drift as DriftSnapshot, 'drifted')
      } else if (eventType === 'scorecard_ready') {
        setScorecard(event.scorecard || null)
        setTrustRefreshKey((k) => k + 1)
      } else if (eventType === 'arena_updated') {
        if (event.arena_html) setArenaHtml(event.arena_html)
        if (event.arena_kind) setArenaSource(event.arena_kind as ArenaSource)
        setArenaPending(Boolean(event.pending_bespoke))
        setArenaError(event.error || (event.arena_kind === 'fallback' ? 'Bespoke canvas failed.' : null))
        setArenaPulse((k) => k + 1)
      } else if (eventType === 'round_started') {
        setCurrentRound(event.round ?? event.round_num ?? 1)
      } else if (eventType === 'round_evaluated') {
        const rNum = event.round ?? event.round_num ?? 1
        const metric = event.metric ?? event.current_metric ?? 0
        const target = event.target ?? event.target_threshold ?? 85
        setCurrentMetric(metric)
        setRoundHistory((prev) => [
          ...prev,
          {
            round_num: rNum,
            metric,
            target_threshold: target,
            target_achieved: metric >= target,
            env_code_snapshot: '',
            avg_return: event.avg_return,
            term_rewards: event.reward_breakdown || event.term_rewards,
          },
        ])
      } else if (eventType === 'auto_tuned') {
        const diagnosis = event.diagnosis || event.tuning_result?.diagnosis || 'Auto-tuned reward function.'
        const changesSummary = event.changes_summary || event.tuning_result?.changes_summary || ''
        setRoundHistory((prev) => {
          const updated = [...prev]
          const lastIdx = updated.length - 1
          if (lastIdx >= 0) {
            updated[lastIdx] = { ...updated[lastIdx], diagnosis, changes_summary: changesSummary }
          }
          return updated
        })
        if (event.tuning_result?.updated_code) setEnvModuleCode(event.tuning_result.updated_code)
      } else if (eventType === 'target_achieved') {
        setIsTraining(false)
        setCurrentMetric(event.metric ?? event.current_metric ?? 100)
      } else if (eventType === 'budget_exhausted') {
        setIsTraining(false)
        setCurrentMetric(event.metric ?? event.current_metric ?? 0)
      } else if (eventType === 'training_error' || eventType === 'error') {
        setIsTraining(false)
      } else if (eventType === 'turn_state' && event.turn) {
        setTurn(event.turn as TurnStatus)
      }

      if (cid) scheduleSync(cid)
    },
    [applyDrift, scheduleSync],
  )

  const connectWebSocket = useCallback(
    (sessionId: string) => {
      if (wsRef.current) wsRef.current.close()
      wsRef.current = connectTelemetryWebSocket(
        sessionId,
        handleWebSocketEvent,
        (err) => console.warn('WebSocket error:', err),
        () => undefined,
      )
    },
    [handleWebSocketEvent],
  )

  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close()
        wsRef.current = null
      }
      if (syncTimerRef.current) window.clearTimeout(syncTimerRef.current)
    }
  }, [])

  // Poll only while another party owns the turn — that is what makes it turn-by-turn.
  useEffect(() => {
    if (!activeConversationId || turn?.state !== 'running') return
    const id = window.setInterval(() => {
      void syncThread(activeConversationId).catch(() => undefined)
    }, THREAD_POLL_MS)
    return () => window.clearInterval(id)
  }, [activeConversationId, turn?.state, syncThread])

  // ------------------------------------------------------------------ sending

  const applyStudioResponse = useCallback(
    (resp: StudioChatResponse, cid: string) => {
      if (resp.arena_html) setArenaHtml(resp.arena_html)
      if (resp.arena_source) setArenaSource(resp.arena_source)
      setArenaPending(false)
      setArenaError(resp.arena_source === 'fallback' ? 'Fell back to the schema-driven preview.' : null)
      if (resp.arena_locked !== undefined) setArenaLocked(resp.arena_locked)
      if (resp.env_code) setEnvModuleCode(resp.env_code)
      if (resp.best_metric !== undefined || resp.current_metric !== undefined) {
        setCurrentMetric(resp.best_metric !== undefined ? resp.best_metric : resp.current_metric || 0)
      }
      if (resp.current_round !== undefined) setCurrentRound(resp.current_round)
      if (resp.training_params) {
        setTrainingParams(resp.training_params)
        if (resp.training_params.max_auto_tune_rounds)
          setMaxRounds(resp.training_params.max_auto_tune_rounds)
        if (resp.training_params.target_metric_threshold)
          setTargetThreshold(resp.training_params.target_metric_threshold)
      }

      if (resp.blueprint) {
        const effectiveMetric =
          resp.best_metric !== undefined ? resp.best_metric : resp.current_metric || 0
        setTargetThreshold(resp.blueprint.target_metric_threshold || 85)
        setTargetMetricName(resp.blueprint.target_metric_name || 'Success Rate')
        setMaxRounds(resp.blueprint.max_auto_tune_rounds || 4)
        setActiveSession((prev) => ({
          session_id: cid,
          blueprint: resp.blueprint!,
          dataset_id: attachedDataset?.dataset_id,
          status: resp.status,
          current_round: resp.current_round ?? prev?.current_round ?? 1,
          max_rounds: resp.blueprint?.max_auto_tune_rounds || prev?.max_rounds || 4,
          target_threshold: resp.blueprint?.target_metric_threshold || prev?.target_threshold || 85,
          current_metric: effectiveMetric,
          env_module_code: resp.env_code || prev?.env_module_code || '',
          arena_html: resp.arena_html || prev?.arena_html || '',
          rounds_history: prev?.rounds_history || [],
          episodes: prev?.episodes || [],
        }))
      }

      if (resp.status === 'training') {
        setIsTraining(true)
        connectWebSocket(cid)
      } else if (resp.status === 'completed' || resp.status === 'target_achieved') {
        setIsTraining(false)
      }
    },
    [attachedDataset?.dataset_id, connectWebSocket],
  )

  const sendTurn = useCallback(
    async (message: string) => {
      const trimmed = message.trim()
      if (!trimmed || isSending) return
      if (!isLLMVerified) {
        setIsGateModalOpen(true)
        return
      }

      const cid =
        activeConversationId || `conv_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`
      if (!activeConversationId) setActiveConversationId(cid)

      clearNotices()
      setIsSending(true)
      setQueuedMessage(null)
      setPendingEntries([
        {
          seq: Number.MAX_SAFE_INTEGER - 1,
          turn_id: turn?.turn_id || 'local',
          actor: 'user',
          kind: 'instruction',
          status: 'success',
          title: 'You',
          body: trimmed,
          at: new Date().toISOString(),
        },
      ])

      try {
        const resp = await chatWithStudioOmni(
          {
            conversation_id: cid,
            user_message: trimmed,
            dataset_id: attachedDataset?.dataset_id,
            selected_options: {},
            training_params: trainingParams,
          },
          (stage, detail) => {
            setArenaStage(detail ? `${stage}: ${detail}` : stage)
            if (stage === 'arena_preview' || stage === 'arena') {
              setArenaPending(true)
              setArenaError(null)
            }
            // The server records each stage in the thread; pull it in as it happens.
            scheduleSync(cid)
          },
        )
        applyStudioResponse(resp, cid)
        await syncThread(cid, { reset: true })
        void loadConversationsList()
        void refreshUsage(cid)
      } catch (err: any) {
        if (isModelRequired(err)) {
          setIsLLMVerified(false)
          setIsGateModalOpen(true)
        }
        // A refused turn is information, not a failure: show who owns the conversation.
        await syncThread(cid, { reset: true }).catch(() => undefined)
        const cancelled = String(err?.message || '').toLowerCase().includes('cancel')
        if (!cancelled) {
          const busy = err?.status === 409 || /mid-turn/i.test(String(err?.message || ''))
          pushNotice(
            busy ? 'Not your turn' : 'Turn failed',
            err?.message || 'The studio could not complete this turn.',
            'warning',
          )
        }
      } finally {
        setIsSending(false)
        setPendingEntries([])
      }
    },
    [
      activeConversationId,
      applyStudioResponse,
      attachedDataset?.dataset_id,
      clearNotices,
      isLLMVerified,
      isSending,
      loadConversationsList,
      pushNotice,
      refreshUsage,
      scheduleSync,
      syncThread,
      trainingParams,
      turn?.turn_id,
    ],
  )

  // Training is a turn like any other: when the owner hands it back, the run is over.
  useEffect(() => {
    if (turn && turn.state !== 'running') setIsTraining(false)
  }, [turn?.state])

  // A queued message is the next turn, and is only sent once the owner hands the turn back.
  useEffect(() => {
    if (!queuedMessage || isSending) return
    if (turn?.state === 'running') return
    const message = queuedMessage
    setQueuedMessage(null)
    void sendTurn(message)
  }, [queuedMessage, isSending, turn?.state, sendTurn])

  /**
   * Anything that wants to say something to the studio goes through here: if another party owns
   * the turn, the message is queued rather than injected into their turn.
   */
  const submitInstruction = useCallback(
    (message: string) => {
      if (!message.trim()) return
      if (isSending || turn?.state === 'running') {
        setQueuedMessage(message)
        return
      }
      void sendTurn(message)
    },
    [isSending, sendTurn, turn?.state],
  )

  const handleStop = useCallback(async () => {
    const cid = activeConversationRef.current
    if (!cid || isStopping) return
    setIsStopping(true)
    try {
      const res = await stopTurn(cid)
      setTurn(res.turn)
      setIsTraining(false)
      await syncThread(cid)
    } catch (err: any) {
      pushNotice('Could not stop', err?.message || 'Stop failed.', 'warning')
    } finally {
      setIsStopping(false)
    }
  }, [isStopping, pushNotice, syncThread])

  // ------------------------------------------------------------ coach & trust

  const handleCoachAction = useCallback(
    async (action: CoachAction, item: CoachItem) => {
      void item
      if (action.kind === 'apply_params' || action.kind === 'run_episodes') {
        const patch = action.payload || {}
        setTrainingParams((prev) => ({
          ...prev,
          ...(patch.max_auto_tune_rounds !== undefined
            ? { max_auto_tune_rounds: Number(patch.max_auto_tune_rounds) }
            : {}),
          ...(patch.episodes_per_round !== undefined
            ? { episodes_per_round: Number(patch.episodes_per_round) }
            : {}),
          ...(patch.target_metric_threshold !== undefined
            ? { target_metric_threshold: Number(patch.target_metric_threshold) }
            : {}),
        }))
        pushNotice('Coach recommendation applied', action.label, 'success')
        return
      }

      if (action.kind === 'open_guide') {
        setGuideSection(String(action.payload?.section || '') || null)
        setIsGuideOpen(true)
        return
      }

      const message =
        action.kind === 'use_objective'
          ? String(action.payload?.objective || '')
          : String(action.payload?.message || '')
      if (!message) return
      submitInstruction(message)
    },
    [pushNotice, submitInstruction],
  )

  const handleAcknowledgeCoach = useCallback(
    async (item: CoachItem) => {
      const cid = activeConversationRef.current
      if (!cid) return
      try {
        await acknowledgeGuidance(cid, item.rule_id, item.fingerprint)
        setThreadEntries((prev) =>
          prev.map((entry) =>
            entry.kind === 'finding' && entry.data?.rule_id === item.rule_id
              ? {
                  ...entry,
                  data: { ...entry.data, item: { ...entry.data.item, status: 'acknowledged' } },
                }
              : entry,
          ),
        )
      } catch (err: any) {
        pushNotice(
          'Acknowledge failed',
          err?.message || 'Could not acknowledge the finding.',
          'warning',
        )
      }
    },
    [pushNotice],
  )

  // ---------------------------------------------------------------- restoring

  const handleSelectConversation = useCallback(
    async (cid: string) => {
      try {
        setIsLoadingThread(true)
        const conv = await fetchConversation(cid)
        setActiveConversationId(cid)
        setArenaHtml(conv.arena_html || '')
        setArenaLocked(conv.arena_locked ?? false)
        setArenaSource((conv.arena_source as ArenaSource) || null)
        setArenaError(null)
        const job = conv.active_job
        setArenaPending(
          Boolean(
            job &&
              (job.stage === 'env_synthesis' ||
                job.stage === 'arena' ||
                job.stage === 'verification'),
          ),
        )
        setArenaStage(job?.stage || '')
        setEnvModuleCode(conv.env_code || '')
        setEpisodes(conv.episodes || [])
        setScorecard(conv.scorecard || null)
        applyDrift(conv.drift, conv.drift_status)
        if (conv.best_metric !== undefined) setCurrentMetric(conv.best_metric)
        if (conv.current_round !== undefined) setCurrentRound(conv.current_round)
        if (conv.training_params) {
          setTrainingParams(conv.training_params)
          if (conv.training_params.max_auto_tune_rounds)
            setMaxRounds(conv.training_params.max_auto_tune_rounds)
          if (conv.training_params.target_metric_threshold)
            setTargetThreshold(conv.training_params.target_metric_threshold)
        }
        setRoundHistory(
          (conv.rounds_history || []).map((r: any) => ({
            round_num: r.round_num,
            metric: r.metric,
            target_threshold: r.target_threshold,
            target_achieved: r.metric >= r.target_threshold,
            env_code_snapshot: '',
            diagnosis: r.diagnosis,
            changes_summary: r.changes_summary,
          })),
        )

        // Replay the durable thread verbatim — no relabelling, nothing dropped.
        setThreadEntries(conv.thread || [])
        setTurn(conv.turn || null)
        setQueuedMessage(null)
        clearNotices()

        if (conv.blueprint_spec) {
          setActiveSession({
            session_id: cid,
            blueprint: conv.blueprint_spec,
            dataset_id: conv.dataset_id,
            status: conv.status,
            current_round: conv.current_round,
            max_rounds: conv.training_params?.max_auto_tune_rounds || 5,
            target_threshold: conv.training_params?.target_metric_threshold || 90,
            current_metric: conv.best_metric,
            env_module_code: conv.env_code || '',
            arena_html: conv.arena_html || '',
            rounds_history: conv.rounds_history,
            episodes: conv.episodes,
          })
        } else {
          setActiveSession(null)
        }

        const stillTraining = conv.status === 'training'
        setIsTraining(stillTraining)
        if (stillTraining || conv.turn?.state === 'running') connectWebSocket(cid)
        setIsConversationsOpen(false)
        void refreshUsage(cid)
      } catch (err: any) {
        pushNotice('Restore failed', err?.message || 'Could not load the conversation.', 'warning')
      } finally {
        setIsLoadingThread(false)
      }
    },
    [applyDrift, clearNotices, connectWebSocket, pushNotice, refreshUsage],
  )

  const handleNewAgent = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    setActiveSession(null)
    setActiveConversationId(undefined)
    setThreadEntries([])
    setTurn(null)
    setPendingEntries([])
    setNotices([])
    setQueuedMessage(null)
    setArenaHtml('')
    setArenaLocked(false)
    setArenaSource(null)
    setArenaError(null)
    setArenaPending(false)
    setEnvModuleCode('')
    setEpisodes([])
    setLatestFrame(undefined)
    setScorecard(null)
    setCurrentMetric(0)
    setCurrentRound(1)
    setRoundHistory([])
    applyDrift(null, 'unmonitored')
    setDriftSamples(0)
    setDriftDetail('')
    setIsTraining(false)
    setIsConversationsOpen(false)
    chatInputRef.current?.focus()
  }, [applyDrift])

  const handleRequestNewAgent = useCallback(() => {
    if (
      attachedDataset &&
      !window.confirm(
        `Start a new agent? "${attachedDataset.name}" will be detached from this draft. The upload stays in the dataset library.`,
      )
    ) {
      return
    }
    handleNewAgent()
  }, [attachedDataset, handleNewAgent])

  const handleDeleteConversation = useCallback(
    async (cid: string) => {
      if (!window.confirm('Delete this conversation and everything it owns?')) return
      try {
        await deleteConversation(cid)
        if (cid === activeConversationRef.current) handleNewAgent()
        void loadConversationsList()
      } catch (err: any) {
        pushNotice('Delete failed', err?.message || 'Could not delete the conversation.', 'warning')
      }
    },
    [handleNewAgent, loadConversationsList, pushNotice],
  )

  const handleClearAllConversations = useCallback(async () => {
    if (!window.confirm('Delete every saved conversation? This cannot be undone.')) return
    try {
      await clearAllConversations()
      handleNewAgent()
      void loadConversationsList()
    } catch (err: any) {
      pushNotice('Clear failed', err?.message || 'Could not clear the history.', 'warning')
    }
  }, [handleNewAgent, loadConversationsList, pushNotice])

  // ------------------------------------------------------------------- arena

  const handleRetryArena = useCallback(async () => {
    const cid = activeConversationRef.current
    if (!cid) return
    if (!isLLMVerified) {
      setIsGateModalOpen(true)
      return
    }
    setArenaPending(true)
    setArenaError(null)
    try {
      const res = await updateArenaLook(
        cid,
        'Rebuild this arena as a bespoke, domain-specific canvas that renders the real state fields.',
      )
      if (res.arena_html) setArenaHtml(res.arena_html)
      if (res.arena_source === 'bespoke') {
        setArenaSource('bespoke')
        setArenaError(null)
      } else {
        setArenaError(res.error || 'The provider did not return a valid canvas.')
      }
      void syncThread(cid)
    } catch (err: any) {
      setArenaError(err?.message || 'Arena retry failed')
    } finally {
      setArenaPending(false)
    }
  }, [isLLMVerified, syncThread])

  const handleEditArenaLook = useCallback(() => {
    chatInputRef.current?.focus()
    chatInputRef.current?.select()
  }, [])

  const handleLockArena = useCallback(async () => {
    // Locking is a turn: say it in the thread instead of mutating the arena from a button.
    submitInstruction('Looks good, lock the arena and start training')
  }, [submitInstruction])

  // ------------------------------------------------------- rounds & evidence

  const handleRollback = useCallback(
    async (roundNum: number) => {
      const cid = activeConversationRef.current
      if (!cid) return
      if (!window.confirm(`Roll the policy and environment back to Round ${roundNum}?`)) return
      setIsRollingBack(true)
      try {
        const resp = await rollbackSession(cid, roundNum)
        setCurrentRound(resp.restored_round)
        setCurrentMetric(resp.restored_metric)
        pushNotice(
          `Rolled back to round ${resp.restored_round}`,
          `Policy weights and environment code restored. Metric ${resp.restored_metric.toFixed(1)}%.`,
          'warning',
        )
        await syncThread(cid)
      } catch (err: any) {
        pushNotice('Rollback failed', err?.message || 'Error during rollback.', 'warning')
      } finally {
        setIsRollingBack(false)
      }
    },
    [pushNotice, syncThread],
  )

  const handleExportPolicy = useCallback(async () => {
    const cid = activeConversationRef.current
    if (!cid) return

    const evidenceGaps: string[] = []
    if (scorecard) {
      if (!scorecard.beats_all_baselines) {
        const best = scorecard.baselines.length
          ? scorecard.baselines.reduce(
              (b, x) => (x.value_pct > b.value_pct ? x : b),
              scorecard.baselines[0],
            )
          : null
        evidenceGaps.push(
          best
            ? `Held-out ${scorecard.policy.value_pct.toFixed(1)}% does not beat the best baseline (${best.name.replace(/_/g, ' ')} ${best.value_pct.toFixed(1)}%).`
            : `Held-out ${scorecard.policy.value_pct.toFixed(1)}% has no baseline comparison.`,
        )
      }
      if (scorecard.crossfire && scorecard.crossfire.verdict !== 'win') {
        evidenceGaps.push(`Crossfire verdict: ${scorecard.crossfire.verdict}.`)
      }
      if (scorecard.eval_corpus && scorecard.eval_corpus.failed_cases > 0) {
        evidenceGaps.push(`${scorecard.eval_corpus.failed_cases} golden-corpus case(s) fail.`)
      }
      if (scorecard.safety && scorecard.safety.violations > 0) {
        evidenceGaps.push(
          `${scorecard.safety.violations} raw hard-constraint violation(s) — the runtime mask will override them.`,
        )
      }
    }
    if (
      evidenceGaps.length > 0 &&
      !window.confirm(
        `This policy still has unresolved evidence:\n\n- ${evidenceGaps.join('\n- ')}\n\nExport anyway? The bundle will include the safety rules it can enforce.`,
      )
    ) {
      return
    }

    setIsExporting(true)
    try {
      const exportRes = await exportSessionPolicy(cid)
      pushNotice(
        'Policy exported',
        `ONNX parity verified (max diff ${exportRes.onnx_parity_diff.toExponential(2)}). Runbundle packaged.`,
        'success',
      )
      window.open(exportRes.download_url, '_blank')
      setRegistryCount((prev) => prev + 1)
      await syncThread(cid)
    } catch (err: any) {
      pushNotice('Export failed', err?.message || 'Error exporting the policy.', 'warning')
    } finally {
      setIsExporting(false)
    }
  }, [pushNotice, scorecard, syncThread])

  /** Jump from a number in the scoreboard to the entry that produced it. */
  const focusRoundInThread = useCallback(
    (round: number) => {
      const match = [...threadEntries]
        .reverse()
        .find((e) => e.anchor?.round === round && (e.kind === 'round' || e.kind === 'tune'))
      setFocusedSeq(match ? match.seq : null)
    },
    [threadEntries],
  )

  // ------------------------------------------------------------------ uploads

  const handleFileUpload = useCallback(
    async (file: File) => {
      try {
        const summary = await uploadDatasetFile(file)
        setAttachedDataset(summary)
        submitInstruction(`Use the attached dataset "${summary.name}" for this agent`)
      } catch (e: any) {
        pushNotice('Upload failed', e?.message || 'Could not ingest the file.', 'warning')
      }
    },
    [pushNotice, submitInstruction],
  )

  const handleSQLDatasetSelected = useCallback(
    (ds: IngestedDatasetSummary) => {
      setAttachedDataset(ds)
      submitInstruction(`Use the SQL dataset "${ds.name}" (${ds.num_rows} rows) for this agent`)
    },
    [submitInstruction],
  )

  // ----------------------------------------------------------------- settings

  const handleSaveLLMSettings = useCallback((settings: LLMSettings) => {
    setLlmSettings(settings)
    setIsLLMVerified(true)
    localStorage.setItem('rl_ide_v4_llm_settings', JSON.stringify(settings))
  }, [])

  const handleGateVerified = useCallback((settings: LLMSettings) => {
    setLlmSettings(settings)
    setIsLLMVerified(true)
    setIsGateModalOpen(false)
  }, [])

  // ------------------------------------------------------------------- render

  const displayedEntries = useMemo(
    () => [...threadEntries, ...pendingEntries, ...notices],
    [threadEntries, pendingEntries, notices],
  )

  // While our own request is in flight the studio owns the turn, even before the server says so.
  const effectiveTurn: TurnStatus | null = useMemo(() => {
    if (isSending && turn?.state !== 'running') {
      return {
        conversation_id: activeConversationId || '',
        turn_id: turn?.turn_id || 'local',
        state: 'running',
        actor: 'studio',
        opened_at: new Date().toISOString(),
        closed_at: null,
        outcome: null,
      }
    }
    return turn
  }, [activeConversationId, isSending, turn])

  const quickActions = useMemo(() => {
    if (!arenaLocked) {
      return [
        '🎨 Neon cyberpunk theme',
        '📊 Add a telemetry HUD',
        '🔍 Larger canvas',
        '🔒 Lock arena & start training',
      ]
    }
    return [
      '🔄 Train 5 more rounds',
      '🧠 Make it more aggressive',
      '🛡️ Penalise mistakes harder',
      '🎨 Redesign the arena look',
    ]
  }, [arenaLocked])

  const handleQuickAction = useCallback(
    (chip: string) => {
      if (chip.includes('Lock arena')) {
        void handleLockArena()
      } else if (chip.includes('Train 5 more rounds')) {
        submitInstruction('Train for 5 more rounds with 500 episodes per round')
      } else if (chip.includes('aggressive')) {
        submitInstruction('Change the agent behaviour: make it more aggressive and prioritise attacking')
      } else if (chip.includes('Penalise')) {
        submitInstruction('Change the agent behaviour: heavily penalise stalling and defensive mistakes')
      } else if (chip.includes('cyberpunk')) {
        submitInstruction('Change the arena look to a neon cyberpunk theme with glowing borders')
      } else if (chip.includes('telemetry HUD')) {
        submitInstruction(
          'Change the arena look to add a detailed action probability radar and telemetry HUD',
        )
      } else if (chip.includes('Larger canvas')) {
        submitInstruction('Change the arena look: make the canvas larger with gold highlights')
      } else {
        submitInstruction(chip)
      }
    },
    [handleLockArena, submitInstruction],
  )

  const conversationIdForDrawers = activeConversationId ?? activeSession?.session_id ?? undefined

  return (
    <div className="flex flex-col h-screen w-screen overflow-hidden bg-gray-950 text-gray-100 font-sans relative">
      <Header
        llmSettings={llmSettings}
        isLLMVerified={isLLMVerified}
        onOpenSettings={() => (isLLMVerified ? setIsSettingsOpen(true) : setIsGateModalOpen(true))}
        driftStatus={driftStatus}
        driftDetail={driftDetail}
        driftSamples={driftSamples}
        onOpenConversations={() => setIsConversationsOpen(true)}
        conversationCount={conversations.length}
        usageCostUsd={usage.cost_usd}
        usageCalls={usage.calls}
        isCheckingLLM={isCheckingLLM}
        onOpenGuide={() => {
          setGuideSection(null)
          setIsGuideOpen(true)
        }}
        onOpenTrust={() => setIsTrustOpen(true)}
        trustVerdict={scorecard?.crossfire?.verdict ?? activeSession?.crossfire?.verdict}
        trustBlockers={
          scorecard?.crossfire?.reasons?.length ?? activeSession?.crossfire?.reasons?.length ?? 0
        }
        onOpenRegistry={() => setIsRegistryOpen(true)}
        registryCount={registryCount}
      />

      {/* Scoreboard left · Thread centre · Arena right */}
      <div className="flex-1 flex overflow-hidden">
        <ScoreboardRail
          targetMetricName={targetMetricName}
          targetThreshold={targetThreshold}
          currentMetric={currentMetric}
          currentRound={currentRound}
          maxRounds={maxRounds}
          roundHistory={roundHistory}
          scorecard={scorecard}
          driftStatus={driftStatus}
          driftDetail={driftDetail}
          usage={usage}
          dataset={attachedDataset}
          sessionStatus={isTraining ? 'training' : activeSession?.status}
          onFocusRound={focusRoundInThread}
          onExport={handleExportPolicy}
          isExporting={isExporting}
          hasPolicy={Boolean(activeSession)}
        />

        <main className="flex-1 min-w-0 flex flex-col bg-gray-950">
          {isLoadingThread ? (
            <div className="flex-1 flex items-center justify-center text-xs text-gray-500">
              Loading the thread…
            </div>
          ) : (
            <ThreadView
              entries={displayedEntries}
              turn={effectiveTurn}
              conversationTitle={
                conversations.find((c) => c.conversation_id === activeConversationId)?.title
              }
              onCoachAction={handleCoachAction}
              onAcknowledge={handleAcknowledgeCoach}
              onOpenTrust={() => setIsTrustOpen(true)}
              onSelectRound={focusRoundInThread}
              onRollbackRound={activeSession ? handleRollback : undefined}
              canRollback={Boolean(activeSession) && !isRollingBack}
              focusedSeq={focusedSeq}
            />
          )}

          <TurnComposer
            turn={effectiveTurn}
            isLLMVerified={isLLMVerified}
            onOpenSettings={() => setIsGateModalOpen(true)}
            onSend={(message) => void sendTurn(message)}
            onStop={() => void handleStop()}
            isStopping={isStopping}
            queued={queuedMessage}
            onQueue={setQueuedMessage}
            onCancelQueued={() => setQueuedMessage(null)}
            attachedDataset={attachedDataset}
            onAttachFile={(file) => void handleFileUpload(file)}
            onOpenSQLModal={() => setIsSQLModalOpen(true)}
            onRemoveDataset={() => setAttachedDataset(null)}
            trainingParams={trainingParams}
            onUpdateTrainingParams={setTrainingParams}
            quickActions={quickActions}
            onQuickAction={handleQuickAction}
            inputRef={chatInputRef}
            hasPolicy={arenaLocked}
          />
        </main>

        <ArenaStage
          sessionId={activeSession?.session_id ?? activeConversationId}
          arenaHtml={arenaHtml}
          isTraining={isTraining}
          isArenaLocked={arenaLocked}
          latestFrame={latestFrame}
          episodes={episodes}
          onOpenCodeInspector={() => setIsCodeInspectorOpen(true)}
          onLockArena={handleLockArena}
          onEditArenaLook={handleEditArenaLook}
          arenaSource={arenaSource}
          arenaPending={arenaPending}
          arenaError={arenaError}
          onRetryArena={() => void handleRetryArena()}
          onStepObserved={setLastStep}
          pulse={arenaPulse}
        />
      </div>

      <ConversationSidebar
        isOpen={isConversationsOpen}
        onClose={() => setIsConversationsOpen(false)}
        conversations={conversations}
        activeConversationId={activeConversationId}
        onSelectConversation={(cid) => void handleSelectConversation(cid)}
        onNewAgent={handleRequestNewAgent}
        onDeleteConversation={(cid) => void handleDeleteConversation(cid)}
        onClearAll={() => void handleClearAllConversations()}
        isLoading={isLoadingConversations}
      />

      <LLMVerificationGateModal
        isOpen={!isCheckingLLM && isGateModalOpen}
        canDismiss={isLLMVerified === false}
        currentSettings={llmSettings}
        onVerified={handleGateVerified}
        onClose={() => setIsGateModalOpen(false)}
      />

      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        settings={llmSettings}
        onSaveSettings={handleSaveLLMSettings}
      />

      <RegistryModal isOpen={isRegistryOpen} onClose={() => setIsRegistryOpen(false)} />

      <React.Suspense fallback={null}>
        <GuideModal
          isOpen={isGuideOpen}
          onClose={() => {
            setIsGuideOpen(false)
            setGuideSection(null)
          }}
          initialSection={guideSection}
        />
      </React.Suspense>

      <TrustPanel
        isOpen={isTrustOpen}
        onClose={() => setIsTrustOpen(false)}
        conversationId={conversationIdForDrawers}
        session={activeSession}
        lastStep={lastStep}
        refreshKey={trustRefreshKey}
        actionLabels={activeSession?.blueprint?.action_labels ?? []}
        onCorpusChanged={() => setTrustRefreshKey((k) => k + 1)}
      />

      <CodeInspectorModal
        isOpen={isCodeInspectorOpen}
        onClose={() => setIsCodeInspectorOpen(false)}
        envModuleCode={envModuleCode}
        arenaHtml={arenaHtml}
      />

      <SQLIngestModal
        isOpen={isSQLModalOpen}
        onClose={() => setIsSQLModalOpen(false)}
        onDatasetSelected={handleSQLDatasetSelected}
      />

      {/* The arena reports its stage without a separate panel competing with the thread. */}
      {arenaStage && (
        <span className="sr-only" aria-live="polite">
          {arenaStage}
        </span>
      )}
    </div>
  )
}

export default App
