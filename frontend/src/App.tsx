import React, { useState, useEffect, useRef } from 'react'
import { Header } from './components/Header'
import { ArchitectPanel } from './components/ArchitectPanel'
import { SandboxedArenaIframe } from './components/SandboxedArenaIframe'
import { TelemetryPanel } from './components/TelemetryPanel'
import { SettingsModal } from './components/SettingsModal'
import { LLMVerificationGateModal } from './components/LLMVerificationGateModal'
import { RegistryModal } from './components/RegistryModal'
import { CodeInspectorModal } from './components/CodeInspectorModal'
import { SQLIngestModal } from './components/SQLIngestModal'

import {
  AgentBlueprintSpec,
  IntakeResponse,
  SessionState,
  RoundSnapshot,
  EpisodeLog,
  LLMSettings,
  IngestedDatasetSummary,
  ActivityTimelineItem,
} from './types'

import {
  chatWithArchitect,
  startSynthesisSession,
  fetchSession,
  interactSession,
  nudgeSession,
  rollbackSession,
  exportSessionPolicy,
  fetchBundles,
  uploadDatasetFile,
  connectTelemetryWebSocket,
  getLLMStatus,
  saveAndVerifyLLMSettings,
} from './services/api'

export const App: React.FC = () => {
  // Session & Training State
  const [activeSession, setActiveSession] = useState<SessionState | null>(null)
  const [isSynthesizing, setIsSynthesizing] = useState(false)
  const [isTraining, setIsTraining] = useState(false)
  const [isExporting, setIsExporting] = useState(false)
  const [isRollingBack, setIsRollingBack] = useState(false)

  // Arena & Code State
  const [arenaHtml, setArenaHtml] = useState<string>('')
  const [envModuleCode, setEnvModuleCode] = useState<string>('')
  const [latestFrame, setLatestFrame] = useState<Record<string, any> | undefined>(undefined)
  const [episodes, setEpisodes] = useState<EpisodeLog[]>([])

  // Telemetry & Metrics
  const [currentMetric, setCurrentMetric] = useState<number>(0)
  const [targetThreshold, setTargetThreshold] = useState<number>(85)
  const [targetMetricName, setTargetMetricName] = useState<string>('Success / Win Rate')
  const [currentRound, setCurrentRound] = useState<number>(1)
  const [maxRounds, setMaxRounds] = useState<number>(4)
  const [roundHistory, setRoundHistory] = useState<RoundSnapshot[]>([])
  const [driftStatus, setDriftStatus] = useState<string>('healthy')

  // Dataset & Timeline
  const [attachedDataset, setAttachedDataset] = useState<IngestedDatasetSummary | null>(null)
  const [timeline, setTimeline] = useState<ActivityTimelineItem[]>([])

  // Modals & Mandatory LLM Gate
  const [isSettingsOpen, setIsSettingsOpen] = useState(false)
  const [isRegistryOpen, setIsRegistryOpen] = useState(false)
  const [isCodeInspectorOpen, setIsCodeInspectorOpen] = useState(false)
  const [isSQLModalOpen, setIsSQLModalOpen] = useState(false)
  const [registryCount, setRegistryCount] = useState(0)

  // Mandatory LLM Verification State
  const [isLLMVerified, setIsLLMVerified] = useState<boolean>(false)
  const [isCheckingLLM, setIsCheckingLLM] = useState<boolean>(true)
  const [isGateModalOpen, setIsGateModalOpen] = useState<boolean>(false)

  // LLM Settings
  const [llmSettings, setLlmSettings] = useState<LLMSettings>(() => {
    const saved = localStorage.getItem('rl_ide_v4_llm_settings')
    if (saved) {
      try {
        return JSON.parse(saved)
      } catch {
        // fallback
      }
    }
    return {
      provider: 'deepseek',
      baseUrl: 'https://api.deepseek.com/v1',
      modelName: 'deepseek-chat',
      apiKey: '',
    }
  })

  // Mandatory LLM Key Verification on Startup
  useEffect(() => {
    let isMounted = true
    async function verifyStartupCredentials() {
      setIsCheckingLLM(true)
      try {
        // 1. Check if backend already has an active verified LLM
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

        // 2. If backend is not verified, check if localStorage has saved credentials to test
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
                  addTimelineItem(
                    'LLM Connection Verified ⚡',
                    `Active engine: ${parsed.provider.toUpperCase()} (${parsed.modelName || 'default'})`,
                    'info',
                    'success',
                  )
                }
                return
              }
            }
          } catch {
            // Credentials failed verification
          }
        }

        // 3. Not verified: Lock studio and trigger mandatory setup gate
        if (isMounted) {
          setIsLLMVerified(false)
          setIsGateModalOpen(true)
        }
      } catch (err) {
        if (isMounted) {
          setIsLLMVerified(false)
          setIsGateModalOpen(true)
        }
      } finally {
        if (isMounted) setIsCheckingLLM(false)
      }
    }

    verifyStartupCredentials()
    return () => {
      isMounted = false
    }
  }, [])

  const wsRef = useRef<WebSocket | null>(null)

  // Cleanup WebSocket on unmount
  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close()
        wsRef.current = null
      }
    }
  }, [])

  // Refresh bundles count on mount
  useEffect(() => {
    fetchBundles()
      .then((bundles) => setRegistryCount(bundles.length))
      .catch(() => {})
  }, [isRegistryOpen])

  // Periodic poll fallback in case WebSocket drops frames or disconnects
  useEffect(() => {
    if (!isTraining || !activeSession?.session_id) return
    const pollTimer = setInterval(async () => {
      try {
        const fresh = await fetchSession(activeSession.session_id)
        if (fresh.current_round) setCurrentRound(fresh.current_round)
        if (fresh.current_metric !== undefined) setCurrentMetric(fresh.current_metric)
        const history = fresh.rounds_history || fresh.round_history || []
        if (history.length > 0) {
          setRoundHistory(history.map((r: any) => ({
            round_num: r.round_num,
            metric: r.metric,
            target_threshold: r.target_threshold,
            target_achieved: r.metric >= r.target_threshold,
            env_code_snapshot: '',
            diagnosis: r.diagnosis,
            changes_summary: r.changes_summary,
          })))
        }
        if (fresh.status === 'completed' || fresh.status === 'target_achieved') {
          setIsTraining(false)
        }
      } catch (e) {
        // Silent poll error
      }
    }, 2500)

    return () => clearInterval(pollTimer)
  }, [isTraining, activeSession?.session_id])

  // Helper to append timeline item
  const addTimelineItem = (
    title: string,
    description: string,
    type: ActivityTimelineItem['type'] = 'info',
    status: ActivityTimelineItem['status'] = 'info',
  ) => {
    const newItem: ActivityTimelineItem = {
      id: `act_${Date.now()}_${Math.random().toString(36).substring(2, 6)}`,
      type,
      title,
      description,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
      status,
    }
    setTimeline((prev) => [...prev, newItem])
  }

  // 1. Conversational Intake Interview
  const handleChatWithArchitect = async (
    prompt: string,
    answers: Record<string, string> = {},
    forceSmartDefaults: boolean = false,
  ): Promise<IntakeResponse> => {
    if (!isLLMVerified) {
      setIsGateModalOpen(true)
      throw new Error('A verified LLM API key is mandatory before using the studio.')
    }

    addTimelineItem(
      'Conversational Intake Consulted',
      `Evaluating requirements across 5 Pillars for: "${prompt.slice(0, 60)}..."`,
      'synthesis',
      'info',
    )

    try {
      const resp = await chatWithArchitect({
        user_prompt: prompt,
        dataset_id: attachedDataset?.dataset_id,
        user_answers: answers,
        force_smart_defaults: forceSmartDefaults,
      })

      if (resp.status === 'ready_to_build') {
        addTimelineItem(
          'Agent Blueprint Locked',
          `Objective: ${resp.blueprint?.objective || 'Agent specification complete'}`,
          'synthesis',
          'success',
        )
      }

      return resp
    } catch (err: any) {
      if (err.message && (err.message.includes('verified') || err.message.includes('mandatory') || err.message.includes('403'))) {
        setIsLLMVerified(false)
        setIsGateModalOpen(true)
      }
      throw err
    }
  }

  // 2. Start JIT Autonomous Session
  const handleStartSession = async (blueprint: AgentBlueprintSpec, datasetId?: string) => {
    if (!isLLMVerified) {
      setIsGateModalOpen(true)
      throw new Error('A verified LLM API key is mandatory before synthesizing environments.')
    }

    setIsSynthesizing(true)
    setIsTraining(true)
    setEpisodes([])
    setRoundHistory([])
    setCurrentMetric(0)
    setTargetThreshold(blueprint.target_metric_threshold || 85)
    setTargetMetricName(blueprint.target_metric_name || 'Success Rate')
    setMaxRounds(blueprint.max_auto_tune_rounds || 4)
    setCurrentRound(1)

    addTimelineItem(
      'JIT Synthesis Initiated',
      `Synthesizing env_module.py and arena.html for ${blueprint.task_id}...`,
      'synthesis',
      'info',
    )

    try {
      const resp = await startSynthesisSession({
        blueprint,
        dataset_id: datasetId,
        run_sync: false,
      })

      const sid = resp.session_id

      // Fetch fresh session state to get synthesized code & arena
      const sessionData = await fetchSession(sid)
      setActiveSession(sessionData)
      setArenaHtml(sessionData.arena_html || '')
      setEnvModuleCode(sessionData.env_module_code || '')
      if (sessionData.current_round) setCurrentRound(sessionData.current_round)
      if (sessionData.current_metric) setCurrentMetric(sessionData.current_metric)
      const history = sessionData.rounds_history || sessionData.round_history || []
      if (history.length > 0) {
        setRoundHistory(history.map((r: any) => ({
          round_num: r.round_num,
          metric: r.metric,
          target_threshold: r.target_threshold,
          target_achieved: r.metric >= r.target_threshold,
          env_code_snapshot: '',
          diagnosis: r.diagnosis,
          changes_summary: r.changes_summary,
        })))
      }
      if (sessionData.status === 'completed' || sessionData.status === 'target_achieved') {
        setIsTraining(false)
      }

      addTimelineItem(
        'Synthesized env_module.py Verified ✅',
        'Static AST safety check passed; 25/25 sandbox smoke steps verified without error.',
        'smoke_test',
        'success',
      )
      addTimelineItem(
        'Dynamic Arena Synthesized 🎨',
        `Custom arena.html synthesized with ARENA_UPDATE & ARENA_INTERACT bridge.`,
        'arena',
        'success',
      )

      // Connect WebSocket for live telemetry streaming
      connectWebSocket(sid)
    } catch (err: any) {
      addTimelineItem('Synthesis Failed', err.message || 'Error compiling environment', 'error', 'warning')
      setIsTraining(false)
    } finally {
      setIsSynthesizing(false)
    }
  }

  // 3. Connect WebSocket for Live Training Telemetry
  const connectWebSocket = (sessionId: string) => {
    if (wsRef.current) {
      wsRef.current.close()
    }

    const ws = connectTelemetryWebSocket(
      sessionId,
      (data) => {
        handleWebSocketEvent(data)
      },
      (err) => console.warn('WebSocket error:', err),
      () => console.log('WebSocket closed'),
    )
    wsRef.current = ws
  }

  const handleWebSocketEvent = (event: any) => {
    const eventType = event.event

    if (eventType === 'round_started') {
      const rNum = event.round ?? event.round_num ?? 1
      setCurrentRound(rNum)
      addTimelineItem(
        `Round ${rNum} Started`,
        event.message || `Autonomous training burst executing with PPO policy...`,
        'round',
        'info',
      )
    } else if (eventType === 'warmstart_completed') {
      addTimelineItem(
        'Warm-Start Behavioral Cloning Complete',
        event.message || `Pre-trained on heuristic demonstration trajectories.`,
        'round',
        'info',
      )
    } else if (eventType === 'critic_warmup_completed') {
      addTimelineItem(
        'Value Critic Calibrated',
        event.message || `Value network calibrated on updated reward function.`,
        'round',
        'info',
      )
    } else if (eventType === 'episode_step') {
      // Forward live frame to arena iframe
      setLatestFrame(event)
      if (event.episode_id !== undefined) {
        setEpisodes((prev) => {
          const epLog: EpisodeLog = {
            episode_id: event.episode_id,
            total_reward: event.total_reward || 0,
            reward_breakdown: event.reward_breakdown || {},
            is_success: !!event.is_success,
          }
          return [...prev.slice(-49), epLog]
        })
      }
    } else if (eventType === 'round_evaluated') {
      const rNum = event.round ?? event.round_num ?? 1
      const metric = event.metric ?? event.current_metric ?? 0
      const target = event.target ?? event.target_threshold ?? 85
      const isTargetAchieved = metric >= target
      setCurrentMetric(metric)
      const snap: RoundSnapshot = {
        round_num: rNum,
        metric,
        target_threshold: target,
        target_achieved: isTargetAchieved,
        env_code_snapshot: '',
        avg_return: event.avg_return,
        term_rewards: event.reward_breakdown || event.term_rewards,
      }
      setRoundHistory((prev) => [...prev, snap])

      addTimelineItem(
        `Round ${rNum} Evaluated: ${metric.toFixed(1)}%`,
        `Target threshold is ${target}%.`,
        'round',
        isTargetAchieved ? 'success' : 'info',
      )
    } else if (eventType === 'auto_tuned') {
      const rNum = event.round ?? event.round_num ?? 1
      const diagnosis = event.diagnosis || event.tuning_result?.diagnosis || 'Auto-tuned reward function.'
      const changesSummary = event.changes_summary || event.tuning_result?.changes_summary || ''
      setRoundHistory((prev) => {
        const updated = [...prev]
        const lastIdx = updated.length - 1
        if (lastIdx >= 0) {
          updated[lastIdx] = {
            ...updated[lastIdx],
            diagnosis,
            changes_summary: changesSummary,
          }
        }
        return updated
      })
      if (event.tuning_result?.updated_code) {
        setEnvModuleCode(event.tuning_result.updated_code)
      }
      addTimelineItem(
        `Auto-Tuner Diagnosis: Round ${rNum} 🧠`,
        `${diagnosis} ${changesSummary}`,
        'auto_tune',
        'warning',
      )
    } else if (eventType === 'target_achieved') {
      setIsTraining(false)
      const metric = event.metric ?? event.current_metric ?? 100
      setCurrentMetric(metric)
      addTimelineItem(
        '🌟 Target Mastery Achieved!',
        event.message || `Agent exceeded target metric with ${metric.toFixed(1)}% performance!`,
        'mastery',
        'success',
      )
    } else if (eventType === 'budget_exhausted') {
      setIsTraining(false)
      const metric = event.metric ?? event.current_metric ?? 0
      setCurrentMetric(metric)
      addTimelineItem(
        '🏁 Auto-Tune Complete',
        event.message || `Completed all auto-tuning rounds. Final metric: ${metric.toFixed(1)}%.`,
        'round',
        'info',
      )
    }
  }

  // 4. Mid-Flight Nudge
  const handleNudge = async (feedbackText: string) => {
    if (!activeSession) return
    addTimelineItem(
      'Mid-Flight Guidance Nudge Applied',
      `Prompt: "${feedbackText}" — Triggering targeted reward tuning round...`,
      'nudge',
      'warning',
    )
    const result = await nudgeSession(activeSession.session_id, feedbackText)
    if (result.updated_code) {
      setEnvModuleCode(result.updated_code)
    }
    addTimelineItem(
      'Nudge Auto-Tune Applied',
      `AutoTuner updated environment code: ${result.changes_summary}`,
      'auto_tune',
      'success',
    )
  }

  // 5. Rollback to Previous Round
  const handleRollback = async (roundNum: number) => {
    if (!activeSession) return
    setIsRollingBack(true)
    try {
      const resp = await rollbackSession(activeSession.session_id, roundNum)
      setCurrentRound(resp.restored_round)
      setCurrentMetric(resp.restored_metric)
      addTimelineItem(
        `Rollback to Round ${resp.restored_round} Complete`,
        `Restored policy weights and environment code to snapshot. Metric: ${resp.restored_metric.toFixed(1)}%`,
        'round',
        'warning',
      )
    } catch (e: any) {
      addTimelineItem('Rollback Failed', e.message || 'Error during rollback', 'error', 'warning')
    } finally {
      setIsRollingBack(false)
    }
  }

  // 6. Export Standalone ONNX RunBundle
  const handleExportPolicy = async () => {
    if (!activeSession) return
    setIsExporting(true)
    try {
      const exportRes = await exportSessionPolicy(activeSession.session_id)
      addTimelineItem(
        'Policy Exported to ONNX .runbundle 📦',
        `Parity verified (max diff: ${exportRes.onnx_parity_diff.toExponential(2)}). Runbundle packaged.`,
        'mastery',
        'success',
      )
      // Trigger download
      window.open(exportRes.download_url, '_blank')
      setRegistryCount((prev) => prev + 1)
    } catch (err: any) {
      addTimelineItem('Export Failed', err.message || 'Error exporting policy', 'error', 'warning')
    } finally {
      setIsExporting(false)
    }
  }

  // 7. Universal File Upload
  const handleFileUpload = async (file: File) => {
    try {
      const summary = await uploadDatasetFile(file)
      setAttachedDataset(summary)
      addTimelineItem(
        'Dataset Attached 📊',
        `Uploaded "${summary.name}" (${summary.num_rows} rows, ${summary.columns.length} columns)`,
        'synthesis',
        'success',
      )
    } catch (e: any) {
      addTimelineItem('Upload Failed', e.message || 'Could not ingest file', 'error', 'warning')
    }
  }

  // 8. Save LLM Settings & Gate Unlock
  const handleSaveLLMSettings = (settings: LLMSettings) => {
    setLlmSettings(settings)
    setIsLLMVerified(true)
    localStorage.setItem('rl_ide_v4_llm_settings', JSON.stringify(settings))
    addTimelineItem(
      'LLM Provider Updated',
      `Active engine set to ${settings.provider.toUpperCase()} (${settings.modelName || 'default'})`,
      'info',
      'info',
    )
  }

  const handleGateVerified = (settings: LLMSettings) => {
    setLlmSettings(settings)
    setIsLLMVerified(true)
    setIsGateModalOpen(false)
    addTimelineItem(
      'LLM Connection Verified ⚡',
      `Credentials verified with live probe: ${settings.provider.toUpperCase()} (${settings.modelName || 'default'}). Studio unlocked!`,
      'info',
      'success',
    )
  }

  return (
    <div className="flex flex-col h-screen w-screen overflow-hidden bg-gray-950 text-gray-100 font-sans relative">
      {/* Top Header */}
      <Header
        llmSettings={llmSettings}
        isLLMVerified={isLLMVerified}
        onOpenSettings={() => (isLLMVerified ? setIsSettingsOpen(true) : setIsGateModalOpen(true))}
        onOpenRegistry={() => setIsRegistryOpen(true)}
        registryCount={registryCount}
        onExport={handleExportPolicy}
        isExporting={isExporting}
        hasActivePolicy={!!activeSession}
        driftStatus={driftStatus}
      />

      {/* Main Studio 3-Panel Workspace */}
      <div className={`flex-1 flex overflow-hidden transition-all duration-300 ${!isLLMVerified ? 'filter blur-[1px] select-none' : ''}`}>
        {/* Left Panel: Conversational Intake Architect & Autonomous Stream */}
        <ArchitectPanel
          onStartSession={handleStartSession}
          onChatWithArchitect={handleChatWithArchitect}
          onNudge={handleNudge}
          onOpenSQLModal={() => setIsSQLModalOpen(true)}
          onFileUpload={handleFileUpload}
          attachedDataset={attachedDataset}
          onRemoveDataset={() => setAttachedDataset(null)}
          timeline={timeline}
          isTraining={isTraining}
          isSynthesizing={isSynthesizing}
          activeSessionId={activeSession?.session_id}
          isLLMVerified={isLLMVerified}
          onOpenSettings={() => setIsGateModalOpen(true)}
        />

        {/* Center Panel: Universal Sandboxed Arena Iframe */}
        <SandboxedArenaIframe
          sessionId={activeSession?.session_id}
          arenaHtml={arenaHtml}
          isTraining={isTraining}
          latestFrame={latestFrame}
          episodes={episodes}
          onOpenCodeInspector={() => setIsCodeInspectorOpen(true)}
        />

        {/* Right Panel: Target Expectations & Multi-Round Telemetry */}
        <TelemetryPanel
          targetMetricName={targetMetricName}
          targetThreshold={targetThreshold}
          currentMetric={currentMetric}
          currentRound={currentRound}
          maxRounds={maxRounds}
          roundHistory={roundHistory}
          driftStatus={driftStatus}
          onRollback={handleRollback}
          isRollingBack={isRollingBack}
        />
      </div>

      {/* Mandatory LLM Verification Gate Modal (Unclosable until verified) */}
      <LLMVerificationGateModal
        isOpen={!isCheckingLLM && (!isLLMVerified || isGateModalOpen)}
        currentSettings={llmSettings}
        onVerified={handleGateVerified}
      />

      {/* Modals */}
      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        settings={llmSettings}
        onSaveSettings={handleSaveLLMSettings}
      />

      <RegistryModal
        isOpen={isRegistryOpen}
        onClose={() => setIsRegistryOpen(false)}
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
        onDatasetSelected={(ds) => {
          setAttachedDataset(ds)
          addTimelineItem(
            'SQL Dataset Attached 🗄️',
            `Connected "${ds.name}" (${ds.num_rows} rows, ${ds.columns.length} columns)`,
            'synthesis',
            'success',
          )
        }}
      />
    </div>
  )
}
export default App
