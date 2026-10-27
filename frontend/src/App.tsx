import React, { useState, useEffect, useRef } from 'react'
import { Header } from './components/Header'
import { LeftPanel } from './components/LeftPanel'
import { CenterPanel } from './components/CenterPanel'
import { RightPanel } from './components/RightPanel'
import { ExportModal } from './components/ExportModal'
import { SettingsModal } from './components/SettingsModal'
import { RegistryModal } from './components/RegistryModal'
import { VibeCreatorBar } from './components/VibeCreatorBar'
import { VibeRulesPanel } from './components/VibeRulesPanel'
import { PolymorphicArena } from './components/PolymorphicArena'
import { VibeCoachPanel } from './components/VibeCoachPanel'
import {
  RewardDSL,
  EpisodeLog,
  RewardDiffProposal,
  ExportResponse,
  VersionRecord,
  LLMSettings,
  TaskConfig,
  PlainRuleCard,
  SkillSummary,
  IngestedDatasetSummary,
} from './types'
import {
  compileTask,
  runTrainingBurst,
  toggleLockEpisode,
  previewCritique,
  applyCritiqueDiff,
  rollbackTaskVersion,
  exportPolicyBundle,
  fetchBundles,
  createVibeSession,
  coachVibeSession,
} from './services/api'

const DEFAULT_PROMPT =
  'Train an unbeatable Tic-Tac-Toe agent, take center when open, block opponent wins, and win.'

interface CoachMessage {
  id: string
  sender: 'user' | 'coach'
  text: string
  masteryDelta?: number
  afterMastery?: number
  appliedDiff?: Array<{ term_name: string; action: string }>
  timestamp: string
}

export const App: React.FC = () => {
  // Mode Switch: 'vibe' (Default) | 'pro'
  const [viewMode, setViewMode] = useState<'vibe' | 'pro'>('vibe')

  // State: Task & Adapter
  const [adapter, setAdapter] = useState('DynamicSimAdapter')
  const [taskId, setTaskId] = useState('task_vibe_default')
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT)
  const [currentVersion, setCurrentVersion] = useState(1)
  const [availableVersions, setAvailableVersions] = useState<number[]>([1])
  const [versionHistory, setVersionHistory] = useState<VersionRecord[]>([])
  const [rewardDsl, setRewardDsl] = useState<RewardDSL | null>(null)

  // State: Episodes & Training
  const [episodes, setEpisodes] = useState<EpisodeLog[]>([])
  const [selectedEpisodeId, setSelectedEpisodeId] = useState<number | null>(null)
  const [lockedEpisodeIds, setLockedEpisodeIds] = useState<number[]>([])
  const [isPaused, setIsPaused] = useState(false)
  const [driftStatus, setDriftStatus] = useState<'healthy' | 'drifted'>('healthy')

  // State: Critique & Diff
  const [diffProposal, setDiffProposal] = useState<RewardDiffProposal | null>(null)
  const [currentCritiqueText, setCurrentCritiqueText] = useState('')

  // State: Modals & Loading
  const [isCompiling, setIsCompiling] = useState(false)
  const [isRunningBurst, setIsRunningBurst] = useState(false)
  const [isPreviewing, setIsPreviewing] = useState(false)
  const [isApplying, setIsApplying] = useState(false)
  const [isExporting, setIsExporting] = useState(false)
  const [isExportModalOpen, setIsExportModalOpen] = useState(false)
  const [isSettingsModalOpen, setIsSettingsModalOpen] = useState(false)
  const [isRegistryModalOpen, setIsRegistryModalOpen] = useState(false)
  const [registryCount, setRegistryCount] = useState(0)
  const [exportData, setExportData] = useState<ExportResponse | null>(null)

  // ===========================================================================
  // Vibe Studio State (Phase 3D)
  // ===========================================================================
  const [vibeSessionId, setVibeSessionId] = useState<string>('')
  const [vibeUiMode, setVibeUiMode] = useState<string>('board_grid')
  const [vibeRules, setVibeRules] = useState<PlainRuleCard[]>([])
  const [skillSummary, setSkillSummary] = useState<SkillSummary | null>(null)
  const [activeDataset, setActiveDataset] = useState<IngestedDatasetSummary | null>(null)
  const [vibeLatestFrame, setVibeLatestFrame] = useState<Record<string, any>>({})
  const [isLaunchingVibe, setIsLaunchingVibe] = useState(false)
  const [isCoaching, setIsCoaching] = useState(false)
  const [coachMessages, setCoachMessages] = useState<CoachMessage[]>([])

  // State: LLM Settings
  const [llmSettings, setLlmSettings] = useState<LLMSettings>(() => {
    const saved = localStorage.getItem('rl_ide_llm_settings')
    if (saved) {
      try {
        return JSON.parse(saved)
      } catch {
        // fallback
      }
    }
    return {
      provider: 'mock',
      openaiApiKey: '',
      anthropicApiKey: '',
      ollamaBaseUrl: 'http://localhost:11434/v1',
      modelName: 'mock-rules',
    }
  })

  const wsRef = useRef<WebSocket | null>(null)

  // Load bundle registry count on mount
  useEffect(() => {
    fetchBundles()
      .then((bundles) => setRegistryCount(bundles.length))
      .catch(() => {})
  }, [isRegistryModalOpen, isExportModalOpen])

  // Auto-launch initial Vibe session on mount for immediate out-of-the-box delight
  useEffect(() => {
    handleLaunchVibe(DEFAULT_PROMPT)
  }, [])

  // Initialize WebSocket connection for real-time telemetry streaming
  useEffect(() => {
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const host = window.location.host
    const wsUrl = `${proto}//${host}/ws/stream`

    let ws: WebSocket
    try {
      ws = new WebSocket(wsUrl)
      wsRef.current = ws

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          if (data.type === 'episode') {
            const epLog: EpisodeLog = {
              episode_id: data.episode_id,
              steps: [],
              total_reward: data.total_reward,
              breakdown: data.breakdown || {},
              step_rewards: [],
              step_breakdowns: [],
            }
            setEpisodes((prev) => {
              const updated = [...prev, epLog]
              return updated.slice(-200)
            })
          }
        } catch (e) {
          console.error('WebSocket parse error:', e)
        }
      }

      ws.onerror = () => {
        // Fall back gracefully to REST
      }
    } catch {
      // WS unavailable in preview sandbox, REST polling fallback works automatically
    }

    return () => {
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [])

  // ===========================================================================
  // Vibe Studio Actions
  // ===========================================================================
  const handleLaunchVibe = async (userPrompt: string, datasetId?: string) => {
    setIsLaunchingVibe(true)
    try {
      const res = await createVibeSession(userPrompt, datasetId, 15, llmSettings)
      setVibeSessionId(res.session_id)
      setTaskId(res.task_id)
      setVibeUiMode(res.ui_mode || 'board_grid')
      setAdapter(res.adapter_name || 'DynamicSimAdapter')
      setVibeRules(res.rules || [])
      setSkillSummary({
        mastery_score: res.mastery_score,
        status_badge: res.status_badge,
        badge_variant: res.mastery_score >= 85 ? 'emerald' : res.mastery_score >= 70 ? 'blue' : 'amber',
        win_rate: res.win_rate,
        total_episodes: res.recent_episodes?.length || 15,
        key_insights: [
          `Agent initialized with ${res.rules?.length || 0} active behavioral rules.`,
          `Initial training burst completed across ${res.recent_episodes?.length || 15} episodes.`,
        ],
      })
      if (res.recent_episodes?.length) {
        setEpisodes(res.recent_episodes)
        setSelectedEpisodeId(res.recent_episodes[res.recent_episodes.length - 1].episode_id)
      }
      if (res.latest_frame) {
        setVibeLatestFrame(res.latest_frame)
      }

      // Initial coach greeting
      setCoachMessages([
        {
          id: `msg_${Date.now()}`,
          sender: 'coach',
          text: res.coach_message || `✨ Initialized and trained your ${res.adapter_name} agent!`,
          afterMastery: res.mastery_score,
          timestamp: new Date().toLocaleTimeString(),
        },
      ])
    } catch (err: any) {
      console.error('Launch Vibe error:', err)
    } finally {
      setIsLaunchingVibe(false)
    }
  }

  const handleSendCoachCritique = async (critiqueText: string) => {
    if (!vibeSessionId || isCoaching) return
    setIsCoaching(true)

    // Append user message immediately
    const userMsg: CoachMessage = {
      id: `usr_${Date.now()}`,
      sender: 'user',
      text: critiqueText,
      timestamp: new Date().toLocaleTimeString(),
    }
    setCoachMessages((prev) => [...prev, userMsg])

    try {
      const res = await coachVibeSession(vibeSessionId, {
        critiqueText,
        burstEpisodes: 10,
        llmSettings,
      })

      if (res.rules) setVibeRules(res.rules)
      if (res.after_mastery !== undefined && skillSummary) {
        setSkillSummary((prev) =>
          prev
            ? {
                ...prev,
                mastery_score: res.after_mastery,
                status_badge: res.after_mastery >= 85 ? 'Mastered' : res.after_mastery >= 70 ? 'Proficient' : 'Learning',
              }
            : null
        )
      }
      if (res.recent_episodes?.length) {
        setEpisodes(res.recent_episodes)
        setSelectedEpisodeId(res.recent_episodes[res.recent_episodes.length - 1].episode_id)
      }
      if (res.latest_frame) {
        setVibeLatestFrame(res.latest_frame)
      }

      const coachMsg: CoachMessage = {
        id: `coach_${Date.now()}`,
        sender: 'coach',
        text: res.coach_reply,
        masteryDelta: res.mastery_delta,
        afterMastery: res.after_mastery,
        appliedDiff: res.applied_diff,
        timestamp: new Date().toLocaleTimeString(),
      }
      setCoachMessages((prev) => [...prev, coachMsg])
    } catch (err: any) {
      console.error('Coach critique error:', err)
    } finally {
      setIsCoaching(false)
    }
  }

  const handleAdjustRuleImportance = async (termName: string, newImportance: string) => {
    if (!vibeSessionId || isCoaching) return
    setIsCoaching(true)
    try {
      const res = await coachVibeSession(vibeSessionId, {
        termName,
        newImportance,
        burstEpisodes: 8,
      })
      if (res.rules) setVibeRules(res.rules)
      if (res.after_mastery !== undefined && skillSummary) {
        setSkillSummary((prev) =>
          prev ? { ...prev, mastery_score: res.after_mastery } : null
        )
      }
      if (res.recent_episodes?.length) {
        setEpisodes(res.recent_episodes)
      }
      const coachMsg: CoachMessage = {
        id: `coach_${Date.now()}`,
        sender: 'coach',
        text: res.coach_reply,
        masteryDelta: res.mastery_delta,
        afterMastery: res.after_mastery,
        appliedDiff: res.applied_diff,
        timestamp: new Date().toLocaleTimeString(),
      }
      setCoachMessages((prev) => [...prev, coachMsg])
    } catch (err: any) {
      console.error('Adjust rule error:', err)
    } finally {
      setIsCoaching(false)
    }
  }

  const handleFeedback = async (type: 'thumbs_up' | 'thumbs_down', episodeId: number) => {
    if (!vibeSessionId || isCoaching) return
    if (type === 'thumbs_up') {
      setIsCoaching(true)
      try {
        const res = await coachVibeSession(vibeSessionId, {
          feedbackType: 'thumbs_up',
          targetEpisodeId: episodeId,
        })
        setLockedEpisodeIds((prev) => [...prev, episodeId])
        const coachMsg: CoachMessage = {
          id: `coach_${Date.now()}`,
          sender: 'coach',
          text: res.coach_reply,
          timestamp: new Date().toLocaleTimeString(),
        }
        setCoachMessages((prev) => [...prev, coachMsg])
      } catch (err) {
        console.error('Lock feedback error:', err)
      } finally {
        setIsCoaching(false)
      }
    } else {
      // Thumbs down: prompt AI Coach to fix bad actions in this episode
      handleSendCoachCritique(`Fix mistakes and avoid bad outcomes observed in episode #${episodeId}`)
    }
  }

  // ===========================================================================
  // Pro Workbench Actions
  // ===========================================================================
  const handleCompile = async () => {
    setIsCompiling(true)
    try {
      const res = await compileTask(prompt, adapter, taskId, llmSettings)
      if (res.task_config) {
        setRewardDsl(res.task_config.reward_dsl)
        setCurrentVersion(res.version || 1)
        setAvailableVersions([res.version || 1])
        const now = new Date().toLocaleTimeString()
        setVersionHistory([
          {
            version: res.version || 1,
            timestamp: now,
            description: 'Initial NL Compilation & Warm-Start',
          },
        ])

        const burstRes = await runTrainingBurst(taskId, 15)
        if (burstRes.episodes?.length) {
          setEpisodes(burstRes.episodes)
          setSelectedEpisodeId(burstRes.episodes[burstRes.episodes.length - 1].episode_id)
        }
      }
    } catch (err) {
      console.error('Compile error:', err)
    } finally {
      setIsCompiling(false)
    }
  }

  const handleRunBurst = async (nEpisodes = 15) => {
    setIsRunningBurst(true)
    try {
      const res = await runTrainingBurst(taskId, nEpisodes)
      if (res.episodes?.length) {
        setEpisodes((prev) => {
          const combined = [...prev, ...res.episodes]
          return combined.slice(-200)
        })
        setSelectedEpisodeId(res.episodes[res.episodes.length - 1].episode_id)
      }
    } catch (err) {
      console.error('Burst error:', err)
    } finally {
      setIsRunningBurst(false)
    }
  }

  const handleToggleLock = async (episodeId: number) => {
    try {
      const res = await toggleLockEpisode(episodeId)
      setLockedEpisodeIds(res.locked_episodes)
      setEpisodes((prev) =>
        prev.map((ep) =>
          ep.episode_id === episodeId ? { ...ep, locked: res.locked } : ep
        )
      )
    } catch (err) {
      console.error('Lock error:', err)
    }
  }

  const handlePreviewCritique = async (critiqueText: string) => {
    if (!selectedEpisodeId) return
    setIsPreviewing(true)
    setCurrentCritiqueText(critiqueText)
    try {
      const proposal = await previewCritique(
        taskId,
        selectedEpisodeId,
        critiqueText,
        llmSettings
      )
      setDiffProposal(proposal)
    } catch (err) {
      console.error('Critique preview error:', err)
    } finally {
      setIsPreviewing(false)
    }
  }

  const handleApplyDiff = async () => {
    if (!diffProposal) return
    setIsApplying(true)
    try {
      const res = await applyCritiqueDiff(
        taskId,
        diffProposal.new_dsl,
        currentCritiqueText,
        true
      )
      if (res.status === 'applied') {
        setRewardDsl(diffProposal.new_dsl)
        const nextVer = res.version || currentVersion + 1
        setCurrentVersion(nextVer)
        setAvailableVersions((prev) => [...prev, nextVer])
        const now = new Date().toLocaleTimeString()
        setVersionHistory((prev) => [
          ...prev,
          {
            version: nextVer,
            timestamp: now,
            description: `Critique: "${currentCritiqueText}"`,
            critique_text: currentCritiqueText,
          },
        ])
        setDiffProposal(null)
        setCurrentCritiqueText('')
        handleRunBurst(15)
      }
    } catch (err) {
      console.error('Apply diff error:', err)
    } finally {
      setIsApplying(false)
    }
  }

  const handleDiscardDiff = () => {
    setDiffProposal(null)
    setCurrentCritiqueText('')
  }

  const handleRollback = async (targetVersion: number) => {
    try {
      const res = await rollbackTaskVersion(taskId, targetVersion)
      if (res.status === 'rolled_back') {
        setCurrentVersion(res.restored_version)
        const now = new Date().toLocaleTimeString()
        setVersionHistory((prev) => [
          ...prev,
          {
            version: res.restored_version,
            timestamp: now,
            description: `Rollback to v${res.restored_version}`,
          },
        ])
        handleRunBurst(10)
      }
    } catch (err) {
      console.error('Rollback error:', err)
    }
  }

  const handleExport = async () => {
    setIsExporting(true)
    try {
      const res = await exportPolicyBundle(taskId)
      setExportData(res)
      setIsExportModalOpen(true)
    } catch (err) {
      console.error('Export error:', err)
    } finally {
      setIsExporting(false)
    }
  }

  const handleSaveSettings = (newSettings: LLMSettings) => {
    setLlmSettings(newSettings)
    localStorage.setItem('rl_ide_llm_settings', JSON.stringify(newSettings))
  }

  const handleLoadBundleFromRegistry = (taskConfig: TaskConfig) => {
    setTaskId(taskConfig.task_id)
    setAdapter(taskConfig.adapter_name)
    setRewardDsl(taskConfig.reward_dsl)
    setCurrentVersion(1)
    setAvailableVersions([1])
    setPrompt(taskConfig.description || prompt)
    setIsRegistryModalOpen(false)
    handleRunBurst(15)
  }

  const selectedEpisode =
    episodes.find((e) => e.episode_id === selectedEpisodeId) ||
    (episodes.length > 0 ? episodes[episodes.length - 1] : null)

  return (
    <div className="flex flex-col h-screen w-screen bg-canvas text-gray-100 overflow-hidden font-sans select-none">
      {/* 1. Header Bar with Mode Switch */}
      <Header
        viewMode={viewMode}
        onViewModeChange={setViewMode}
        adapter={adapter}
        onAdapterChange={setAdapter}
        currentVersion={currentVersion}
        availableVersions={availableVersions}
        onRollback={handleRollback}
        driftStatus={driftStatus}
        onOpenSettings={() => setIsSettingsModalOpen(true)}
        onOpenRegistry={() => setIsRegistryModalOpen(true)}
        registryCount={registryCount}
        onExport={handleExport}
        isExporting={isExporting}
      />

      {/* 2. Main Workbench Body */}
      {viewMode === 'vibe' ? (
        /* ================================================================= */
        /* ✨ VIBE STUDIO WORKBENCH (Phase 3D Default)                        */
        /* ================================================================= */
        <div className="flex flex-col flex-1 overflow-hidden">
          {/* Hero Prompt & Ingest Bar */}
          <VibeCreatorBar
            prompt={prompt}
            onPromptChange={setPrompt}
            onLaunch={(p, dsId) => handleLaunchVibe(p, dsId)}
            isLaunching={isLaunchingVibe}
            activeDataset={activeDataset}
            onClearDataset={() => setActiveDataset(null)}
            onDatasetAttached={(ds) => {
              setActiveDataset(ds)
              setPrompt(`Audit and triage records from ${ds.name} applying detected business rules.`)
            }}
          />

          {/* Three-Panel Vibe Layout (28% | 44% | 28%) */}
          <div className="flex flex-1 overflow-hidden">
            {/* Left: Plain-English Behavior Rules & Mastery Meter */}
            <div className="w-[28%] min-w-[280px] h-full overflow-hidden">
              <VibeRulesPanel
                skillSummary={skillSummary}
                rules={vibeRules}
                onAdjustRuleImportance={handleAdjustRuleImportance}
                isAdjusting={isCoaching}
              />
            </div>

            {/* Center: Polymorphic Interactive Arena */}
            <div className="flex-1 h-full overflow-hidden">
              <PolymorphicArena
                uiMode={vibeUiMode}
                episodes={episodes}
                selectedEpisodeId={selectedEpisodeId}
                onSelectEpisode={setSelectedEpisodeId}
                onFeedback={handleFeedback}
                latestFrame={vibeLatestFrame}
                sessionId={vibeSessionId}
              />
            </div>

            {/* Right: Conversational AI Coach */}
            <div className="w-[28%] min-w-[300px] h-full overflow-hidden">
              <VibeCoachPanel
                messages={coachMessages}
                onSendCritique={handleSendCoachCritique}
                isCoaching={isCoaching}
              />
            </div>
          </div>
        </div>
      ) : (
        /* ================================================================= */
        /* 🛠️ PRO RL WORKBENCH (Existing Hardened v2.0 Developer IDE)        */
        /* ================================================================= */
        <div className="flex flex-1 overflow-hidden">
          {/* Left Panel: TASK DEFINITION & REWARD DSL (28%) */}
          <LeftPanel
            prompt={prompt}
            onPromptChange={setPrompt}
            onCompile={handleCompile}
            isCompiling={isCompiling}
            rewardDsl={rewardDsl}
            versionHistory={versionHistory}
            currentVersion={currentVersion}
            onRollback={handleRollback}
          />

          {/* Center Panel: LIVE AGENT (44%) */}
          <CenterPanel
            adapter={adapter}
            episodes={episodes}
            selectedEpisodeId={selectedEpisodeId}
            onSelectEpisode={setSelectedEpisodeId}
            onRunBurst={handleRunBurst}
            isRunningBurst={isRunningBurst}
            isPaused={isPaused}
            onTogglePause={() => setIsPaused((prev) => !prev)}
            onToggleLock={handleToggleLock}
            lockedEpisodeIds={lockedEpisodeIds}
          />

          {/* Right Panel: CRITIQUE & DIFF WORKBENCH (28%) */}
          <RightPanel
            adapter={adapter}
            selectedEpisode={selectedEpisode}
            onPreviewCritique={handlePreviewCritique}
            isPreviewing={isPreviewing}
            diffProposal={diffProposal}
            onApplyDiff={handleApplyDiff}
            isApplying={isApplying}
            onDiscardDiff={handleDiscardDiff}
            isGoldenLocked={
              selectedEpisode
                ? lockedEpisodeIds.includes(selectedEpisode.episode_id) ||
                  Boolean(selectedEpisode.locked)
                : false
            }
          />
        </div>
      )}

      {/* Modals */}
      <ExportModal
        isOpen={isExportModalOpen}
        onClose={() => setIsExportModalOpen(false)}
        exportData={exportData}
        taskId={taskId}
      />

      <SettingsModal
        isOpen={isSettingsModalOpen}
        onClose={() => setIsSettingsModalOpen(false)}
        settings={llmSettings}
        onSaveSettings={handleSaveSettings}
      />

      <RegistryModal
        isOpen={isRegistryModalOpen}
        onClose={() => setIsRegistryModalOpen(false)}
        onLoadBundle={handleLoadBundleFromRegistry}
      />
    </div>
  )
}

export default App
