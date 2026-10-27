import React, { useState, useEffect, useRef } from 'react'
import { Header } from './components/Header'
import { LeftPanel } from './components/LeftPanel'
import { CenterPanel } from './components/CenterPanel'
import { RightPanel } from './components/RightPanel'
import { ExportModal } from './components/ExportModal'
import { SettingsModal } from './components/SettingsModal'
import { RegistryModal } from './components/RegistryModal'
import {
  RewardDSL,
  EpisodeLog,
  RewardDiffProposal,
  ExportResponse,
  VersionRecord,
  LLMSettings,
  TaskConfig,
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
} from './services/api'

const DEFAULT_PROMPT =
  'Detect sudden volatility spikes and price jumps in the financial stream. Reward catching real anomalies, heavily penalize false alarms when volatility is normal, and enforce cooldown respect.'

export const App: React.FC = () => {
  // State: Task & Adapter
  const [adapter, setAdapter] = useState('CSVStreamAdapter')
  const [taskId, setTaskId] = useState('task_csvstreamadapter')
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

  // Action: Compile Task from Natural Language
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

        // Automatically trigger first micro-burst after compilation
        const burstRes = await runTrainingBurst(taskId, 15)
        if (burstRes.episodes?.length) {
          setEpisodes(burstRes.episodes)
          setSelectedEpisodeId(burstRes.episodes[burstRes.episodes.length - 1].episode_id)
        }
      }
    } catch (err: any) {
      alert(`Compilation failed: ${err.message || 'Server error'}`)
    } finally {
      setIsCompiling(false)
    }
  }

  // Action: Run 15-Ep Burst
  const handleRunBurst = async (nEpisodes = 15) => {
    if (isRunningBurst) return
    setIsRunningBurst(true)
    try {
      const res = await runTrainingBurst(taskId, nEpisodes)
      if (res.episodes?.length) {
        setEpisodes(res.episodes)
        setSelectedEpisodeId(res.episodes[res.episodes.length - 1].episode_id)
      }
    } catch (err: any) {
      alert(`Burst failed: ${err.message || 'Make sure task is compiled first'}`)
    } finally {
      setIsRunningBurst(false)
    }
  }

  // Action: Toggle Golden Episode Lock
  const handleToggleLock = async (epId: number) => {
    try {
      const res = await toggleLockEpisode(epId)
      setLockedEpisodeIds(res.locked_episodes)
      setEpisodes((prev) =>
        prev.map((e) => (e.episode_id === epId ? { ...e, locked: res.locked } : e)),
      )
    } catch (err: any) {
      console.error('Lock error:', err)
    }
  }

  // Action: Preview Critique (Two-Stage)
  const handlePreviewCritique = async (critique: string) => {
    if (!selectedEpisodeId) return
    setIsPreviewing(true)
    setCurrentCritiqueText(critique)
    try {
      const proposal = await previewCritique(taskId, selectedEpisodeId, critique, llmSettings)
      setDiffProposal(proposal)
    } catch (err: any) {
      alert(`Critique Preview Failed: ${err.message}`)
    } finally {
      setIsPreviewing(false)
    }
  }

  // Action: Apply Confirmed Diff
  const handleApplyDiff = async () => {
    if (!diffProposal) return
    setIsApplying(true)
    try {
      const res = await applyCritiqueDiff(
        taskId,
        diffProposal.new_dsl,
        currentCritiqueText,
        true,
      )

      const nextVersion = res.version ?? currentVersion + 1
      setCurrentVersion(nextVersion)
      setAvailableVersions((prev) =>
        prev.includes(nextVersion) ? prev : [...prev, nextVersion],
      )
      setRewardDsl(diffProposal.new_dsl)

      const now = new Date().toLocaleTimeString()
      setVersionHistory((prev) => [
        {
          version: nextVersion,
          timestamp: now,
          critique_text: currentCritiqueText,
          description: `Applied critique: "${currentCritiqueText}"`,
        },
        ...prev,
      ])

      setDiffProposal(null)
      setCurrentCritiqueText('')

      // Run follow-up burst to show updated policy performance
      const burstRes = await runTrainingBurst(taskId, 15)
      if (burstRes.episodes?.length) {
        setEpisodes(burstRes.episodes)
        setSelectedEpisodeId(burstRes.episodes[burstRes.episodes.length - 1].episode_id)
      }
    } catch (err: any) {
      alert(`Failed to apply diff: ${err.message}`)
    } finally {
      setIsApplying(false)
    }
  }

  // Action: Discard Diff
  const handleDiscardDiff = () => {
    setDiffProposal(null)
    setCurrentCritiqueText('')
  }

  // Action: Rollback to previous version
  const handleRollback = async (version: number) => {
    try {
      const res = await rollbackTaskVersion(taskId, version)
      setCurrentVersion(res.restored_version)
      const now = new Date().toLocaleTimeString()
      setVersionHistory((prev) => [
        {
          version: res.restored_version,
          timestamp: now,
          description: `Rolled back to v${res.restored_version}`,
        },
        ...prev,
      ])
      // Trigger burst on restored version
      const burstRes = await runTrainingBurst(taskId, 10)
      if (burstRes.episodes?.length) {
        setEpisodes(burstRes.episodes)
        setSelectedEpisodeId(burstRes.episodes[burstRes.episodes.length - 1].episode_id)
      }
    } catch (err: any) {
      alert(`Rollback failed: ${err.message}`)
    }
  }

  // Action: Export Owned Policy (.runbundle)
  const handleExport = async () => {
    setIsExporting(true)
    try {
      const res = await exportPolicyBundle(taskId)
      setExportData(res)
      setIsExportModalOpen(true)
      // Refresh registry count
      fetchBundles().then((b) => setRegistryCount(b.length)).catch(() => {})
    } catch (err: any) {
      alert(`Export failed: ${err.message || 'Make sure task is compiled and trained.'}`)
    } finally {
      setIsExporting(false)
    }
  }

  // Action: Save LLM BYOK Settings
  const handleSaveSettings = (newSettings: LLMSettings) => {
    setLlmSettings(newSettings)
    localStorage.setItem('rl_ide_llm_settings', JSON.stringify(newSettings))
  }

  // Action: Load Bundle from Registry into Workbench
  const handleLoadBundleFromRegistry = async (cfg: any) => {
    if (!cfg) return
    setAdapter(cfg.adapter_name || 'CSVStreamAdapter')
    setTaskId(cfg.task_id || `task_${(cfg.adapter_name || '').toLowerCase()}`)
    if (cfg.description) setPrompt(cfg.description)
    if (cfg.reward_dsl) setRewardDsl(cfg.reward_dsl)
    setCurrentVersion(1)
    setAvailableVersions([1])
    setVersionHistory([
      {
        version: 1,
        timestamp: new Date().toLocaleTimeString(),
        description: `Imported policy bundle: ${cfg.name || cfg.task_id}`,
      },
    ])
    // Run initial burst with the loaded policy
    const burstRes = await runTrainingBurst(cfg.task_id, 15)
    if (burstRes.episodes?.length) {
      setEpisodes(burstRes.episodes)
      setSelectedEpisodeId(burstRes.episodes[burstRes.episodes.length - 1].episode_id)
    }
  }

  const selectedEpisode =
    episodes.find((e) => e.episode_id === selectedEpisodeId) ||
    episodes[episodes.length - 1] ||
    null

  return (
    <div className="h-screen w-screen flex flex-col bg-background text-gray-100 overflow-hidden font-sans select-none">
      {/* 1. Header Bar */}
      <Header
        adapter={adapter}
        onAdapterChange={(val) => {
          setAdapter(val)
          setTaskId(`task_${val.toLowerCase()}`)
          if (val === 'TextClassifyAdapter') {
            setPrompt(
              'Classify support tickets into urgent, normal, spam, and escalate based on urgency keywords, angry tone, and VIP status. Reward accurate triage, heavily penalize routing spam as urgent, and ensure angry VIP tickets are escalated.',
            )
          } else {
            setPrompt(DEFAULT_PROMPT)
          }
          setEpisodes([])
          setSelectedEpisodeId(null)
          setRewardDsl(null)
          setDiffProposal(null)
        }}
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

      {/* 2. Three-Panel Workbench Layout */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Panel: TASK & REWARD DSL (28%) */}
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

      {/* 3. Export Modal */}
      <ExportModal
        isOpen={isExportModalOpen}
        onClose={() => setIsExportModalOpen(false)}
        exportData={exportData}
        taskId={taskId}
      />

      {/* 4. Settings Modal */}
      <SettingsModal
        isOpen={isSettingsModalOpen}
        onClose={() => setIsSettingsModalOpen(false)}
        settings={llmSettings}
        onSaveSettings={handleSaveSettings}
      />

      {/* 5. Policy Registry Modal */}
      <RegistryModal
        isOpen={isRegistryModalOpen}
        onClose={() => setIsRegistryModalOpen(false)}
        onLoadBundle={handleLoadBundleFromRegistry}
      />
    </div>
  )
}

export default App
