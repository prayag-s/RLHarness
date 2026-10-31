import React, { useRef, useEffect, useState, useCallback } from 'react'
import {
  Tv,
  Gamepad2,
  Code,
  Sparkles,
  Palette,
  Lock,
  Unlock,
  Play,
  RotateCcw,
  AlertTriangle,
} from 'lucide-react'
import { interactSession } from '../services/api'
import { ArenaSource, EpisodeLog, ObservedStep } from '../types'

interface SandboxedArenaIframeProps {
  sessionId?: string
  arenaHtml?: string
  isTraining: boolean
  isArenaLocked?: boolean
  latestFrame?: Record<string, any>
  episodes: EpisodeLog[]
  onOpenCodeInspector: () => void
  onLockArena?: () => void
  onEditArenaLook?: () => void
  /** How the current arena was produced, and whether a bespoke canvas is still on its way. */
  arenaSource?: ArenaSource | null
  arenaPending?: boolean
  arenaError?: string | null
  onRetryArena?: () => void
  /** Reports the observation behind each interactive step, so the user can lock it as a test. */
  onStepObserved?: (step: ObservedStep) => void
}

export const SandboxedArenaIframe: React.FC<SandboxedArenaIframeProps> = ({
  sessionId,
  arenaHtml,
  isTraining,
  isArenaLocked = true,
  latestFrame,
  episodes,
  onOpenCodeInspector,
  onLockArena,
  onEditArenaLook,
  arenaSource = null,
  arenaPending = false,
  arenaError = null,
  onRetryArena,
  onStepObserved,
}) => {
  const iframeRef = useRef<HTMLIFrameElement>(null)
  const [mode, setMode] = useState<'mock' | 'watch' | 'play'>(isArenaLocked ? 'watch' : 'mock')
  const [selectedEpisodeIdx, setSelectedEpisodeIdx] = useState<number | null>(null)
  const [iframeLoaded, setIframeLoaded] = useState(false)

  // Sync mode if arena lock status changes
  useEffect(() => {
    if (!isArenaLocked) {
      setMode('mock')
    } else if (mode === 'mock') {
      setMode('watch')
    }
  }, [isArenaLocked])

  // 1. Send ARENA_UPDATE to iframe contentWindow
  const postArenaUpdate = useCallback((payload: Record<string, any>) => {
    if (!iframeRef.current || !iframeRef.current.contentWindow) return
    iframeRef.current.contentWindow.postMessage(
      {
        type: 'ARENA_UPDATE',
        ...payload,
        payload,
      },
      '*',
    )
  }, [])

  // 2. Dispatch mode to iframe
  const setArenaMode = useCallback((newMode: 'mock_preview' | 'live' | 'interactive') => {
    if (!iframeRef.current || !iframeRef.current.contentWindow) return
    iframeRef.current.contentWindow.postMessage(
      {
        type: 'SET_ARENA_MODE',
        mode: newMode,
      },
      '*',
    )
  }, [])

  const handleSwitchMode = (newMode: 'mock' | 'watch' | 'play') => {
    setMode(newMode)
    setSelectedEpisodeIdx(null)
    if (newMode === 'mock') {
      setArenaMode('mock_preview')
    } else if (newMode === 'watch') {
      setArenaMode('live')
    } else {
      setArenaMode('interactive')
    }
  }

  // 3. Stream latestFrame into iframe when in 'watch' mode
  useEffect(() => {
    if (mode === 'watch' && latestFrame && iframeLoaded) {
      postArenaUpdate(latestFrame)
    }
  }, [latestFrame, mode, iframeLoaded, postArenaUpdate])

  // 4. Bi-Directional window.postMessage Bridge for ARENA_INTERACT
  useEffect(() => {
    const handleWindowMessage = async (event: MessageEvent) => {
      // Validate that message originates exclusively from our sandboxed iframe
      if (iframeRef.current && event.source !== iframeRef.current.contentWindow) return
      if (!event.data || typeof event.data !== 'object') return

      if (event.data.type === 'ARENA_INTERACT' && sessionId) {
        try {
          const res = await interactSession(sessionId, event.data.payload)
          // The backend echoes the encoded observation behind this step; without it a "this was
          // wrong" judgement could not become an executable assertion.
          const obs: number[] = Array.isArray(res.obs_vector)
            ? res.obs_vector.map((v: any) => Number(v))
            : []
          if (obs.length > 0 && onStepObserved) {
            const probs: number[] = Array.isArray(res.action_probs) ? res.action_probs : []
            let chosen: number | null = null
            if (probs.length > 0) {
              chosen = probs.reduce((best, p, i) => (p > probs[best] ? i : best), 0)
            }
            onStepObserved({
              obs,
              action: chosen,
              probs,
              step:
                typeof res.step === 'number'
                  ? res.step
                  : typeof res.arena_state?.move_count === 'number'
                    ? res.arena_state.move_count
                    : null,
            })
          }
          // Send updated state back to iframe
          postArenaUpdate({
            arena_state: res.arena_state || {},
            action_probs: res.action_probs || [],
            total_reward: res.reward || 0,
            reward_breakdown: res.reward_breakdown || {},
            is_success: res.is_success,
            terminated: res.terminated,
          })
        } catch (err) {
          console.error('Interactive step failed:', err)
        }
      }
    }

    window.addEventListener('message', handleWindowMessage)
    return () => {
      window.removeEventListener('message', handleWindowMessage)
    }
  }, [sessionId, postArenaUpdate, onStepObserved])

  // 5. Handle Episode Replay Scrubber
  const handleSelectEpisode = (ep: EpisodeLog, idx: number) => {
    setSelectedEpisodeIdx(idx)
    setMode('watch')
    setArenaMode('live')
    // Replay the recorded frames so the stage shows real state instead of an empty payload.
    const steps = Array.isArray(ep.steps) ? ep.steps : []
    if (steps.length > 0) {
      let stepIdx = 0
      const renderFrame = () => {
        const step = steps[stepIdx]
        postArenaUpdate({
          episode_id: ep.episode_id,
          step: step?.step ?? stepIdx + 1,
          total_reward: step?.reward ?? ep.total_reward,
          reward_breakdown: step?.reward_breakdown || ep.reward_breakdown || {},
          is_success: step?.is_success ?? ep.is_success,
          arena_state: step?.arena_state || {},
        })
        stepIdx += 1
        if (stepIdx < steps.length) {
          window.setTimeout(renderFrame, 500)
        }
      }
      renderFrame()
    } else {
      postArenaUpdate({
        episode_id: ep.episode_id,
        total_reward: ep.total_reward,
        reward_breakdown: ep.reward_breakdown || {},
        is_success: ep.is_success,
      })
    }
  }

  return (
    <main className="flex-1 flex flex-col h-full bg-gray-950 overflow-hidden">
      {/* Top Arena Toolbar */}
      <div className="h-12 border-b border-gray-800 bg-gray-900/90 px-4 flex items-center justify-between z-10">
        {/* Mode Switcher */}
        <div className="flex items-center space-x-2">
          <div className="flex items-center bg-gray-950 border border-gray-800 rounded-lg p-0.5 text-xs font-semibold">
            <button
              type="button"
              onClick={() => handleSwitchMode('mock')}
              className={`flex items-center space-x-1.5 px-2.5 py-1 rounded-md transition ${
                mode === 'mock'
                  ? 'bg-amber-600 text-white shadow-sm'
                  : 'text-gray-400 hover:text-gray-200'
              }`}
              title="Preview with continuous simulated moves before training"
            >
              <Sparkles className="h-3.5 w-3.5" />
              <span>✨ Mock Preview</span>
            </button>
            <button
              type="button"
              onClick={() => handleSwitchMode('watch')}
              className={`flex items-center space-x-1.5 px-2.5 py-1 rounded-md transition ${
                mode === 'watch'
                  ? 'bg-indigo-600 text-white shadow-sm'
                  : 'text-gray-400 hover:text-gray-200'
              }`}
              title="Stream real training episodes live"
            >
              <Tv className="h-3.5 w-3.5" />
              <span>📺 Live Stream</span>
            </button>
            <button
              type="button"
              onClick={() => handleSwitchMode('play')}
              className={`flex items-center space-x-1.5 px-2.5 py-1 rounded-md transition ${
                mode === 'play'
                  ? 'bg-emerald-600 text-white shadow-sm'
                  : 'text-gray-400 hover:text-gray-200'
              }`}
              title="Test agent interactively"
            >
              <Gamepad2 className="h-3.5 w-3.5" />
              <span>🎮 Play / Interact</span>
            </button>
          </div>

          {isTraining && (
            <span className="flex items-center space-x-1 text-xs text-emerald-400 font-mono bg-emerald-950/60 border border-emerald-800/40 px-2 py-0.5 rounded-full">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
              <span>STREAMING EPISODES</span>
            </span>
          )}

          {/* Arena provenance: a preview must never be mistaken for the bespoke canvas, and a
              synthesis in flight must never look like a finished (or broken) generator. */}
          {arenaPending && (
            <span
              className="flex items-center space-x-1.5 text-[11px] text-amber-300 font-mono bg-amber-950/60 border border-amber-700/50 px-2 py-0.5 rounded-full"
              title="The deterministic preview is live and playable. The bespoke canvas replaces it automatically when it arrives."
            >
              <Palette className="h-3 w-3 animate-pulse" />
              <span>🎨 Synthesizing bespoke domain canvas (~90s) — showing live telemetry preview</span>
            </span>
          )}

          {!arenaPending && arenaError && (
            <span className="flex items-center space-x-1.5 text-[11px] text-red-300 font-mono bg-red-950/60 border border-red-700/50 px-2 py-0.5 rounded-full">
              <AlertTriangle className="h-3 w-3" />
              <span>Bespoke canvas failed</span>
              {onRetryArena && (
                <button
                  type="button"
                  onClick={onRetryArena}
                  className="ml-1 px-1.5 py-0.5 rounded bg-red-900/70 hover:bg-red-800 text-red-100 border border-red-700/60 transition"
                  title={arenaError}
                >
                  Retry
                </button>
              )}
            </span>
          )}

          {!arenaPending && arenaSource === 'preview' && !arenaError && (
            <span className="text-[11px] text-gray-400 font-mono bg-gray-800/70 border border-gray-700 px-2 py-0.5 rounded-full">
              Generic preview
            </span>
          )}
        </div>

        {/* Action Buttons: Quick Restyle & Code Inspector */}
        <div className="flex items-center space-x-2">
          {onEditArenaLook && (
            <button
              type="button"
              onClick={onEditArenaLook}
              className="flex items-center space-x-1.5 text-xs text-amber-300 hover:text-white px-2.5 py-1 rounded-lg bg-amber-950/40 hover:bg-amber-900/60 border border-amber-700/50 transition font-medium"
              title="Ask in chat to change themes, colors, or components"
            >
              <Palette className="h-3.5 w-3.5 text-amber-400" />
              <span>🎨 Edit Arena Look</span>
            </button>
          )}

          <button
            type="button"
            onClick={onOpenCodeInspector}
            className="flex items-center space-x-1.5 text-xs text-gray-300 hover:text-white px-2.5 py-1 rounded-lg bg-gray-800 hover:bg-gray-700 border border-gray-700 transition"
            title="Inspect synthesized Python env_module.py and arena.html"
          >
            <Code className="h-3.5 w-3.5 text-cyan-400" />
            <span>🔍 Inspect Code</span>
          </button>
        </div>
      </div>

      {/* Prominent Banner when Arena is Not Locked */}
      {!isArenaLocked && arenaHtml && (
        <div className="bg-gradient-to-r from-amber-950/90 via-purple-950/90 to-cyan-950/90 border-b border-amber-600/40 px-4 py-2 flex items-center justify-between z-10 text-xs shadow-md">
          <div className="flex items-center space-x-2 text-amber-200">
            <Sparkles className="h-4 w-4 text-amber-400 shrink-0 animate-pulse" />
            <span className="font-medium">
              {arenaSource === 'bespoke'
                ? 'Bespoke canvas, mock data & animation — ask in chat to tweak the look, or lock when ready.'
                : 'Generic preview built from the declared actions and state (not the bespoke canvas yet). You can lock and train on it now.'}
            </span>
          </div>
          {onLockArena && (
            <button
              type="button"
              onClick={onLockArena}
              className="px-3 py-1 bg-emerald-600 hover:bg-emerald-500 text-white font-bold rounded-lg shadow-md transition flex items-center space-x-1.5 text-xs active:scale-95"
            >
              <Lock className="h-3.5 w-3.5" />
              <span>🔒 Lock Arena & Start Training</span>
            </button>
          )}
        </div>
      )}

      {/* Episode Replay Scrubber Strip (if episodes exist) */}
      {episodes.length > 0 && (
        <div className="h-8 border-b border-gray-800/80 bg-gray-900/50 px-3 flex items-center space-x-2 overflow-x-auto text-[11px] text-gray-400 scrollbar-thin">
          <span className="shrink-0 font-medium text-gray-500 uppercase tracking-wider text-[10px]">
            Replay Episodes:
          </span>
          <div className="flex items-center space-x-1">
            {episodes.slice(-20).map((ep, idx) => {
              const isSelected = selectedEpisodeIdx === idx
              const epIdStr = String(ep.episode_id ?? idx + 1)
              let chipLabel = `#${epIdStr}`
              const match = epIdStr.match(/^r(\d+)_(eval|train)_(\d+)$/)
              if (match) {
                const rNum = match[1]
                const epType = match[2] === 'eval' ? 'Eval' : 'Train'
                const epNum = match[3]
                chipLabel = `R${rNum} • ${epType} #${epNum}`
              } else if (ep.round) {
                chipLabel = `R${ep.round} • #${epIdStr}`
              }

              return (
                <button
                  type="button"
                  key={`${ep.round || 0}_${ep.episode_id}_${idx}`}
                  onClick={() => handleSelectEpisode(ep, idx)}
                  className={`px-2 py-0.5 rounded font-mono text-[10px] whitespace-nowrap transition border ${
                    isSelected
                      ? 'bg-indigo-600 border-indigo-400 text-white'
                      : ep.is_success
                      ? 'bg-emerald-950/50 border-emerald-800/40 text-emerald-400 hover:border-emerald-600'
                      : 'bg-gray-800 border-gray-700 text-gray-400 hover:text-gray-200'
                  }`}
                  title={`Episode #${ep.episode_id}: Return ${ep.total_reward.toFixed(2)}`}
                >
                  {chipLabel}
                </button>
              )
            })}
          </div>
        </div>
      )}

      {/* Center Sandboxed Iframe Stage */}
      <div className="flex-1 relative w-full h-full bg-gray-950 flex items-center justify-center">
        {arenaHtml ? (
          <iframe
            ref={iframeRef}
            title="Sandboxed Dynamic Arena"
            sandbox="allow-scripts"
            srcDoc={arenaHtml}
            onLoad={() => {
              setIframeLoaded(true)
              if (mode === 'mock') {
                setArenaMode('mock_preview')
              } else if (latestFrame) {
                postArenaUpdate(latestFrame)
              }
            }}
            className="w-full h-full border-none bg-gray-950"
          />
        ) : (
          /* Empty / Initial State Hero */
          <div className="max-w-md text-center p-8 space-y-4 text-gray-400">
            <div className="h-16 w-16 mx-auto rounded-2xl bg-indigo-950/60 border border-indigo-800/50 flex items-center justify-center text-indigo-400 shadow-xl shadow-indigo-950/50">
              <Sparkles className="h-8 w-8" />
            </div>
            <div>
              <h3 className="text-lg font-bold text-gray-200">Universal Sandboxed Arena</h3>
              <p className="text-xs text-gray-400 mt-1 leading-relaxed">
                Describe your agent in the Conversational Studio below. The IDE will instantly generate a live mock-animated
                Arena UI (<code className="text-indigo-300">arena.html</code>) in the center panel before training begins.
              </p>
            </div>
            <div className="flex items-center justify-center space-x-2 text-[11px] text-gray-500 font-mono">
              <span className="px-2 py-0.5 rounded bg-gray-900 border border-gray-800">Zero-Training Preview</span>
              <span>•</span>
              <span className="px-2 py-0.5 rounded bg-gray-900 border border-gray-800">Live Mock Engine</span>
              <span>•</span>
              <span className="px-2 py-0.5 rounded bg-gray-900 border border-gray-800">Continuous Resumability</span>
            </div>
          </div>
        )}
      </div>
    </main>
  )
}
