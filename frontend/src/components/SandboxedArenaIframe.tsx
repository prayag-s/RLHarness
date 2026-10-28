import React, { useRef, useEffect, useState, useCallback } from 'react'
import {
  Tv,
  Gamepad2,
  Code,
  Layers,
  Sparkles,
  Maximize2,
  RefreshCw,
  Play,
  RotateCcw,
} from 'lucide-react'
import { interactSession } from '../services/api'
import { EpisodeLog } from '../types'

interface SandboxedArenaIframeProps {
  sessionId?: string
  arenaHtml?: string
  isTraining: boolean
  latestFrame?: Record<string, any>
  episodes: EpisodeLog[]
  onOpenCodeInspector: () => void
}

export const SandboxedArenaIframe: React.FC<SandboxedArenaIframeProps> = ({
  sessionId,
  arenaHtml,
  isTraining,
  latestFrame,
  episodes,
  onOpenCodeInspector,
}) => {
  const iframeRef = useRef<HTMLIFrameElement>(null)
  const [mode, setMode] = useState<'watch' | 'play'>('watch')
  const [selectedEpisodeIdx, setSelectedEpisodeIdx] = useState<number | null>(null)
  const [iframeLoaded, setIframeLoaded] = useState(false)

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

  // 2. Stream latestFrame into iframe when in 'watch' mode
  useEffect(() => {
    if (mode === 'watch' && latestFrame && iframeLoaded) {
      postArenaUpdate(latestFrame)
    }
  }, [latestFrame, mode, iframeLoaded, postArenaUpdate])

  // 3. Bi-Directional window.postMessage Bridge for ARENA_INTERACT
  useEffect(() => {
    const handleWindowMessage = async (event: MessageEvent) => {
      // Validate that message originates exclusively from our sandboxed iframe
      if (iframeRef.current && event.source !== iframeRef.current.contentWindow) return
      if (!event.data || typeof event.data !== 'object') return

      if (event.data.type === 'ARENA_INTERACT' && sessionId) {
        try {
          const res = await interactSession(sessionId, event.data.payload)
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
  }, [sessionId, postArenaUpdate])

  // 4. Handle Episode Replay Scrubber
  const handleSelectEpisode = (ep: EpisodeLog, idx: number) => {
    setSelectedEpisodeIdx(idx)
    setMode('watch')
    postArenaUpdate({
      total_reward: ep.total_reward,
      reward_breakdown: ep.reward_breakdown || {},
      is_success: ep.is_success,
      arena_state: {
        episode_id: ep.episode_id,
        replayed: true,
      },
    })
  }

  return (
    <main className="flex-1 flex flex-col h-full bg-gray-950 overflow-hidden">
      {/* Top Arena Toolbar */}
      <div className="h-12 border-b border-gray-800 bg-gray-900/90 px-4 flex items-center justify-between z-10">
        {/* Mode Switcher */}
        <div className="flex items-center space-x-2">
          <div className="flex items-center bg-gray-950 border border-gray-800 rounded-lg p-0.5 text-xs font-semibold">
            <button
              onClick={() => {
                setMode('watch')
                setSelectedEpisodeIdx(null)
              }}
              className={`flex items-center space-x-1.5 px-3 py-1 rounded-md transition ${
                mode === 'watch'
                  ? 'bg-indigo-600 text-white shadow-sm'
                  : 'text-gray-400 hover:text-gray-200'
              }`}
            >
              <Tv className="h-3.5 w-3.5" />
              <span>📺 Watch Live Training</span>
            </button>
            <button
              onClick={() => {
                setMode('play')
                setSelectedEpisodeIdx(null)
              }}
              className={`flex items-center space-x-1.5 px-3 py-1 rounded-md transition ${
                mode === 'play'
                  ? 'bg-emerald-600 text-white shadow-sm'
                  : 'text-gray-400 hover:text-gray-200'
              }`}
            >
              <Gamepad2 className="h-3.5 w-3.5" />
              <span>🎮 Interact / Play Live</span>
            </button>
          </div>

          {isTraining && (
            <span className="flex items-center space-x-1 text-xs text-emerald-400 font-mono bg-emerald-950/60 border border-emerald-800/40 px-2 py-0.5 rounded-full">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
              <span>STREAMING EPISODES</span>
            </span>
          )}
        </div>

        {/* Action Buttons: Code Inspector Drawer Toggle */}
        <div className="flex items-center space-x-2">
          <button
            onClick={onOpenCodeInspector}
            className="flex items-center space-x-1.5 text-xs text-gray-300 hover:text-white px-2.5 py-1 rounded-lg bg-gray-800 hover:bg-gray-700 border border-gray-700 transition"
            title="Inspect synthesized Python env_module.py and arena.html"
          >
            <Code className="h-3.5 w-3.5 text-cyan-400" />
            <span>🔍 Inspect Synthesized Code</span>
          </button>
        </div>
      </div>

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
                const type = match[2] === 'eval' ? 'Eval' : 'Train'
                const epNum = match[3]
                chipLabel = `R${rNum} • ${type} #${epNum}`
              } else if (ep.round) {
                chipLabel = `R${ep.round} • #${epIdStr}`
              }

              return (
                <button
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
              if (latestFrame) postArenaUpdate(latestFrame)
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
                Describe your agent in the Conversational Architect panel or drop a dataset.
                The IDE will autonomously synthesize an isolated Gymnasium environment (<code className="text-indigo-300">env_module.py</code>)
                and an interactive HTML5/SVG arena (<code className="text-indigo-300">arena.html</code>) in real time.
              </p>
            </div>
            <div className="flex items-center justify-center space-x-2 text-[11px] text-gray-500 font-mono">
              <span className="px-2 py-0.5 rounded bg-gray-900 border border-gray-800">Zero-Adapter JIT</span>
              <span>•</span>
              <span className="px-2 py-0.5 rounded bg-gray-900 border border-gray-800">Self-Healing Smoke Test</span>
              <span>•</span>
              <span className="px-2 py-0.5 rounded bg-gray-900 border border-gray-800">ONNX Export</span>
            </div>
          </div>
        )}
      </div>
    </main>
  )
}
