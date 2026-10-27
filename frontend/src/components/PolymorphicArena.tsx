import React, { useState } from 'react'
import {
  Gamepad2,
  Navigation,
  FileText,
  Database,
  TrendingUp,
  ThumbsUp,
  ThumbsDown,
  RotateCcw,
  Lock,
  Unlock,
  Radio,
  Crosshair,
  Sparkles,
  Zap,
} from 'lucide-react'
import { EpisodeLog } from '../types'
import { interactSim } from '../services/api'

interface PolymorphicArenaProps {
  uiMode: string
  episodes: EpisodeLog[]
  selectedEpisodeId: number | null
  onSelectEpisode: (id: number) => void
  onFeedback: (type: 'thumbs_up' | 'thumbs_down', episodeId: number) => void
  latestFrame?: Record<string, any>
  sessionId?: string
}

export const PolymorphicArena: React.FC<PolymorphicArenaProps> = ({
  uiMode,
  episodes,
  selectedEpisodeId,
  onSelectEpisode,
  onFeedback,
  latestFrame,
  sessionId,
}) => {
  // Live Interactive Play states for Tic-Tac-Toe
  const [isLiveGameMode, setIsLiveGameMode] = useState(false)
  const [liveBoard, setLiveBoard] = useState<number[]>([0, 0, 0, 0, 0, 0, 0, 0, 0])
  const [liveWinner, setLiveWinner] = useState<string>('none')
  const [liveProbs, setLiveProbs] = useState<number[]>([0, 0, 0, 0, 0, 0, 0, 0, 0])
  const [isInteracting, setIsInteracting] = useState(false)

  // Live Interactive states for Robot Rover
  const [isInteractiveRover, setIsInteractiveRover] = useState(false)
  const [liveTarget, setLiveTarget] = useState<{ x: number; y: number }>({ x: 75, y: 75 })
  const [liveRobot, setLiveRobot] = useState<any>({ x: 20, y: 20, vx: 0, vy: 0, battery: 100 })
  const [liveObstacles, setLiveObstacles] = useState<Array<{ x: number; y: number; radius: number }>>([
    { x: 45, y: 45, radius: 8 },
    { x: 30, y: 65, radius: 9 },
  ])
  const [liveTrail, setLiveTrail] = useState<Array<[number, number]>>([])

  const currentEp = episodes.find((e) => e.episode_id === selectedEpisodeId) || episodes[episodes.length - 1]
  const lastStep = currentEp?.steps?.[currentEp.steps.length - 1]
  const events = lastStep?.events || {}

  // ===========================================================================
  // 1. Tic-Tac-Toe Live Sparring Handlers
  // ===========================================================================
  const handleCellClick = async (cellIdx: number) => {
    if (!isLiveGameMode || liveBoard[cellIdx] !== 0 || liveWinner !== 'none' || isInteracting) {
      return
    }

    setIsInteracting(true)
    try {
      const res = await interactSim({
        sessionId,
        boardState: liveBoard,
        humanCellIdx: cellIdx,
      })
      if (res.board) {
        setLiveBoard(res.board)
      }
      if (res.winner) {
        setLiveWinner(res.winner)
      }
      if (res.action_probabilities) {
        setLiveProbs(res.action_probabilities)
      }
    } catch (err) {
      console.error('Tic-Tac-Toe live interact error:', err)
    } finally {
      setIsInteracting(false)
    }
  }

  const resetLiveGame = () => {
    setLiveBoard([0, 0, 0, 0, 0, 0, 0, 0, 0])
    setLiveWinner('none')
    setLiveProbs([0, 0, 0, 0, 0, 0, 0, 0, 0])
  }

  // ===========================================================================
  // 2. Robot Rover Live Click to Move Target
  // ===========================================================================
  const handleArenaClick = async (e: React.MouseEvent<SVGSVGElement>) => {
    if (!isInteractiveRover || isInteracting) return

    const rect = e.currentTarget.getBoundingClientRect()
    const clickX = ((e.clientX - rect.left) / rect.width) * 100
    const clickY = ((e.clientY - rect.top) / rect.height) * 100

    const newTarget = { x: Math.round(clickX), y: Math.round(clickY) }
    setLiveTarget(newTarget)

    setIsInteracting(true)
    try {
      const res = await interactSim({
        sessionId,
        robotState: liveRobot,
        targetXy: [newTarget.x, newTarget.y],
        obstaclesList: liveObstacles,
        nFrames: 6,
      })
      if (res.robot) setLiveRobot(res.robot)
      if (res.trajectory) setLiveTrail(res.trajectory)
    } catch (err) {
      console.error('Robot Rover live interact error:', err)
    } finally {
      setIsInteracting(false)
    }
  }

  return (
    <div className="flex flex-col h-full bg-canvas overflow-hidden">
      {/* Polymorphic Header Bar */}
      <div className="h-12 border-b border-panel-border bg-panel px-4 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-2">
            {uiMode === 'board_grid' && <Gamepad2 className="h-4 w-4 text-purple-400" />}
            {uiMode === 'robot_canvas' && <Navigation className="h-4 w-4 text-cyan-400" />}
            {uiMode === 'document_reader' && <FileText className="h-4 w-4 text-amber-400" />}
            {uiMode === 'tabular_inspector' && <Database className="h-4 w-4 text-emerald-400" />}
            {uiMode !== 'board_grid' && uiMode !== 'robot_canvas' && uiMode !== 'document_reader' && uiMode !== 'tabular_inspector' && (
              <TrendingUp className="h-4 w-4 text-blue-400" />
            )}
            <span className="text-xs font-bold text-gray-200 uppercase tracking-wider">
              {uiMode.replace('_', ' ')} Arena
            </span>
          </div>

          {/* Episode Tabs */}
          {episodes.length > 0 && (
            <div className="flex items-center space-x-1 overflow-x-auto max-w-[240px] no-scrollbar">
              {episodes.slice(-5).map((ep) => (
                <button
                  key={ep.episode_id}
                  onClick={() => onSelectEpisode(ep.episode_id)}
                  className={`px-2 py-0.5 rounded text-[11px] font-mono transition ${
                    (selectedEpisodeId || episodes[episodes.length - 1].episode_id) === ep.episode_id
                      ? 'bg-blue-600 text-white font-bold'
                      : 'bg-gray-800 text-gray-400 hover:text-gray-200'
                  }`}
                >
                  #{ep.episode_id}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Action / Feedback Buttons */}
        <div className="flex items-center space-x-2">
          {uiMode === 'board_grid' && (
            <button
              onClick={() => {
                setIsLiveGameMode(!isLiveGameMode)
                if (!isLiveGameMode) resetLiveGame()
              }}
              className={`px-2.5 py-1 rounded-lg text-xs font-semibold flex items-center space-x-1.5 transition ${
                isLiveGameMode
                  ? 'bg-purple-600 text-white shadow-lg shadow-purple-600/30'
                  : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
              }`}
            >
              <Gamepad2 className="h-3.5 w-3.5" />
              <span>{isLiveGameMode ? '🎮 Playing Live' : 'Play Live vs Agent'}</span>
            </button>
          )}

          {uiMode === 'robot_canvas' && (
            <button
              onClick={() => setIsInteractiveRover(!isInteractiveRover)}
              className={`px-2.5 py-1 rounded-lg text-xs font-semibold flex items-center space-x-1.5 transition ${
                isInteractiveRover
                  ? 'bg-cyan-600 text-white shadow-lg shadow-cyan-600/30'
                  : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
              }`}
            >
              <Crosshair className="h-3.5 w-3.5" />
              <span>{isInteractiveRover ? '🎯 Drag Target Active' : 'Interactive Target'}</span>
            </button>
          )}

          {/* Feedback Buttons */}
          <button
            onClick={() => onFeedback('thumbs_up', currentEp?.episode_id || 1)}
            title="Lock this episode as golden behavior reference"
            className="px-2.5 py-1 rounded-lg bg-emerald-950/60 hover:bg-emerald-900 border border-emerald-800 text-emerald-300 text-xs font-semibold flex items-center space-x-1 transition"
          >
            <ThumbsUp className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Good (Lock)</span>
          </button>
          <button
            onClick={() => onFeedback('thumbs_down', currentEp?.episode_id || 1)}
            title="Coach agent to fix mistakes in this episode"
            className="px-2.5 py-1 rounded-lg bg-rose-950/60 hover:bg-rose-900 border border-rose-800 text-rose-300 text-xs font-semibold flex items-center space-x-1 transition"
          >
            <ThumbsDown className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Fix This</span>
          </button>
        </div>
      </div>

      {/* Main Canvas Viewport */}
      <div className="flex-1 flex items-center justify-center p-6 overflow-auto">
        {/* =================================================================== */}
        {/* CARTRIDGE 1: Tic-Tac-Toe Grid                                      */}
        {/* =================================================================== */}
        {uiMode === 'board_grid' && (
          <div className="flex flex-col items-center space-y-4 max-w-sm w-full">
            {/* Status indicator */}
            <div className="flex items-center justify-between w-full px-2">
              <span className="text-xs font-bold text-gray-300">
                {isLiveGameMode ? (
                  liveWinner !== 'none' ? (
                    liveWinner === 'agent' ? (
                      <span className="text-cyan-400 font-extrabold">🏆 Agent (X) Won!</span>
                    ) : liveWinner === 'opponent' ? (
                      <span className="text-rose-400 font-extrabold">🎉 You Won (O)!</span>
                    ) : (
                      <span className="text-amber-400 font-extrabold">🤝 Draw Match!</span>
                    )
                  ) : (
                    <span className="text-gray-300">Your Turn: Click an open cell as (O)</span>
                  )
                ) : (
                  <span>
                    Episode #{currentEp?.episode_id || 1} Outcome:{' '}
                    <strong className="text-cyan-400">
                      {events.state?.winner === 'agent'
                        ? 'Agent Victory'
                        : events.state?.winner === 'draw'
                        ? 'Draw'
                        : 'Opponent'}
                    </strong>
                  </span>
                )}
              </span>

              {isLiveGameMode && (
                <button
                  onClick={resetLiveGame}
                  className="p-1 rounded hover:bg-gray-800 text-gray-400 hover:text-white transition"
                  title="Reset board"
                >
                  <RotateCcw className="h-3.5 w-3.5" />
                </button>
              )}
            </div>

            {/* 3x3 Grid */}
            <div className="grid grid-cols-3 gap-2.5 bg-gray-900/90 p-3 rounded-2xl border border-gray-800 shadow-2xl w-72 h-72">
              {(isLiveGameMode ? liveBoard : latestFrame?.board || [0, 0, 0, 0, 0, 0, 0, 0, 0]).map(
                (cellVal: number, idx: number) => {
                  const prob = liveProbs[idx] ? Math.round(liveProbs[idx] * 100) : null
                  return (
                    <button
                      key={idx}
                      onClick={() => handleCellClick(idx)}
                      disabled={!isLiveGameMode || cellVal !== 0 || liveWinner !== 'none'}
                      className={`relative rounded-xl border flex flex-col items-center justify-center transition-all ${
                        cellVal === 1
                          ? 'bg-cyan-950/40 border-cyan-700/60 shadow-lg shadow-cyan-950/50'
                          : cellVal === -1
                          ? 'bg-rose-950/40 border-rose-700/60 shadow-lg shadow-rose-950/50'
                          : 'bg-gray-800/60 border-gray-700/50 hover:bg-gray-800 hover:border-gray-600'
                      }`}
                    >
                      {cellVal === 1 && (
                        <span className="text-3xl font-extrabold text-cyan-400 font-mono">X</span>
                      )}
                      {cellVal === -1 && (
                        <span className="text-3xl font-extrabold text-rose-400 font-mono">O</span>
                      )}
                      {cellVal === 0 && prob !== null && prob > 0 && isLiveGameMode && (
                        <span className="text-[10px] text-gray-400 font-mono font-semibold">
                          {prob}%
                        </span>
                      )}
                    </button>
                  )
                }
              )}
            </div>

            <div className="flex items-center space-x-4 text-xs text-gray-400">
              <span className="flex items-center space-x-1.5">
                <span className="h-2 w-2 rounded-full bg-cyan-400" />
                <span>Agent (X)</span>
              </span>
              <span className="flex items-center space-x-1.5">
                <span className="h-2 w-2 rounded-full bg-rose-400" />
                <span>Opponent / Human (O)</span>
              </span>
            </div>
          </div>
        )}

        {/* =================================================================== */}
        {/* CARTRIDGE 2: 2D Robot Kinematics Arena                             */}
        {/* =================================================================== */}
        {uiMode === 'robot_canvas' && (
          <div className="flex flex-col items-center space-y-3 w-full max-w-xl">
            <div className="flex items-center justify-between w-full text-xs text-gray-300">
              <div className="flex items-center space-x-2">
                <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
                <span>
                  {isInteractiveRover
                    ? '🎯 Click arena to set waypoint target'
                    : '2D Continuous Rover Trajectory'}
                </span>
              </div>
              <div className="flex items-center space-x-3 font-mono text-[11px] text-gray-400">
                <span>Dist: {(events.obs?.dist_to_target ?? 32).toFixed(1)}m</span>
                <span>Jerk: {(events.obs?.jerk_magnitude ?? 0.2).toFixed(2)}</span>
                <span>Battery: {Math.round(events.obs?.battery_remaining ?? 95)}%</span>
              </div>
            </div>

            {/* SVG 500x350 Robot Canvas */}
            <div className="relative w-full aspect-[5/3.5] bg-gray-950 border border-gray-800 rounded-2xl shadow-2xl overflow-hidden">
              <svg
                viewBox="0 0 100 70"
                onClick={handleArenaClick}
                className="w-full h-full cursor-crosshair select-none"
              >
                {/* Grid Lines */}
                <defs>
                  <pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse">
                    <path d="M 10 0 L 0 0 0 10" fill="none" stroke="#1f2937" strokeWidth="0.4" />
                  </pattern>
                  <radialGradient id="targetGlow">
                    <stop offset="0%" stopColor="#38bdf8" stopOpacity="0.8" />
                    <stop offset="100%" stopColor="#38bdf8" stopOpacity="0" />
                  </radialGradient>
                </defs>
                <rect width="100" height="70" fill="url(#grid)" />

                {/* Target Goal */}
                <circle
                  cx={isInteractiveRover ? liveTarget.x : latestFrame?.target?.x ?? 75}
                  cy={isInteractiveRover ? liveTarget.y : latestFrame?.target?.y ?? 50}
                  r="7"
                  fill="url(#targetGlow)"
                />
                <circle
                  cx={isInteractiveRover ? liveTarget.x : latestFrame?.target?.x ?? 75}
                  cy={isInteractiveRover ? liveTarget.y : latestFrame?.target?.y ?? 50}
                  r="3.5"
                  fill="#0284c7"
                  stroke="#38bdf8"
                  strokeWidth="0.8"
                />

                {/* Obstacles */}
                {(isInteractiveRover ? liveObstacles : latestFrame?.obstacles ?? [
                  { x: 45, y: 35, radius: 8 },
                  { x: 30, y: 55, radius: 9 },
                ]).map((obs: any, idx: number) => (
                  <g key={idx}>
                    <circle cx={obs.x} cy={obs.y} r={obs.radius} fill="#450a0a" stroke="#dc2626" strokeWidth="0.8" opacity="0.85" />
                    <circle cx={obs.x} cy={obs.y} r={obs.radius + 2} fill="none" stroke="#ef4444" strokeWidth="0.3" strokeDasharray="1,1" />
                  </g>
                ))}

                {/* Trajectory Trail */}
                {(isInteractiveRover ? liveTrail : latestFrame?.trail ?? []).length > 1 && (
                  <polyline
                    fill="none"
                    stroke="#06b6d4"
                    strokeWidth="0.7"
                    strokeDasharray="1,1"
                    points={(isInteractiveRover ? liveTrail : latestFrame?.trail ?? [])
                      .map((p: any) => `${p[0]},${p[1]}`)
                      .join(' ')}
                  />
                )}

                {/* Robot Rover */}
                <g
                  transform={`translate(${isInteractiveRover ? liveRobot.x : latestFrame?.robot?.x ?? 25}, ${
                    isInteractiveRover ? liveRobot.y : latestFrame?.robot?.y ?? 25
                  })`}
                >
                  <circle r="4" fill="#0891b2" stroke="#22d3ee" strokeWidth="1" />
                  <line x1="0" y1="0" x2="4" y2="0" stroke="#f8fafc" strokeWidth="1.2" />
                </g>
              </svg>
            </div>
          </div>
        )}

        {/* =================================================================== */}
        {/* CARTRIDGE 3: Universal Document Reader / SOP Auditor               */}
        {/* =================================================================== */}
        {uiMode === 'document_reader' && (
          <div className="max-w-lg w-full bg-gray-900 border border-gray-800 rounded-2xl p-6 shadow-2xl space-y-4">
            <div className="flex items-center justify-between border-b border-gray-800 pb-3">
              <div className="flex items-center space-x-2">
                <FileText className="h-5 w-5 text-amber-400" />
                <h3 className="text-sm font-bold text-gray-100">
                  {latestFrame?.document_name || 'Procurement Policy & Invoice Audit'}
                </h3>
              </div>
              <span className="px-2 py-0.5 rounded text-[10px] font-bold uppercase bg-amber-950 border border-amber-800 text-amber-300">
                Chunk #{events.obs?.page_number || 1}
              </span>
            </div>

            <div className="p-4 rounded-xl bg-gray-950 border border-gray-800 text-xs font-mono text-gray-300 space-y-2 leading-relaxed">
              <p>
                {latestFrame?.raw_chunk ||
                  'Invoice #INV-2026-9042 for $12,500.00 from Vendor Acme Corp. Purchase order verification: MISSING. Department: Operations. Priority: URGENT.'}
              </p>
            </div>

            {/* Extracted NLP Audit Entities */}
            <div className="grid grid-cols-2 gap-2 text-xs font-mono">
              <div className="p-2.5 rounded-lg bg-gray-950/80 border border-gray-800">
                <span className="text-gray-400 block text-[10px]">MAX DOLLAR VALUE</span>
                <span className="text-emerald-400 font-bold text-sm">
                  ${(events.obs?.max_dollar_value ?? 12500).toLocaleString()}
                </span>
              </div>
              <div className="p-2.5 rounded-lg bg-gray-950/80 border border-gray-800">
                <span className="text-gray-400 block text-[10px]">RISK KEYWORDS DETECTED</span>
                <span className="text-rose-400 font-bold text-sm">
                  {events.obs?.risk_keyword_count ?? 3} flagged
                </span>
              </div>
            </div>

            {/* Agent Stamp */}
            <div className="p-3 rounded-xl bg-rose-950/40 border border-rose-800/60 flex items-center justify-between">
              <div>
                <span className="text-[10px] text-gray-400 uppercase font-semibold">Agent Audit Decision</span>
                <div className="text-sm font-extrabold text-rose-300">
                  ACTION: {events.action?.label || 'ESCALATE / HOLD'}
                </div>
              </div>
              <span className="text-xs font-mono font-bold text-rose-400 px-2.5 py-1 rounded bg-rose-950 border border-rose-800">
                AUDIT FLAGGED
              </span>
            </div>
          </div>
        )}

        {/* =================================================================== */}
        {/* CARTRIDGE 4: Tabular Record Inspector (SQL / CSV / Excel)          */}
        {/* =================================================================== */}
        {uiMode === 'tabular_inspector' && (
          <div className="max-w-md w-full bg-gray-900 border border-gray-800 rounded-2xl p-5 shadow-2xl space-y-4">
            <div className="flex items-center justify-between border-b border-gray-800 pb-3">
              <div className="flex items-center space-x-2">
                <Database className="h-4 w-4 text-emerald-400" />
                <h3 className="text-sm font-bold text-gray-100">Live Database Record</h3>
              </div>
              <span className="text-xs font-mono text-gray-400">Row #{events.state?.step_idx ?? 1}</span>
            </div>

            <div className="space-y-2 text-xs">
              {Object.entries(events.obs || {}).slice(0, 6).map(([key, val]) => (
                <div key={key} className="flex items-center justify-between py-1 border-b border-gray-800/50">
                  <span className="text-gray-400 font-mono">{key}</span>
                  <span className="text-gray-100 font-mono font-semibold">{String(val)}</span>
                </div>
              ))}
            </div>

            <div className="p-3 rounded-xl bg-emerald-950/40 border border-emerald-800/60 flex items-center justify-between">
              <div>
                <span className="text-[10px] text-gray-400 uppercase font-semibold">Policy Output</span>
                <div className="text-sm font-extrabold text-emerald-300">
                  {events.action?.label || 'CLEAR / PASS'}
                </div>
              </div>
              <span className="text-xs font-mono font-bold text-emerald-400 px-2 py-0.5 rounded bg-emerald-950 border border-emerald-800">
                MATCHED
              </span>
            </div>
          </div>
        )}

        {/* Fallback visualizer */}
        {uiMode !== 'board_grid' && uiMode !== 'robot_canvas' && uiMode !== 'document_reader' && uiMode !== 'tabular_inspector' && (
          <div className="p-8 text-center text-gray-400 text-xs">
            Simulation dashboard active. Use the coaching panel on the right to train and align policies.
          </div>
        )}
      </div>
    </div>
  )
}
