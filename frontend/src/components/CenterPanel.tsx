import React, { useState } from 'react'
import {
  Play,
  Pause,
  Star,
  Activity,
  BarChart2,
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  Mail,
  Paperclip,
  Flame,
  Crown,
  Inbox,
  ChevronLeft,
  ChevronRight,
  CheckCircle2,
  Tag,
} from 'lucide-react'
import { EpisodeLog } from '../types'

interface CenterPanelProps {
  adapter: string
  episodes: EpisodeLog[]
  selectedEpisodeId: number | null
  onSelectEpisode: (id: number) => void
  onRunBurst: (nEpisodes?: number) => void
  isRunningBurst: boolean
  isPaused: boolean
  onTogglePause: () => void
  onToggleLock: (id: number) => void
  lockedEpisodeIds: number[]
}

export const CenterPanel: React.FC<CenterPanelProps> = ({
  adapter,
  episodes,
  selectedEpisodeId,
  onSelectEpisode,
  onRunBurst,
  isRunningBurst,
  isPaused,
  onTogglePause,
  onToggleLock,
  lockedEpisodeIds,
}) => {
  const [hoveredStepIdx, setHoveredStepIdx] = useState<number | null>(null)
  const [ticketStepIdx, setTicketStepIdx] = useState<number>(0)

  const selectedEpisode =
    episodes.find((ep) => ep.episode_id === selectedEpisodeId) ||
    episodes[episodes.length - 1] ||
    null

  const isSelectedLocked = selectedEpisode
    ? lockedEpisodeIds.includes(selectedEpisode.episode_id) || Boolean(selectedEpisode.locked)
    : false

  const steps = selectedEpisode?.steps || []
  const nSteps = Math.max(steps.length, 1)

  // CSV Time-Series Visualization calculations
  const prices = steps.map((s) => Number(s.events?.obs?.price ?? s.obs_vector?.[0] ?? 100))
  const minPrice = prices.length ? Math.min(...prices) * 0.98 : 90
  const maxPrice = prices.length ? Math.max(...prices) * 1.02 : 110
  const priceRange = maxPrice - minPrice || 1

  const svgWidth = 800
  const svgHeight = 220
  const padX = 40
  const padY = 25
  const chartW = svgWidth - padX * 2
  const chartH = svgHeight - padY * 2

  const getX = (idx: number) => padX + (idx / Math.max(nSteps - 1, 1)) * chartW
  const getY = (price: number) => padY + chartH - ((price - minPrice) / priceRange) * chartH

  const priceLineD = steps.reduce((acc, step, idx) => {
    const p = Number(step.events?.obs?.price ?? step.obs_vector?.[0] ?? 100)
    const x = getX(idx)
    const y = getY(p)
    return idx === 0 ? `M ${x} ${y}` : `${acc} L ${x} ${y}`
  }, '')

  const vol90ThresholdY = getY(minPrice + priceRange * 0.75)

  const actionFlags = steps
    .map((s, idx) => ({
      idx,
      step: s,
      isFlagged: s.events?.action?.label === 'anomaly' || s.events?.action?.action_idx === 1,
      isTrueAnomaly: Boolean(s.events?.truth?.true_anomaly),
      price: Number(s.events?.obs?.price ?? s.obs_vector?.[0] ?? 100),
    }))
    .filter((a) => a.isFlagged)

  // Live reward sparkline data
  const rewardValues = episodes.map((e) => e.total_reward)
  const minReward = rewardValues.length ? Math.min(...rewardValues) : 0
  const maxReward = rewardValues.length ? Math.max(...rewardValues) : 1
  const rewardRange = maxReward - minReward || 1

  const sparkWidth = 400
  const sparkHeight = 60
  const sparkPad = 8
  const sparkW = sparkWidth - sparkPad * 2
  const sparkH = sparkHeight - sparkPad * 2

  const sparkLineD = episodes.reduce((acc, ep, idx) => {
    const x = sparkPad + (idx / Math.max(episodes.length - 1, 1)) * sparkW
    const y = sparkPad + sparkH - ((ep.total_reward - minReward) / rewardRange) * sparkH
    return idx === 0 ? `M ${x} ${y}` : `${acc} L ${x} ${y}`
  }, '')

  // Term Breakdown for selected episode
  const breakdownEntries = Object.entries(selectedEpisode?.breakdown || {})
  const totalPositive = breakdownEntries
    .filter(([_, v]) => v > 0)
    .reduce((sum, [_, v]) => sum + v, 0)
  const totalNegative = Math.abs(
    breakdownEntries
      .filter(([_, v]) => v < 0)
      .reduce((sum, [_, v]) => sum + v, 0),
  )
  const maxBreakdownMagnitude = Math.max(totalPositive, totalNegative, 1)

  // Selected Step for TextClassifyAdapter
  const currentStep = steps[ticketStepIdx] || steps[0] || null
  const currentTicket = currentStep?.info || {}

  const isTextDomain =
    adapter === 'TextClassifyAdapter' || Boolean(currentTicket.ticket_id || currentStep?.events?.obs?.sender_tier)

  return (
    <div className="flex-1 h-full flex flex-col bg-background overflow-hidden">
      {/* Playback Controls & Status Bar */}
      <div className="h-12 border-b border-panel-border bg-panel/70 px-4 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          {/* Run Burst */}
          <button
            onClick={() => onRunBurst(15)}
            disabled={isRunningBurst}
            className="flex items-center space-x-1.5 bg-blue-600 hover:bg-blue-500 text-white font-medium text-xs px-3.5 py-1.5 rounded-lg shadow-md shadow-blue-500/20 transition disabled:opacity-40"
          >
            {isRunningBurst ? (
              <>
                <div className="h-3 w-3 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                <span>Running Burst...</span>
              </>
            ) : (
              <>
                <Play className="h-3.5 w-3.5 fill-current" />
                <span>Run 15-Ep Burst</span>
              </>
            )}
          </button>

          {/* Pause / Resume */}
          <button
            onClick={onTogglePause}
            className={`flex items-center space-x-1 px-2.5 py-1.5 rounded-lg border text-xs font-medium transition ${
              isPaused
                ? 'bg-amber-950/60 border-amber-800/80 text-amber-400'
                : 'bg-gray-900 border-gray-800 text-gray-300 hover:bg-gray-800'
            }`}
          >
            <Pause className="h-3 w-3" />
            <span>{isPaused ? 'Resume' : 'Pause'}</span>
          </button>

          {/* Golden Episode Lock */}
          {selectedEpisode && (
            <button
              onClick={() => onToggleLock(selectedEpisode.episode_id)}
              className={`flex items-center space-x-1.5 px-2.5 py-1.5 rounded-lg border text-xs font-medium transition ${
                isSelectedLocked
                  ? 'bg-amber-950/80 border-amber-500 text-amber-300 shadow-sm shadow-amber-500/20'
                  : 'bg-gray-900 border-gray-800 text-gray-400 hover:text-amber-400 hover:border-gray-700'
              }`}
            >
              <Star
                className={`h-3.5 w-3.5 ${
                  isSelectedLocked ? 'fill-amber-400 text-amber-400' : 'text-gray-400'
                }`}
              />
              <span>{isSelectedLocked ? 'Episode Locked ★' : 'Lock Episode ★'}</span>
            </button>
          )}
        </div>

        {/* Buffer Stats Counter */}
        <div className="flex items-center space-x-3 text-xs">
          <div className="flex items-center space-x-1.5 text-gray-400">
            <Activity className="h-3.5 w-3.5 text-blue-400" />
            <span>
              Episodes in Buffer:{' '}
              <strong className="text-gray-200 font-mono">{episodes.length}</strong>
            </span>
          </div>
          {lockedEpisodeIds.length > 0 && (
            <span className="text-amber-400 font-mono font-medium bg-amber-950/60 px-2 py-0.5 rounded border border-amber-800/40">
              ★ {lockedEpisodeIds.length} Golden Locked
            </span>
          )}
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {/* DOMAIN VISUALIZER */}
        {isTextDomain ? (
          /* Text / Support Ticket Triage Episode Renderer */
          <div className="p-3.5 rounded-xl bg-panel border border-panel-border shadow-sm space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <Mail className="h-4 w-4 text-indigo-400" />
                <h3 className="text-xs font-bold text-gray-200 uppercase tracking-wider">
                  Text Triage Domain Visualizer — Support Ticket Batch ({steps.length} Tickets)
                </h3>
              </div>

              {selectedEpisode && (
                <div className="flex items-center space-x-2 text-xs font-mono">
                  <span className="text-gray-400">Batch:</span>
                  <span className="text-blue-400 font-bold">
                    Episode #{selectedEpisode.episode_id}
                  </span>
                  <span
                    className={`px-1.5 py-0.2 rounded font-bold ${
                      selectedEpisode.total_reward >= 0
                        ? 'text-emerald-400 bg-emerald-950/60'
                        : 'text-rose-400 bg-rose-950/60'
                    }`}
                  >
                    Score: {selectedEpisode.total_reward.toFixed(1)}
                  </span>
                </div>
              )}
            </div>

            {/* Ticket Card View */}
            {currentStep ? (
              <div className="bg-gray-950 rounded-xl p-4 border border-gray-800 space-y-3">
                {/* Ticket Header & Badges */}
                <div className="flex items-center justify-between pb-2 border-b border-gray-800">
                  <div className="flex items-center space-x-2">
                    <span className="font-mono text-xs font-bold text-blue-400 bg-blue-950/80 px-2 py-0.5 rounded border border-blue-800/50">
                      {currentTicket.ticket_id || `TICK-#${ticketStepIdx + 1}`}
                    </span>

                    {/* Sender Tier Badge */}
                    {currentTicket.sender_tier === 'vip' ? (
                      <span className="flex items-center space-x-1 text-xs bg-amber-950 text-amber-300 px-2 py-0.5 rounded border border-amber-700/60 font-semibold">
                        <Crown className="h-3 w-3 text-amber-400" />
                        <span>VIP Customer</span>
                      </span>
                    ) : currentTicket.sender_tier === 'internal' ? (
                      <span className="text-xs bg-blue-950 text-blue-300 px-2 py-0.5 rounded border border-blue-700/60 font-semibold">
                        Internal Service
                      </span>
                    ) : (
                      <span className="text-xs bg-gray-900 text-gray-400 px-2 py-0.5 rounded border border-gray-800">
                        Free Tier
                      </span>
                    )}

                    {currentTicket.has_attachment && (
                      <span className="flex items-center space-x-1 text-xs text-gray-400 bg-gray-900 px-2 py-0.5 rounded border border-gray-800">
                        <Paperclip className="h-3 w-3" />
                        <span>Attachment</span>
                      </span>
                    )}
                  </div>

                  {/* Step Selector Controls */}
                  <div className="flex items-center space-x-2 text-xs">
                    <span className="text-gray-400 font-mono">
                      Ticket {ticketStepIdx + 1} of {steps.length}
                    </span>
                    <button
                      onClick={() => setTicketStepIdx((prev) => Math.max(0, prev - 1))}
                      disabled={ticketStepIdx === 0}
                      className="p-1 rounded bg-gray-900 hover:bg-gray-800 disabled:opacity-30 border border-gray-800 text-gray-300"
                    >
                      <ChevronLeft className="h-3.5 w-3.5" />
                    </button>
                    <button
                      onClick={() =>
                        setTicketStepIdx((prev) => Math.min(steps.length - 1, prev + 1))
                      }
                      disabled={ticketStepIdx >= steps.length - 1}
                      className="p-1 rounded bg-gray-900 hover:bg-gray-800 disabled:opacity-30 border border-gray-800 text-gray-300"
                    >
                      <ChevronRight className="h-3.5 w-3.5" />
                    </button>
                  </div>
                </div>

                {/* Email Subject & Text */}
                <div className="space-y-1.5">
                  <h4 className="text-sm font-semibold text-gray-100">
                    {currentTicket.subject || currentStep.events?.obs?.subject || 'Support Ticket'}
                  </h4>
                  <p className="text-xs text-gray-300 bg-gray-900/60 p-3 rounded-lg border border-gray-800/80 leading-relaxed font-sans">
                    {currentTicket.body_preview ||
                      'Customer message requesting immediate assistance with production infrastructure...'}
                  </p>
                </div>

                {/* Extracted NLP Signals Strip */}
                <div className="grid grid-cols-4 gap-2 pt-1 text-xs">
                  <div className="bg-gray-900/80 p-2 rounded border border-gray-800 flex items-center justify-between">
                    <span className="text-gray-400 flex items-center space-x-1">
                      <Flame className="h-3 w-3 text-rose-400" />
                      <span>Urgency Keywords:</span>
                    </span>
                    <strong className="font-mono text-gray-200">
                      {currentTicket.urgency_keyword_count ?? currentStep.events?.obs?.urgency_keyword_count ?? 0}
                    </strong>
                  </div>

                  <div className="bg-gray-900/80 p-2 rounded border border-gray-800 flex items-center justify-between">
                    <span className="text-gray-400">Angry Tone:</span>
                    <span
                      className={`font-mono font-bold ${
                        (currentTicket.angry_tone_score ?? 0) > 0.5
                          ? 'text-rose-400'
                          : 'text-emerald-400'
                      }`}
                    >
                      {Number(currentTicket.angry_tone_score ?? currentStep.events?.obs?.angry_tone_score ?? 0).toFixed(2)}
                    </span>
                  </div>

                  <div className="bg-gray-900/80 p-2 rounded border border-gray-800 flex items-center justify-between">
                    <span className="text-gray-400">Queue Backlog:</span>
                    <span className="font-mono text-gray-200">
                      {currentStep.events?.state?.queue_load ?? 0}
                    </span>
                  </div>

                  <div className="bg-gray-900/80 p-2 rounded border border-gray-800 flex items-center justify-between">
                    <span className="text-gray-400">Ground Truth:</span>
                    <span className="font-mono font-semibold text-gray-300 uppercase text-[11px]">
                      {currentTicket.true_label ?? currentStep.events?.truth?.label ?? 'normal'}
                    </span>
                  </div>
                </div>

                {/* Agent Routing Decision */}
                <div className="p-2.5 rounded-lg bg-gray-900 border border-gray-800 flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <span className="text-xs text-gray-400 font-medium">Agent Routing:</span>
                    <span
                      className={`px-3 py-1 rounded-md text-xs font-mono font-bold uppercase ${
                        currentStep.events?.action?.label === 'urgent'
                          ? 'bg-rose-950 text-rose-300 border border-rose-700/60'
                          : currentStep.events?.action?.label === 'escalate'
                          ? 'bg-purple-950 text-purple-300 border border-purple-700/60'
                          : currentStep.events?.action?.label === 'spam'
                          ? 'bg-amber-950 text-amber-300 border border-amber-700/60'
                          : 'bg-slate-800 text-slate-200 border border-slate-700'
                      }`}
                    >
                      {currentStep.events?.action?.label}
                    </span>

                    {currentTicket.is_correct !== undefined && (
                      <span className="flex items-center space-x-1 text-xs">
                        {currentTicket.is_correct ? (
                          <span className="text-emerald-400 flex items-center space-x-1">
                            <CheckCircle2 className="h-3.5 w-3.5" />
                            <span>Correct Routing</span>
                          </span>
                        ) : (
                          <span className="text-rose-400 flex items-center space-x-1">
                            <AlertTriangle className="h-3.5 w-3.5" />
                            <span>Misclassified</span>
                          </span>
                        )}
                      </span>
                    )}
                  </div>

                  <div className="text-xs font-mono">
                    <span className="text-gray-400 mr-1.5">Step Reward:</span>
                    <span
                      className={`font-bold ${
                        (selectedEpisode?.step_rewards?.[ticketStepIdx] ?? 0) >= 0
                          ? 'text-emerald-400'
                          : 'text-rose-400'
                      }`}
                    >
                      {(selectedEpisode?.step_rewards?.[ticketStepIdx] ?? 0) >= 0 ? '+' : ''}
                      {(selectedEpisode?.step_rewards?.[ticketStepIdx] ?? 0).toFixed(1)}
                    </span>
                  </div>
                </div>
              </div>
            ) : (
              <div className="h-44 flex flex-col items-center justify-center bg-gray-950 rounded-lg border border-gray-800 text-gray-500 text-xs space-y-2">
                <Inbox className="h-6 w-6 text-gray-600 animate-pulse" />
                <span>No ticket batch data yet. Click [Run 15-Ep Burst] to triage tickets.</span>
              </div>
            )}
          </div>
        ) : (
          /* CSV Stream Financial Time-Series Visualizer */
          <div className="p-3.5 rounded-xl bg-panel border border-panel-border shadow-sm">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center space-x-2">
                <TrendingUp className="h-4 w-4 text-cyan-400" />
                <h3 className="text-xs font-bold text-gray-200 uppercase tracking-wider">
                  Domain Visualizer — CSV Financial Stream (100 Steps)
                </h3>
              </div>
              {selectedEpisode && (
                <div className="flex items-center space-x-2 text-xs font-mono">
                  <span className="text-gray-400">Inspecting:</span>
                  <span className="text-blue-400 font-bold">
                    Episode #{selectedEpisode.episode_id}
                  </span>
                  <span
                    className={`px-1.5 py-0.2 rounded font-bold ${
                      selectedEpisode.total_reward >= 0
                        ? 'text-emerald-400 bg-emerald-950/60'
                        : 'text-rose-400 bg-rose-950/60'
                    }`}
                  >
                    Score: {selectedEpisode.total_reward.toFixed(1)}
                  </span>
                </div>
              )}
            </div>

            {/* Interactive SVG Chart */}
            {steps.length > 0 ? (
              <div className="relative bg-gray-950/90 rounded-lg p-2 border border-gray-800/80">
                <svg
                  viewBox={`0 0 ${svgWidth} ${svgHeight}`}
                  className="w-full h-48 select-none"
                >
                  {[0.25, 0.5, 0.75].map((ratio) => {
                    const y = padY + chartH * ratio
                    return (
                      <line
                        key={ratio}
                        x1={padX}
                        y1={y}
                        x2={svgWidth - padX}
                        y2={y}
                        stroke="#1F2937"
                        strokeDasharray="4 4"
                        strokeWidth="1"
                      />
                    )
                  })}

                  <line
                    x1={padX}
                    y1={vol90ThresholdY}
                    x2={svgWidth - padX}
                    y2={vol90ThresholdY}
                    stroke="#F59E0B"
                    strokeDasharray="3 3"
                    strokeWidth="1.2"
                    opacity="0.6"
                  />
                  <text
                    x={svgWidth - padX - 10}
                    y={vol90ThresholdY - 4}
                    fill="#F59E0B"
                    fontSize="9"
                    textAnchor="end"
                    opacity="0.8"
                    fontFamily="monospace"
                  >
                    Vol 90th %ile Threshold
                  </text>

                  <path
                    d={priceLineD}
                    fill="none"
                    stroke="#38BDF8"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />

                  {actionFlags.map((flag) => {
                    const x = getX(flag.idx)
                    const y = getY(flag.price)
                    const isTP = flag.isTrueAnomaly
                    return (
                      <g key={flag.idx} className="cursor-pointer">
                        <circle
                          cx={x}
                          cy={y}
                          r="6"
                          fill={isTP ? '#10B981' : '#EF4444'}
                          fillOpacity="0.8"
                          stroke="#FFFFFF"
                          strokeWidth="1.5"
                        />
                        <line
                          x1={x}
                          y1={y}
                          x2={x}
                          y2={padY + chartH}
                          stroke={isTP ? '#10B981' : '#EF4444'}
                          strokeDasharray="2 2"
                          strokeWidth="1"
                          opacity="0.5"
                        />
                      </g>
                    )
                  })}

                  {steps.map((_, idx) => {
                    const x = getX(idx)
                    return (
                      <rect
                        key={idx}
                        x={x - chartW / (nSteps * 2)}
                        y={padY}
                        width={chartW / nSteps}
                        height={chartH}
                        fill="transparent"
                        className="cursor-crosshair"
                        onMouseEnter={() => setHoveredStepIdx(idx)}
                        onMouseLeave={() => setHoveredStepIdx(null)}
                      />
                    )
                  })}

                  {hoveredStepIdx !== null && steps[hoveredStepIdx] && (
                    <g>
                      <line
                        x1={getX(hoveredStepIdx)}
                        y1={padY}
                        x2={getX(hoveredStepIdx)}
                        y2={padY + chartH}
                        stroke="#94A3B8"
                        strokeWidth="1.5"
                      />
                      <circle
                        cx={getX(hoveredStepIdx)}
                        cy={getY(
                          Number(
                            steps[hoveredStepIdx].events?.obs?.price ??
                              steps[hoveredStepIdx].obs_vector?.[0] ??
                              100,
                          ),
                        )}
                        r="4"
                        fill="#FFFFFF"
                      />
                    </g>
                  )}
                </svg>

                <div className="flex items-center justify-between px-2 pt-2 text-[11px] border-t border-gray-800 text-gray-400">
                  <div className="flex items-center space-x-4">
                    <div className="flex items-center space-x-1.5">
                      <span className="h-2 w-3 bg-[#38BDF8] rounded-sm" />
                      <span>Price</span>
                    </div>
                    <div className="flex items-center space-x-1.5">
                      <span className="h-2 w-3 bg-amber-500/70 border border-amber-400 border-dashed rounded-sm" />
                      <span>Vol Threshold</span>
                    </div>
                    <div className="flex items-center space-x-1.5">
                      <span className="h-2.5 w-2.5 rounded-full bg-emerald-500" />
                      <span>True Positive Alert</span>
                    </div>
                    <div className="flex items-center space-x-1.5">
                      <span className="h-2.5 w-2.5 rounded-full bg-rose-500" />
                      <span>False Alarm</span>
                    </div>
                  </div>

                  {hoveredStepIdx !== null && steps[hoveredStepIdx] ? (
                    <div className="font-mono text-gray-300 space-x-3 bg-gray-900 px-2 py-0.5 rounded border border-gray-700">
                      <span>Step: {hoveredStepIdx}</span>
                      <span>
                        Price:{' '}
                        {Number(
                          steps[hoveredStepIdx].events?.obs?.price ??
                            steps[hoveredStepIdx].obs_vector?.[0] ??
                            0,
                        ).toFixed(2)}
                      </span>
                      <span>
                        Action:{' '}
                        <strong
                          className={
                            steps[hoveredStepIdx].events?.action?.label === 'anomaly'
                              ? 'text-rose-400'
                              : 'text-gray-400'
                          }
                        >
                          {steps[hoveredStepIdx].events?.action?.label}
                        </strong>
                      </span>
                      <span>
                        Reward:{' '}
                        {(selectedEpisode?.step_rewards?.[hoveredStepIdx] ?? 0) >= 0 ? '+' : ''}
                        {(selectedEpisode?.step_rewards?.[hoveredStepIdx] ?? 0).toFixed(2)}
                      </span>
                    </div>
                  ) : (
                    <span className="text-gray-500 italic">Hover chart for step inspector</span>
                  )}
                </div>
              </div>
            ) : (
              <div className="h-44 flex flex-col items-center justify-center bg-gray-950 rounded-lg border border-gray-800 text-gray-500 text-xs space-y-2">
                <Activity className="h-6 w-6 text-gray-600 animate-pulse" />
                <span>No episode data available. Click [Run 15-Ep Burst] to train agent.</span>
              </div>
            )}
          </div>
        )}

        {/* Clickable Episode History Strip */}
        <div className="p-3 rounded-xl bg-panel border border-panel-border">
          <span className="text-[10px] text-gray-500 uppercase font-bold tracking-wider mb-2 block">
            Episode History Strip (Click to inspect)
          </span>
          <div className="flex space-x-1.5 overflow-x-auto pb-1">
            {episodes.length > 0 ? (
              episodes.map((ep) => {
                const isSelected = ep.episode_id === selectedEpisode?.episode_id
                const isLocked =
                  lockedEpisodeIds.includes(ep.episode_id) || Boolean(ep.locked)
                const isGood = ep.total_reward >= 0
                return (
                  <button
                    key={ep.episode_id}
                    onClick={() => {
                      onSelectEpisode(ep.episode_id)
                      setTicketStepIdx(0)
                    }}
                    className={`flex-shrink-0 flex items-center space-x-1.5 px-2.5 py-1 rounded-md text-xs font-mono transition border ${
                      isSelected
                        ? 'bg-blue-950 border-blue-500 text-blue-200 ring-1 ring-blue-500'
                        : 'bg-gray-900 border-gray-800 text-gray-400 hover:border-gray-700 hover:text-gray-200'
                    }`}
                  >
                    {isLocked && <Star className="h-3 w-3 fill-amber-400 text-amber-400" />}
                    <span>#{ep.episode_id}</span>
                    <span
                      className={`text-[10px] font-bold ${
                        isGood ? 'text-emerald-400' : 'text-rose-400'
                      }`}
                    >
                      {ep.total_reward >= 0 ? '+' : ''}
                      {ep.total_reward.toFixed(1)}
                    </span>
                  </button>
                )
              })
            ) : (
              <span className="text-xs text-gray-500">History empty</span>
            )}
          </div>
        </div>

        {/* Live Reward Curve & Per-Term Breakdown Grid */}
        <div className="grid grid-cols-2 gap-4">
          {/* Live Reward Curve */}
          <div className="p-3.5 rounded-xl bg-panel border border-panel-border shadow-sm flex flex-col justify-between">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center space-x-2">
                <TrendingUp className="h-4 w-4 text-emerald-400" />
                <h3 className="text-xs font-bold text-gray-200 uppercase tracking-wider">
                  Live Reward Curve
                </h3>
              </div>
              <span className="text-[10px] text-gray-500 font-mono">
                Across {episodes.length} Episodes
              </span>
            </div>

            {episodes.length > 1 ? (
              <div className="bg-gray-950/80 rounded-lg p-2 border border-gray-800">
                <svg
                  viewBox={`0 0 ${sparkWidth} ${sparkHeight}`}
                  className="w-full h-16 select-none"
                >
                  <path
                    d={sparkLineD}
                    fill="none"
                    stroke="#10B981"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
                <div className="flex justify-between text-[10px] text-gray-500 font-mono mt-1">
                  <span>Min: {minReward.toFixed(1)}</span>
                  <span>
                    Avg:{' '}
                    {(
                      rewardValues.reduce((a, b) => a + b, 0) / Math.max(rewardValues.length, 1)
                    ).toFixed(1)}
                  </span>
                  <span>Max: {maxReward.toFixed(1)}</span>
                </div>
              </div>
            ) : (
              <div className="h-20 flex items-center justify-center bg-gray-950 rounded-lg text-gray-500 text-xs">
                Awaiting more episodes for reward trajectory...
              </div>
            )}
          </div>

          {/* Per-Term Reward Breakdown */}
          <div className="p-3.5 rounded-xl bg-panel border border-panel-border shadow-sm">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center space-x-2">
                <BarChart2 className="h-4 w-4 text-purple-400" />
                <h3 className="text-xs font-bold text-gray-200 uppercase tracking-wider">
                  Term Breakdown (Ep #{selectedEpisode?.episode_id ?? '-'})
                </h3>
              </div>
            </div>

            {breakdownEntries.length > 0 ? (
              <div className="space-y-2 text-xs">
                {breakdownEntries.map(([termName, val]) => {
                  const isPositive = val >= 0
                  const pct = Math.min((Math.abs(val) / maxBreakdownMagnitude) * 100, 100)
                  return (
                    <div key={termName} className="space-y-0.5">
                      <div className="flex justify-between text-[11px] font-mono">
                        <span className="text-gray-300">{termName}</span>
                        <span
                          className={`font-bold ${
                            isPositive ? 'text-emerald-400' : 'text-rose-400'
                          }`}
                        >
                          {isPositive ? `+${val.toFixed(1)}` : val.toFixed(1)}
                        </span>
                      </div>
                      <div className="h-2 w-full bg-gray-900 rounded-full overflow-hidden flex">
                        <div
                          className={`h-full rounded-full ${
                            isPositive ? 'bg-emerald-500' : 'bg-rose-500'
                          }`}
                          style={{ width: `${Math.max(pct, 4)}%` }}
                        />
                      </div>
                    </div>
                  )
                })}
              </div>
            ) : (
              <div className="h-20 flex items-center justify-center bg-gray-950 rounded-lg text-gray-500 text-xs">
                Select an episode to view term contributions
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
