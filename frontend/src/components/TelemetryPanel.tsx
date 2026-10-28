import React from 'react'
import {
  Target,
  TrendingUp,
  RotateCcw,
  CheckCircle2,
  AlertCircle,
  Brain,
  Layers,
  Sparkles,
  Zap,
} from 'lucide-react'
import { RoundSnapshot } from '../types'

interface TelemetryPanelProps {
  targetMetricName?: string
  targetThreshold?: number
  currentMetric?: number
  currentRound: number
  maxRounds: number
  roundHistory: RoundSnapshot[]
  driftStatus: string
  onRollback: (roundNum: number) => Promise<void>
  isRollingBack?: boolean
}

export const TelemetryPanel: React.FC<TelemetryPanelProps> = ({
  targetMetricName = 'Target Success Rate',
  targetThreshold = 85,
  currentMetric = 0,
  currentRound = 1,
  maxRounds = 4,
  roundHistory = [],
  driftStatus = 'healthy',
  onRollback,
  isRollingBack = false,
}) => {
  const isTargetAchieved = currentMetric >= targetThreshold
  const progressPercent = Math.min(100, Math.max(0, (currentMetric / (targetThreshold || 100)) * 100))

  return (
    <aside className="w-80 flex flex-col h-full bg-gray-900 border-l border-gray-800 text-gray-200 select-none overflow-y-auto">
      {/* 1. Target Expectations & Autopilot Progress Card */}
      <div className="p-4 border-b border-gray-800 bg-gray-950/80 space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-xs font-bold uppercase tracking-wider text-gray-400 flex items-center space-x-1.5">
            <Target className="h-4 w-4 text-emerald-400" />
            <span>Target Expectations</span>
          </span>
          <span
            className={`text-[10px] font-bold px-2 py-0.5 rounded-full border ${
              isTargetAchieved
                ? 'bg-emerald-950/80 text-emerald-400 border-emerald-800/60'
                : 'bg-indigo-950/80 text-indigo-400 border-indigo-800/60'
            }`}
          >
            {isTargetAchieved ? 'Target Achieved' : `Round ${currentRound} of ${maxRounds}`}
          </span>
        </div>

        {/* Big Metric Display */}
        <div className="p-3 rounded-xl bg-gray-900/90 border border-gray-800 space-y-2">
          <div className="flex items-baseline justify-between">
            <span className="text-xs text-gray-400 font-medium truncate">{targetMetricName}</span>
            <div className="flex items-baseline space-x-1">
              <span className={`text-2xl font-black font-mono ${isTargetAchieved ? 'text-emerald-400' : 'text-indigo-300'}`}>
                {currentMetric.toFixed(1)}%
              </span>
              <span className="text-xs text-gray-500 font-mono">/ {targetThreshold}%</span>
            </div>
          </div>

          {/* Animated Progress Bar */}
          <div className="w-full bg-gray-800 rounded-full h-2.5 overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-700 ease-out ${
                isTargetAchieved
                  ? 'bg-gradient-to-r from-emerald-500 to-teal-400'
                  : 'bg-gradient-to-r from-indigo-500 to-cyan-400'
              }`}
              style={{ width: `${progressPercent}%` }}
            />
          </div>

          <div className="flex items-center justify-between text-[11px] text-gray-400 pt-1">
            <span className="flex items-center space-x-1">
              <span className={`h-2 w-2 rounded-full ${driftStatus === 'healthy' ? 'bg-emerald-400' : 'bg-amber-400'}`} />
              <span className="capitalize">{driftStatus} Policy Drift</span>
            </span>
            <span className="font-mono text-gray-500">{isTargetAchieved ? '100% Met' : `${progressPercent.toFixed(0)}% of Goal`}</span>
          </div>
        </div>
      </div>

      {/* 2. Autonomous Closed-Loop Self-Tuning History */}
      <div className="flex-1 p-4 space-y-3 overflow-y-auto">
        <div className="flex items-center justify-between">
          <span className="text-xs font-bold uppercase tracking-wider text-gray-400 flex items-center space-x-1.5">
            <Brain className="h-4 w-4 text-purple-400" />
            <span>Auto-Tuner Evolutions</span>
          </span>
          <span className="text-[10px] text-gray-500 font-mono">{roundHistory.length} Rounds</span>
        </div>

        {roundHistory.length === 0 ? (
          <div className="p-4 rounded-xl border border-dashed border-gray-800 text-center text-xs text-gray-500">
            Waiting for Round 1 to complete telemetry evaluation...
          </div>
        ) : (
          <div className="space-y-3">
            {roundHistory.map((snap) => {
              const isMet = snap.target_achieved || snap.metric >= targetThreshold
              return (
                <div
                  key={snap.round_num}
                  className="p-3 rounded-xl bg-gray-950/70 border border-gray-800 hover:border-gray-700 transition space-y-2 text-xs"
                >
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-gray-200 flex items-center space-x-1.5">
                      <span className="px-1.5 py-0.5 rounded bg-purple-950/60 border border-purple-800/40 text-purple-300 font-mono text-[10px]">
                        Round {snap.round_num}
                      </span>
                      <span className={`font-mono font-bold ${isMet ? 'text-emerald-400' : 'text-indigo-400'}`}>
                        {snap.metric.toFixed(1)}%
                      </span>
                    </span>

                    {/* Rollback Button (if not current or previous round) */}
                    {roundHistory.length > 1 && (
                      <button
                        onClick={() => onRollback(snap.round_num)}
                        disabled={isRollingBack}
                        className="flex items-center space-x-1 text-[10px] text-gray-400 hover:text-amber-400 font-medium px-1.5 py-0.5 rounded hover:bg-gray-800 transition"
                        title={`Rollback model & environment code to Round ${snap.round_num}`}
                      >
                        <RotateCcw className="h-3 w-3" />
                        <span>Rollback</span>
                      </button>
                    )}
                  </div>

                  {/* Diagnosis */}
                  {snap.diagnosis && (
                    <div className="p-2 rounded-lg bg-gray-900 border border-gray-800/80 text-gray-300 text-[11px] leading-relaxed">
                      <span className="font-semibold text-purple-400 block mb-0.5">🧠 Auto-Tuner Diagnosis:</span>
                      {snap.diagnosis}
                    </div>
                  )}

                  {/* Changes Summary */}
                  {snap.changes_summary && (
                    <div className="text-[11px] text-gray-400">
                      <span className="font-semibold text-gray-300">Reward Updates: </span>
                      {snap.changes_summary}
                    </div>
                  )}

                  {/* Term Rewards Breakdown if available */}
                  {snap.term_rewards && Object.keys(snap.term_rewards).length > 0 && (
                    <div className="pt-1 border-t border-gray-800/60 flex flex-wrap gap-1">
                      {Object.entries(snap.term_rewards).map(([term, val]) => (
                        <span
                          key={term}
                          className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-gray-900 text-gray-400 border border-gray-800"
                        >
                          {term}: {val > 0 ? `+${val.toFixed(2)}` : val.toFixed(2)}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        )}
      </div>

      {/* 3. Bottom Sparklines / Summary */}
      {roundHistory.length > 0 && (
        <div className="p-3 border-t border-gray-800 bg-gray-950/90 text-xs">
          <div className="flex items-center justify-between text-gray-400 text-[11px] mb-2 font-semibold">
            <span className="flex items-center space-x-1">
              <TrendingUp className="h-3.5 w-3.5 text-cyan-400" />
              <span>Multi-Round Trajectory</span>
            </span>
            <span className="font-mono text-cyan-400">
              {roundHistory[0].metric.toFixed(0)}% → {roundHistory[roundHistory.length - 1].metric.toFixed(0)}%
            </span>
          </div>

          <div className="flex items-end space-x-2 h-14 bg-gray-900 rounded-lg p-2 border border-gray-800">
            {roundHistory.map((snap) => {
              const heightPct = Math.max(10, Math.min(100, snap.metric))
              const isMet = snap.metric >= targetThreshold
              return (
                <div key={snap.round_num} className="flex-1 flex flex-col items-center h-full justify-end group relative">
                  <div
                    className={`w-full rounded-t transition-all ${
                      isMet ? 'bg-emerald-500' : 'bg-indigo-500 group-hover:bg-indigo-400'
                    }`}
                    style={{ height: `${heightPct}%` }}
                  />
                  <span className="text-[9px] font-mono text-gray-500 mt-1">R{snap.round_num}</span>
                  {/* Tooltip */}
                  <div className="absolute -top-7 hidden group-hover:block bg-gray-950 text-white text-[10px] font-mono px-1.5 py-0.5 rounded border border-gray-700 whitespace-nowrap shadow-lg">
                    {snap.metric.toFixed(1)}%
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}
    </aside>
  )
}
