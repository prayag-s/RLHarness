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
import { RoundSnapshot, Scorecard } from '../types'

interface TelemetryPanelProps {
  targetMetricName?: string
  targetThreshold?: number
  currentMetric?: number
  currentRound: number
  maxRounds: number
  roundHistory: RoundSnapshot[]
  scorecard?: Scorecard | null
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
  scorecard = null,
  driftStatus = 'healthy',
  onRollback,
  isRollingBack = false,
}) => {
  const isTargetAchieved = currentMetric >= targetThreshold
  const progressPercent = Math.min(100, Math.max(0, (currentMetric / (targetThreshold || 100)) * 100))
  const bestBaseline = scorecard?.baselines?.length
    ? scorecard.baselines.reduce(
        (best, b) => (b.value_pct > best.value_pct ? b : best),
        scorecard.baselines[0],
      )
    : null
  const ciLow = scorecard?.policy?.ci95_pct?.[0]
  const statisticallyTied =
    !!scorecard &&
    !!bestBaseline &&
    !scorecard.beats_all_baselines &&
    typeof ciLow === 'number' &&
    ciLow <= bestBaseline.value_pct

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
              <span className={`h-2 w-2 rounded-full ${
                driftStatus === 'healthy' ? 'bg-emerald-400' : driftStatus === 'unmonitored' ? 'bg-slate-600' : 'bg-amber-400'
              }`} />
              <span className="capitalize">{driftStatus === 'unmonitored' ? 'Drift not monitored' : `${driftStatus} Policy Drift`}</span>
            </span>
            <span className="font-mono text-gray-500">{isTargetAchieved ? '100% Met' : `${progressPercent.toFixed(0)}% of Goal`}</span>
          </div>
        </div>
      </div>

      {/* 1b. Held-out scorecard: the only number that describes the shipped policy */}
      {scorecard && (
        <div className="p-4 border-b border-gray-800 bg-gray-950/60 space-y-2.5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold uppercase tracking-wider text-gray-400 flex items-center space-x-1.5">
              <CheckCircle2 className="h-4 w-4 text-cyan-400" />
              <span>Held-Out Scorecard</span>
            </span>
            <span
              className={`text-[10px] font-bold px-2 py-0.5 rounded-full border ${
                scorecard.held_out
                  ? 'bg-cyan-950/80 text-cyan-300 border-cyan-800/60'
                  : 'bg-amber-950/80 text-amber-300 border-amber-800/60'
              }`}
            >
              {scorecard.held_out ? `held out · ${scorecard.eval_split}` : 'in-sample'}
            </span>
          </div>

          <div className="flex items-baseline justify-between">
            <span className="text-[11px] text-gray-500 font-mono truncate pr-2">
              {scorecard.metric_name}
            </span>
            <div className="flex items-baseline space-x-1">
              <span className="text-xl font-black font-mono text-cyan-300">
                {scorecard.policy.value_pct.toFixed(1)}%
              </span>
              {scorecard.policy.ci95_pct && (
                <span className="text-[10px] text-gray-500 font-mono">
                  [{scorecard.policy.ci95_pct[0].toFixed(1)}–{scorecard.policy.ci95_pct[1].toFixed(1)}]
                </span>
              )}
            </div>
          </div>

          <div className="space-y-1">
            {scorecard.baselines.map((b) => (
              <div key={b.name} className="flex items-center justify-between text-[11px]">
                <span className="text-gray-400 truncate pr-2">
                  {b.name.replace(/_/g, ' ')}
                  {b.detail ? <span className="text-gray-600"> · {b.detail}</span> : null}
                </span>
                <span className="font-mono text-gray-500">{b.value_pct.toFixed(1)}%</span>
              </div>
            ))}
          </div>

          <div
            className={`text-[11px] rounded-lg px-2 py-1.5 border ${
              scorecard.beats_all_baselines
                ? 'bg-emerald-950/40 border-emerald-800/40 text-emerald-300'
                : 'bg-amber-950/40 border-amber-800/40 text-amber-300'
            }`}
          >
            {scorecard.beats_all_baselines
              ? `Beats every baseline by +${scorecard.uplift_vs_best_baseline_pct.toFixed(1)} pts`
              : statisticallyTied
                ? `Statistically tied with ${bestBaseline!.name.replace(/_/g, ' ')} (${bestBaseline!.value_pct.toFixed(1)}%): the policy CI includes it`
                : `${Math.abs(scorecard.uplift_vs_best_baseline_pct).toFixed(1)} pts below the best baseline (${bestBaseline?.name.replace(/_/g, ' ') ?? 'baseline'})`}
          </div>

          {scorecard.deployment?.action_agreement_pct !== undefined && (
            <div className="flex items-center justify-between text-[10px] text-gray-500 font-mono">
              <span>ONNX behaviour</span>
              <span>
                {scorecard.deployment.action_agreement_pct.toFixed(1)}% action agreement
              </span>
            </div>
          )}

          <div className="flex items-center justify-between text-[10px] text-gray-600 font-mono">
            <span>
              train {scorecard.split_counts.train ?? 0} · val {scorecard.split_counts.val ?? 0} · test{' '}
              {scorecard.split_counts.test ?? 0}
            </span>
            <span>{scorecard.policy.episodes} eps</span>
          </div>

          {scorecard.eval_corpus && (
            <div className="flex items-center justify-between text-[10px] font-mono">
              <span className="text-gray-500">🛡️ Golden corpus</span>
              <span
                className={
                  scorecard.eval_corpus.regressed_case_ids.length > 0
                    ? 'text-amber-300'
                    : 'text-emerald-300'
                }
              >
                {scorecard.eval_corpus.passed_cases}/{scorecard.eval_corpus.scored_cases} verified
                {scorecard.eval_corpus.regressed_case_ids.length > 0
                  ? ` · ${scorecard.eval_corpus.regressed_case_ids.length} regressed`
                  : ''}
              </span>
            </div>
          )}

          {scorecard.selection?.source === 'best_val_checkpoint' && (
            <div className="flex items-center justify-between text-[10px] text-gray-500 font-mono">
              <span>Selected checkpoint</span>
              <span>round {scorecard.selection.selected_round} (best val)</span>
            </div>
          )}

          {scorecard.crossfire && (
            <div
              className={`text-[10px] rounded-lg px-2 py-1.5 border font-mono ${
                scorecard.crossfire.verdict === 'win'
                  ? 'bg-emerald-950/40 border-emerald-800/40 text-emerald-300'
                  : scorecard.crossfire.verdict === 'loss'
                    ? 'bg-rose-950/40 border-rose-800/40 text-rose-300'
                    : 'bg-gray-900/60 border-gray-800 text-gray-400'
              }`}
              title={(scorecard.crossfire.reasons || []).join(' ')}
            >
              Crossfire: {scorecard.crossfire.verdict}
              {scorecard.crossfire.reasons?.length
                ? ` · ${scorecard.crossfire.reasons[0]}`
                : ''}
            </div>
          )}

          {scorecard.safety && (
            <div
              className={`text-[10px] rounded-lg px-2 py-1.5 border font-mono ${
                scorecard.safety.violations > 0
                  ? 'bg-rose-950/40 border-rose-800/40 text-rose-300'
                  : 'bg-emerald-950/40 border-emerald-800/40 text-emerald-300'
              }`}
              title={
                scorecard.safety.examples?.length
                  ? JSON.stringify(scorecard.safety.examples.slice(0, 3), null, 1)
                  : 'Hard constraints are enforced before scoring, play and export.'
              }
            >
              🛡️ Constraints: {scorecard.safety.violations} raw violation
              {scorecard.safety.violations === 1 ? '' : 's'} / {scorecard.safety.checked} rows
              {scorecard.safety.runtime_enforced ? ' · runtime-enforced' : ''}
            </div>
          )}

          {scorecard.economics && (
            <div className="space-y-0.5 border-t border-gray-800 pt-2">
              <div className="flex items-center justify-between text-[10px] text-gray-500 font-mono">
                <span>⚡ p99 latency</span>
                <span>
                  {scorecard.economics.latency.status === 'measured'
                    ? `${scorecard.economics.latency.end_to_end_p99_ms} ms`
                    : 'not measured'}
                </span>
              </div>
              {Object.entries(scorecard.economics.payback).map(([tier, data]) => (
                <div
                  key={tier}
                  className="flex items-center justify-between text-[10px] text-gray-600 font-mono"
                  title={`${data.label} · token-only payback ${data.basis_token_only.decisions_to_payback} decisions`}
                >
                  <span className="truncate pr-2">{tier.replace(/_/g, ' ')}</span>
                  <span>
                    {data.basis_with_human_review.decisions_to_payback.toLocaleString()} decisions
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

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
            {roundHistory.map((snap, idx) => {
              const isMet = snap.target_achieved || snap.metric >= targetThreshold
              return (
                <div
                  key={`round_${snap.round_num}_${idx}`}
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
                      {Object.entries(snap.term_rewards).map(([term, val], termIdx) => (
                        <span
                          key={`term_${snap.round_num}_${term}_${termIdx}`}
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
            {roundHistory.map((snap, sparkIdx) => {
              const heightPct = Math.max(10, Math.min(100, snap.metric))
              const isMet = snap.metric >= targetThreshold
              return (
                <div key={`sparkline_${snap.round_num}_${sparkIdx}`} className="flex-1 flex flex-col items-center h-full justify-end group relative">
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
