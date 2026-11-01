import React from 'react'
import {
  Activity,
  AlertTriangle,
  BarChart3,
  CheckCircle2,
  Database,
  Download,
  Gauge,
  ShieldCheck,
  Target,
} from 'lucide-react'
import { IngestedDatasetSummary, RoundSnapshot, Scorecard } from '../types'

interface ScoreboardRailProps {
  targetMetricName: string
  targetThreshold: number
  currentMetric: number
  currentRound: number
  maxRounds: number
  roundHistory: RoundSnapshot[]
  scorecard: Scorecard | null
  driftStatus: string
  driftDetail: string
  usage: { cost_usd: number; calls: number }
  dataset: IngestedDatasetSummary | null
  sessionStatus?: string
  onFocusRound: (round: number) => void
  onExport: () => void
  isExporting: boolean
  hasPolicy: boolean
}

const DriftPill: React.FC<{ status: string; detail: string }> = ({ status, detail }) => {
  const tone =
    status === 'healthy'
      ? 'border-emerald-800/60 bg-emerald-950/30 text-emerald-300'
      : status === 'unmonitored'
        ? 'border-gray-700 bg-gray-900 text-gray-400'
        : status === 'warming_up'
          ? 'border-slate-700 bg-slate-900 text-slate-300'
          : 'border-amber-800/60 bg-amber-950/30 text-amber-300'
  const Icon = status === 'healthy' ? CheckCircle2 : status === 'drifted' ? AlertTriangle : Activity
  return (
    <div className={`rounded-lg border px-2.5 py-1.5 ${tone}`} title={detail || undefined}>
      <div className="flex items-center justify-between gap-2">
        <span className="flex items-center gap-1.5 text-[11px] font-medium">
          <Icon className="h-3 w-3" />
          Drift
        </span>
        <span className="font-mono text-[10px] capitalize">{status.replace('_', ' ')}</span>
      </div>
      {detail && <p className="mt-1 text-[10px] leading-snug opacity-80 line-clamp-2">{detail}</p>}
    </div>
  )
}

/**
 * Everything numeric, pinned and read-only. Nothing here starts work: busy numbers drift out of
 * the conversation and get lost, which is exactly why they do not belong in it.
 */
export const ScoreboardRail: React.FC<ScoreboardRailProps> = ({
  targetMetricName,
  targetThreshold,
  currentMetric,
  currentRound,
  maxRounds,
  roundHistory,
  scorecard,
  driftStatus,
  driftDetail,
  usage,
  dataset,
  sessionStatus,
  onFocusRound,
  onExport,
  isExporting,
  hasPolicy,
}) => {
  const pct = Math.max(0, Math.min(100, targetThreshold ? (currentMetric / targetThreshold) * 100 : 0))
  const best = roundHistory.length
    ? roundHistory.reduce((a, b) => (b.metric > a.metric ? b : a), roundHistory[0])
    : null

  return (
    <aside className="w-80 shrink-0 flex flex-col h-full border-r border-gray-800 bg-gray-950/80">
      <div className="px-3 py-2 border-b border-gray-800 flex items-center justify-between">
        <span className="flex items-center gap-1.5 text-[11px] font-bold uppercase tracking-wider text-gray-400">
          <Gauge className="h-3.5 w-3.5 text-cyan-400" />
          Scoreboard
        </span>
        {sessionStatus && (
          <span className="rounded-full border border-gray-700 bg-gray-900 px-2 py-0.5 text-[10px] font-mono text-gray-400">
            {sessionStatus}
          </span>
        )}
      </div>

      <div className="flex-1 overflow-y-auto px-3 py-3 space-y-3">
        {/* Target vs current */}
        <section className="rounded-xl border border-gray-800 bg-gray-900/60 p-3">
          <div className="flex items-center justify-between text-[11px] text-gray-400">
            <span className="flex items-center gap-1.5">
              <Target className="h-3 w-3 text-indigo-400" />
              {targetMetricName || 'Success rate'}
            </span>
            <span className="font-mono text-[10px]">
              R{Math.min(currentRound, maxRounds)}/{maxRounds}
            </span>
          </div>
          <div className="mt-1 flex items-baseline gap-1">
            <span className="text-2xl font-bold text-gray-100 font-mono">{currentMetric.toFixed(1)}</span>
            <span className="text-sm text-gray-500 font-mono">%</span>
            <span className="ml-auto text-[11px] text-gray-500 font-mono">
              target {targetThreshold.toFixed(0)}%
            </span>
          </div>
          <div className="mt-2 h-2 w-full overflow-hidden rounded-full bg-gray-800">
            <div
              className={`h-full rounded-full transition-all duration-500 ${
                currentMetric >= targetThreshold
                  ? 'bg-emerald-500'
                  : 'bg-gradient-to-r from-indigo-500 to-cyan-400'
              }`}
              style={{ width: `${pct}%` }}
            />
          </div>
        </section>

        {/* Held-out scorecard */}
        <section className="rounded-xl border border-gray-800 bg-gray-900/60 p-3 space-y-2">
          <div className="flex items-center justify-between text-[11px] text-gray-400">
            <span className="flex items-center gap-1.5">
              <BarChart3 className="h-3 w-3 text-emerald-400" />
              Held-out scorecard
            </span>
            {scorecard?.crossfire && (
              <span
                className={`rounded-full border px-1.5 py-0.5 font-mono text-[10px] ${
                  scorecard.crossfire.verdict === 'win'
                    ? 'border-emerald-700/60 bg-emerald-950/60 text-emerald-300'
                    : scorecard.crossfire.verdict === 'loss'
                      ? 'border-rose-800/60 bg-rose-950/50 text-rose-300'
                      : 'border-amber-800/60 bg-amber-950/40 text-amber-300'
                }`}
                title={(scorecard.crossfire.reasons || []).join(' ')}
              >
                {String(scorecard.crossfire.verdict).replace(/_/g, ' ')}
              </span>
            )}
          </div>
          {scorecard ? (
            <>
              <div className="flex items-baseline justify-between">
                <span className="font-mono text-lg font-bold text-gray-100">
                  {scorecard.policy.value_pct.toFixed(1)}%
                </span>
                <span className="text-[10px] text-gray-500 font-mono">
                  {scorecard.eval_split} · n={scorecard.policy.episodes}
                </span>
              </div>
              <div className="space-y-1">
                {scorecard.baselines.slice(0, 3).map((b) => (
                  <div key={b.name} className="flex items-center justify-between text-[10px]">
                    <span className="text-gray-500 truncate">{b.name.replace(/_/g, ' ')}</span>
                    <span className="font-mono text-gray-300">{b.value_pct.toFixed(1)}%</span>
                  </div>
                ))}
              </div>
              <div className="flex flex-wrap gap-1 pt-1">
                <span
                  className={`rounded-full border px-1.5 py-0.5 font-mono text-[10px] ${
                    scorecard.beats_all_baselines
                      ? 'border-emerald-800/60 bg-emerald-950/40 text-emerald-300'
                      : 'border-amber-800/60 bg-amber-950/30 text-amber-300'
                  }`}
                >
                  {scorecard.beats_all_baselines ? 'beats all baselines' : 'below a baseline'}
                </span>
                {scorecard.eval_corpus && (
                  <span className="rounded-full border border-gray-700 bg-gray-950 px-1.5 py-0.5 font-mono text-[10px] text-gray-300">
                    corpus {scorecard.eval_corpus.pass_rate_pct.toFixed(0)}%
                  </span>
                )}
                {scorecard.safety && scorecard.safety.violations > 0 && (
                  <span className="rounded-full border border-rose-800/60 bg-rose-950/40 px-1.5 py-0.5 font-mono text-[10px] text-rose-300">
                    {scorecard.safety.violations} violations
                  </span>
                )}
              </div>
            </>
          ) : (
            <p className="text-[10px] text-gray-500 leading-snug">
              No held-out evaluation yet. It is scored on unseen rows when a run completes.
            </p>
          )}
        </section>

        <DriftPill status={driftStatus} detail={driftDetail} />

        {/* Rounds */}
        <section className="rounded-xl border border-gray-800 bg-gray-900/60 p-3 space-y-1.5">
          <div className="flex items-center justify-between text-[11px] text-gray-400">
            <span className="flex items-center gap-1.5">
              <Activity className="h-3 w-3 text-cyan-400" />
              Rounds
            </span>
            {best && (
              <span className="font-mono text-[10px] text-gray-500">
                best {best.metric.toFixed(1)}%
              </span>
            )}
          </div>
          {roundHistory.length ? (
            <ul className="space-y-1">
              {roundHistory.map((r) => (
                <li key={r.round_num}>
                  <button
                    onClick={() => onFocusRound(r.round_num)}
                    className="w-full flex items-center justify-between gap-2 rounded px-1 py-0.5 text-[10px] hover:bg-gray-800/70 transition"
                    title={r.diagnosis || 'Jump to this round in the thread'}
                  >
                    <span className="font-mono text-gray-500">R{r.round_num}</span>
                    <span className="flex-1 h-1 rounded-full bg-gray-800 overflow-hidden">
                      <span
                        className={`block h-full ${
                          r.metric >= r.target_threshold ? 'bg-emerald-500' : 'bg-indigo-500'
                        }`}
                        style={{ width: `${Math.max(0, Math.min(100, r.metric))}%` }}
                      />
                    </span>
                    <span className="font-mono text-gray-300">{r.metric.toFixed(1)}%</span>
                  </button>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-[10px] text-gray-500">No rounds yet.</p>
          )}
        </section>

        {/* Context */}
        <section className="space-y-1.5 text-[10px] text-gray-500">
          <div className="flex items-center justify-between">
            <span className="flex items-center gap-1.5">
              <Database className="h-3 w-3" /> Dataset
            </span>
            <span className="truncate max-w-[150px] text-gray-300">
              {dataset ? dataset.name : 'none'}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="flex items-center gap-1.5">
              <ShieldCheck className="h-3 w-3" /> Spend
            </span>
            <span className="font-mono text-gray-300">
              ${usage.cost_usd < 0.01 ? usage.cost_usd.toFixed(4) : usage.cost_usd.toFixed(2)} ·{' '}
              {usage.calls} call{usage.calls === 1 ? '' : 's'}
            </span>
          </div>
        </section>
      </div>

      <div className="border-t border-gray-800 px-3 py-2">
        <button
          onClick={onExport}
          disabled={!hasPolicy || isExporting}
          className="w-full flex items-center justify-center gap-1.5 rounded-md border border-gray-700 bg-gray-900 px-2 py-1.5 text-[11px] text-gray-300 hover:border-gray-500 transition disabled:opacity-40"
          title={hasPolicy ? 'Export the policy as an ONNX runbundle' : 'Train a policy first'}
        >
          <Download className="h-3 w-3 text-cyan-400" />
          {isExporting ? 'Exporting…' : 'Export policy (.onnx)'}
        </button>
      </div>
    </aside>
  )
}
