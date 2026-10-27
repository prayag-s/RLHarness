import React, { useState } from 'react'
import {
  MessageSquare,
  Sparkles,
  Zap,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  ArrowRight,
  ShieldAlert,
  Star,
  Activity,
  Layers,
  Check,
  X,
} from 'lucide-react'
import { EpisodeLog, RewardDiffProposal } from '../types'

interface RightPanelProps {
  adapter?: string
  selectedEpisode: EpisodeLog | null
  onPreviewCritique: (critiqueText: string) => void
  isPreviewing: boolean
  diffProposal: RewardDiffProposal | null
  onApplyDiff: () => void
  isApplying: boolean
  onDiscardDiff: () => void
  isGoldenLocked: boolean
}

const STREAM_CHIPS = [
  'Penalize false alarms when rolling volatility is normal',
  'Heavily penalize alerts triggered while cooldown timer is active',
  'Reward high-volatility spike detection even more aggressively',
]

const TEXT_CHIPS = [
  'Never route spam as urgent; heavily penalize routing spam to urgent',
  'Escalate angry VIP customer tickets immediately',
  'Reward accurate ticket triage routing matches even more aggressively',
]

export const RightPanel: React.FC<RightPanelProps> = ({
  adapter = 'CSVStreamAdapter',
  selectedEpisode,
  onPreviewCritique,
  isPreviewing,
  diffProposal,
  onApplyDiff,
  isApplying,
  onDiscardDiff,
  isGoldenLocked,
}) => {
  const [critiqueText, setCritiqueText] = useState('')
  const quickChips = adapter === 'TextClassifyAdapter' ? TEXT_CHIPS : STREAM_CHIPS

  const lastStep = selectedEpisode?.steps?.[selectedEpisode.steps.length - 1]
  const actionTaken =
    selectedEpisode?.steps?.find((s) => s.events?.action?.label === 'flag_anomaly')
      ? 'flag_anomaly'
      : lastStep?.events?.action?.label || 'do_nothing'

  const avgPrice = selectedEpisode?.steps?.length
    ? (
        selectedEpisode.steps.reduce(
          (acc, s) => acc + Number(s.events?.obs?.price ?? s.obs_vector?.[0] ?? 0),
          0,
        ) / selectedEpisode.steps.length
      ).toFixed(2)
    : '0.00'

  const maxVol = selectedEpisode?.steps?.length
    ? Math.max(
        ...selectedEpisode.steps.map((s) =>
          Number(s.events?.obs?.volatility ?? s.obs_vector?.[1] ?? 0),
        ),
      ).toFixed(4)
    : '0.0000'

  const totalAlerts =
    selectedEpisode?.steps?.filter(
      (s) => s.events?.action?.label === 'flag_anomaly' || s.events?.action?.action_idx === 1,
    ).length || 0

  return (
    <div className="w-[28%] min-w-[320px] max-w-[420px] h-full flex flex-col border-l border-panel-border bg-panel overflow-hidden">
      {/* Inspected Episode Telemetry Card */}
      <div className="p-3 border-b border-panel-border bg-gray-900/60">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center space-x-1.5">
            <Activity className="h-4 w-4 text-purple-400" />
            <h2 className="text-xs font-bold text-gray-200 uppercase tracking-wider">
              Selected Episode Telemetry
            </h2>
          </div>
          {selectedEpisode && (
            <div className="flex items-center space-x-1">
              {isGoldenLocked && (
                <span className="text-[10px] bg-amber-950/80 text-amber-400 px-1.5 py-0.5 rounded border border-amber-800/40 flex items-center space-x-1">
                  <Star className="h-3 w-3 fill-amber-400 text-amber-400" />
                  <span>Locked</span>
                </span>
              )}
              <span className="text-xs font-mono text-blue-400 font-bold bg-blue-950/60 px-2 py-0.5 rounded border border-blue-800/40">
                #{selectedEpisode.episode_id}
              </span>
            </div>
          )}
        </div>

        {selectedEpisode ? (
          <div className="grid grid-cols-2 gap-2 text-xs font-mono">
            <div className="bg-gray-950/60 p-2 rounded border border-gray-800">
              <span className="text-[10px] text-gray-500 block">Total Reward</span>
              <span
                className={`text-sm font-bold ${
                  selectedEpisode.total_reward >= 0 ? 'text-emerald-400' : 'text-rose-400'
                }`}
              >
                {selectedEpisode.total_reward >= 0 ? '+' : ''}
                {selectedEpisode.total_reward.toFixed(2)}
              </span>
            </div>

            <div className="bg-gray-950/60 p-2 rounded border border-gray-800">
              <span className="text-[10px] text-gray-500 block">Anomaly Alerts</span>
              <span className="text-sm font-bold text-gray-200">{totalAlerts}</span>
            </div>

            <div className="bg-gray-950/60 p-2 rounded border border-gray-800">
              <span className="text-[10px] text-gray-500 block">Avg Price</span>
              <span className="text-xs text-gray-300">{avgPrice}</span>
            </div>

            <div className="bg-gray-950/60 p-2 rounded border border-gray-800">
              <span className="text-[10px] text-gray-500 block">Peak Volatility</span>
              <span className="text-xs text-amber-300">{maxVol}</span>
            </div>

            <div className="col-span-2 bg-gray-950/60 px-2 py-1.5 rounded border border-gray-800 flex justify-between items-center">
              <span className="text-[10px] text-gray-500">Representative Action:</span>
              <span
                className={`px-1.5 py-0.5 rounded text-[11px] font-bold ${
                  actionTaken === 'flag_anomaly'
                    ? 'bg-rose-950 text-rose-300 border border-rose-800/50'
                    : 'bg-gray-800 text-gray-300'
                }`}
              >
                {actionTaken}
              </span>
            </div>
          </div>
        ) : (
          <div className="text-center py-6 text-gray-500 text-xs">
            No episode selected. Run a burst or click an episode strip card.
          </div>
        )}
      </div>

      {/* English Critique Section */}
      <div className="p-3 border-b border-panel-border bg-gray-950/30">
        <div className="flex items-center space-x-1.5 mb-2">
          <MessageSquare className="h-4 w-4 text-emerald-400" />
          <h3 className="text-xs font-bold text-gray-200 uppercase tracking-wider">
            English Critique
          </h3>
        </div>

        {/* Suggestion Chips */}
        <div className="mb-2 space-y-1">
          <span className="text-[10px] text-gray-500 block uppercase font-bold">
            Quick Suggestions:
          </span>
          <div className="flex flex-col space-y-1">
            {quickChips.map((chip, idx) => (
              <button
                key={idx}
                onClick={() => setCritiqueText(chip)}
                className="text-left text-[11px] bg-gray-900/80 hover:bg-gray-800 text-gray-300 hover:text-white px-2 py-1 rounded border border-gray-800 transition truncate"
                title={chip}
              >
                ⚡ {chip}
              </button>
            ))}
          </div>
        </div>

        {/* Critique Input Area */}
        <textarea
          value={critiqueText}
          onChange={(e) => setCritiqueText(e.target.value)}
          placeholder="Tell the agent what to improve in plain English..."
          rows={2}
          className="w-full text-xs bg-gray-950/80 border border-gray-800 rounded-lg p-2 text-gray-200 placeholder-gray-500 focus:outline-none focus:border-emerald-500 transition resize-none leading-relaxed mb-2"
        />

        <button
          onClick={() => onPreviewCritique(critiqueText)}
          disabled={isPreviewing || !critiqueText.trim() || !selectedEpisode}
          className="w-full flex items-center justify-center space-x-2 py-2 px-3 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-medium text-xs shadow-md shadow-emerald-600/20 transition disabled:opacity-40 disabled:cursor-not-allowed"
        >
          {isPreviewing ? (
            <>
              <div className="h-3.5 w-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              <span>Running 2-Stage Preview...</span>
            </>
          ) : (
            <>
              <Zap className="h-3.5 w-3.5" />
              <span>Preview Critique (Two-Stage)</span>
            </>
          )}
        </button>
      </div>

      {/* Interactive Diff Card (Workbench) */}
      <div className="flex-1 overflow-y-auto p-3">
        {diffProposal ? (
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-gray-200 uppercase tracking-wider flex items-center space-x-1.5">
                <Sparkles className="h-4 w-4 text-amber-400" />
                <span>Proposed Reward Diff</span>
              </span>
              <span className="text-[10px] text-amber-400 font-mono bg-amber-950/60 px-1.5 py-0.5 rounded border border-amber-800/40">
                Pending Approval
              </span>
            </div>

            {/* Conflicts Warning */}
            {diffProposal.conflicts && diffProposal.conflicts.length > 0 && (
              <div className="p-2.5 rounded-lg bg-amber-950/40 border border-amber-700/60 text-xs space-y-1">
                <div className="flex items-center space-x-1.5 text-amber-400 font-semibold">
                  <AlertTriangle className="h-4 w-4" />
                  <span>Conflict Warning Detected</span>
                </div>
                {diffProposal.conflicts.map((c, i) => (
                  <p key={i} className="text-amber-200/90 text-[11px] leading-relaxed">
                    <strong>{c.term_name}:</strong> {c.message}
                  </p>
                ))}
              </div>
            )}

            {/* Term Diffs */}
            <div className="space-y-2">
              {diffProposal.term_diffs?.map((td, i) => {
                const isAdded = td.action === 'added'
                const isRemoved = td.action === 'removed'
                const isModified = td.action === 'modified'

                return (
                  <div
                    key={i}
                    className="p-2.5 rounded-lg bg-gray-950 border border-gray-800 text-xs space-y-1"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-1.5">
                        <span
                          className={`font-mono font-bold text-[10px] px-1.5 py-0.2 rounded uppercase ${
                            isAdded
                              ? 'bg-emerald-950 text-emerald-400 border border-emerald-800'
                              : isRemoved
                              ? 'bg-rose-950 text-rose-400 border border-rose-800'
                              : 'bg-amber-950 text-amber-400 border border-amber-800'
                          }`}
                        >
                          {isAdded ? '+ Added' : isRemoved ? '- Removed' : '~ Modified'}
                        </span>
                        <span className="font-mono text-gray-200 font-semibold">
                          {td.term_name}
                        </span>
                      </div>
                    </div>

                    <p className="text-[11px] text-gray-400 italic">"{td.rationale}"</p>

                    {td.new_term && (
                      <div className="bg-gray-900/80 p-1.5 rounded font-mono text-[10px] text-gray-300">
                        <div>
                          <span className="text-gray-500">when: </span>
                          <span className="text-amber-300">{String(td.new_term.when)}</span>
                        </div>
                        <div>
                          <span className="text-gray-500">value: </span>
                          <span className="text-emerald-400 font-bold">
                            {Number(td.new_term.value) > 0 ? '+' : ''}
                            {td.new_term.value}
                          </span>
                        </div>
                      </div>
                    )}
                  </div>
                )
              })}
            </div>

            {/* Stage 1 Preview: Counterfactual Delta */}
            {diffProposal.stage1_counterfactual_preview && (
              <div className="p-2.5 rounded-lg bg-blue-950/30 border border-blue-800/60 text-xs space-y-1.5">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-blue-300 text-[11px] uppercase tracking-wider flex items-center space-x-1">
                    <Zap className="h-3 w-3 text-blue-400" />
                    <span>Stage 1: Counterfactual Score Delta</span>
                  </span>
                </div>
                <div className="flex items-center justify-between font-mono text-xs bg-gray-950/70 p-2 rounded border border-gray-800">
                  <span className="text-gray-400">
                    Before:{' '}
                    <strong className="text-gray-200">
                      {diffProposal.stage1_counterfactual_preview.before_score.toFixed(1)}
                    </strong>
                  </span>
                  <ArrowRight className="h-3 w-3 text-gray-500" />
                  <span className="text-gray-400">
                    After:{' '}
                    <strong className="text-blue-300">
                      {diffProposal.stage1_counterfactual_preview.after_score.toFixed(1)}
                    </strong>
                  </span>
                  <span
                    className={`font-bold px-1.5 py-0.2 rounded text-[11px] ${
                      diffProposal.stage1_counterfactual_preview.score_delta >= 0
                        ? 'bg-emerald-950 text-emerald-400 border border-emerald-800'
                        : 'bg-rose-950 text-rose-400 border border-rose-800'
                    }`}
                  >
                    Δ{' '}
                    {diffProposal.stage1_counterfactual_preview.score_delta >= 0 ? '+' : ''}
                    {diffProposal.stage1_counterfactual_preview.score_delta.toFixed(1)}
                  </span>
                </div>
                <p className="text-[10px] text-gray-400 leading-normal">
                  {diffProposal.stage1_counterfactual_preview.summary}
                </p>
              </div>
            )}

            {/* Stage 2 Preview: Forked Policy Action Flip */}
            {diffProposal.stage2_forked_behavior_preview && (
              <div className="p-2.5 rounded-lg bg-purple-950/30 border border-purple-800/60 text-xs space-y-1.5">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-purple-300 text-[11px] uppercase tracking-wider flex items-center space-x-1">
                    <Layers className="h-3 w-3 text-purple-400" />
                    <span>Stage 2: Forked Policy Flip Rate</span>
                  </span>
                </div>

                <div className="bg-gray-950/70 p-2 rounded border border-gray-800 font-mono text-xs space-y-1">
                  <div className="flex items-center justify-between">
                    <span className="text-gray-400">Target Episode:</span>
                    <span className="text-gray-300">
                      {diffProposal.stage2_forked_behavior_preview.target_action_before} →{' '}
                      <strong className="text-purple-300">
                        {diffProposal.stage2_forked_behavior_preview.target_action_after}
                      </strong>
                    </span>
                  </div>

                  <div className="flex items-center justify-between">
                    <span className="text-gray-400">Action Flipped?</span>
                    <span
                      className={`font-bold ${
                        diffProposal.stage2_forked_behavior_preview.target_flipped
                          ? 'text-emerald-400'
                          : 'text-amber-400'
                      }`}
                    >
                      {diffProposal.stage2_forked_behavior_preview.target_flipped
                        ? 'YES (Corrected ✅)'
                        : 'NO (Same Action)'}
                    </span>
                  </div>

                  <div className="flex items-center justify-between pt-1 border-t border-gray-800 text-[11px]">
                    <span className="text-gray-400">Batch Flip Rate:</span>
                    <span className="text-gray-200">
                      {diffProposal.stage2_forked_behavior_preview.total_actions_flipped} /{' '}
                      {diffProposal.stage2_forked_behavior_preview.total_episodes_evaluated} (
                      {(
                        diffProposal.stage2_forked_behavior_preview.flip_rate * 100
                      ).toFixed(1)}
                      %)
                    </span>
                  </div>
                </div>

                <div className="flex items-center space-x-1.5 text-[11px] text-emerald-400">
                  <CheckCircle2 className="h-3.5 w-3.5" />
                  <span>Preserved 100% of locked golden episodes</span>
                </div>
              </div>
            )}

            {/* Action Buttons: Apply & Discard */}
            <div className="pt-2 flex space-x-2">
              <button
                onClick={onApplyDiff}
                disabled={isApplying}
                className="flex-1 flex items-center justify-center space-x-1.5 py-2 px-3 rounded-lg bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white font-medium text-xs shadow-md shadow-emerald-600/20 transition disabled:opacity-40"
              >
                {isApplying ? (
                  <>
                    <div className="h-3 w-3 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    <span>Applying & Warming Critic...</span>
                  </>
                ) : (
                  <>
                    <Check className="h-3.5 w-3.5" />
                    <span>Apply Diff & Run Burst</span>
                  </>
                )}
              </button>

              <button
                onClick={onDiscardDiff}
                disabled={isApplying}
                className="px-3 py-2 rounded-lg bg-gray-900 hover:bg-gray-800 text-gray-400 hover:text-gray-200 border border-gray-800 text-xs font-medium transition"
              >
                <X className="h-3.5 w-3.5" />
              </button>
            </div>
          </div>
        ) : (
          <div className="h-44 flex flex-col items-center justify-center text-center p-4 border border-dashed border-gray-800 rounded-xl text-gray-500 text-xs space-y-2">
            <Sparkles className="h-6 w-6 text-gray-600" />
            <p>
              No pending diff proposal. Type an English critique above and click{' '}
              <strong className="text-gray-400">[Preview Critique]</strong>.
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
