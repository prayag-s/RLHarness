import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  ShieldCheck,
  ShieldAlert,
  ShieldQuestion,
  X,
  RefreshCw,
  Lock,
  Ban,
  Upload,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  History,
  ListChecks,
  ChevronRight,
} from 'lucide-react'
import {
  CrossfireClosure,
  EvalCorpusCase,
  EvalCorpusResponse,
  GateDecision,
  GateRecord,
  ObservedStep,
  PassiveHarvestResult,
  SessionState,
} from '../types'
import {
  addCorpusCase,
  fetchCrossfireClosure,
  fetchEvalCorpus,
  fetchGateHistory,
  harvestPassiveLog,
  uploadDatasetFile,
  waiveCorpusCase,
} from '../services/api'

type Tab = 'crossfire' | 'corpus' | 'gate'

interface TrustPanelProps {
  isOpen: boolean
  onClose: () => void
  conversationId?: string
  session?: SessionState | null
  /** The last step the user inspected in the Arena; the anchor for a hand-locked assertion. */
  lastStep?: ObservedStep | null
  /** Bumped by the app when training finishes, so an open panel refetches. */
  refreshKey?: number
  actionLabels?: string[]
  onCorpusChanged?: () => void
}

function actionName(index: number | null, labels: string[]): string {
  if (index === null || index < 0) return '—'
  const label = labels[index]
  return label ? `${index} · ${label}` : String(index)
}

function verdictStyle(verdict: string) {
  switch (verdict) {
    case 'win':
      return { icon: ShieldCheck, tone: 'text-emerald-300', border: 'border-emerald-800/60', bg: 'bg-emerald-950/40' }
    case 'loss':
      return { icon: XCircle, tone: 'text-rose-300', border: 'border-rose-800/60', bg: 'bg-rose-950/40' }
    case 'inconclusive':
      return { icon: ShieldQuestion, tone: 'text-amber-300', border: 'border-amber-800/60', bg: 'bg-amber-950/40' }
    default:
      return { icon: AlertTriangle, tone: 'text-gray-300', border: 'border-gray-700', bg: 'bg-gray-900/60' }
  }
}

export const TrustPanel: React.FC<TrustPanelProps> = ({
  isOpen,
  onClose,
  conversationId,
  session,
  lastStep,
  refreshKey = 0,
  actionLabels = [],
  onCorpusChanged,
}) => {
  const [tab, setTab] = useState<Tab>('crossfire')
  const [corpus, setCorpus] = useState<EvalCorpusResponse | null>(null)
  const [closure, setClosure] = useState<CrossfireClosure | null>(null)
  const [gateEntries, setGateEntries] = useState<GateRecord[]>([])
  const [lastGate, setLastGate] = useState<GateDecision | null>(null)
  const [pendingKind, setPendingKind] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [lockBusy, setLockBusy] = useState(false)
  const [lockRationale, setLockRationale] = useState('')
  const [lockNote, setLockNote] = useState<string | null>(null)
  const [waiveTarget, setWaiveTarget] = useState<string | null>(null)
  const [waiveReason, setWaiveReason] = useState('')
  const [waiveBusy, setWaiveBusy] = useState(false)
  const [harvestBusy, setHarvestBusy] = useState(false)
  const [harvest, setHarvest] = useState<PassiveHarvestResult | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const load = useCallback(async () => {
    if (!conversationId) return
    setIsLoading(true)
    setError(null)
    try {
      const [corpusRes, crossfireRes, gatesRes] = await Promise.all([
        fetchEvalCorpus(conversationId),
        fetchCrossfireClosure(conversationId),
        fetchGateHistory(conversationId),
      ])
      setCorpus(corpusRes)
      setClosure(crossfireRes.closure)
      setGateEntries(gatesRes.entries ?? [])
      setLastGate(gatesRes.last ?? null)
      setPendingKind(gatesRes.mutation_pending ?? null)
    } catch (err: any) {
      setError(err?.message || 'Could not load the trust state for this conversation.')
    } finally {
      setIsLoading(false)
    }
  }, [conversationId])

  useEffect(() => {
    if (isOpen) void load()
  }, [isOpen, load, refreshKey])

  const cases = corpus?.cases ?? []
  const report = corpus?.report ?? null
  const failingIds = useMemo(() => new Set(report?.failing_case_ids ?? []), [report])
  const gateHistory = gateEntries.length ? gateEntries : session?.gate_history ?? []
  const newestGate = lastGate ?? session?.critique_gate ?? null
  const mutationPending = pendingKind ?? session?.mutation_pending ?? null

  const handleLock = async (mode: 'expected' | 'forbidden') => {
    if (!conversationId || !lastStep || lastStep.action === null) return
    setLockBusy(true)
    setLockNote(null)
    try {
      await addCorpusCase(conversationId, {
        obs_vector: lastStep.obs,
        expected_actions: mode === 'expected' ? [lastStep.action] : [],
        forbidden_actions: mode === 'forbidden' ? [lastStep.action] : [],
        rationale:
          lockRationale.trim() ||
          (mode === 'expected'
            ? 'Action confirmed while playing the arena.'
            : 'Action rejected while playing the arena.'),
        source: 'user_locked_step',
      })
      setLockRationale('')
      setLockNote(
        mode === 'expected'
          ? 'Locked as an expectation: the policy must keep choosing this action here.'
          : 'Locked as a prohibition: the policy must never choose this action here.',
      )
      await load()
      onCorpusChanged?.()
    } catch (err: any) {
      setLockNote(err?.message || 'Could not lock this step.')
    } finally {
      setLockBusy(false)
    }
  }

  const handleWaive = async () => {
    if (!conversationId || !waiveTarget) return
    setWaiveBusy(true)
    try {
      await waiveCorpusCase(conversationId, waiveTarget, waiveReason.trim())
      setWaiveTarget(null)
      setWaiveReason('')
      await load()
      onCorpusChanged?.()
    } catch (err: any) {
      setError(err?.message || 'Could not waive this case.')
    } finally {
      setWaiveBusy(false)
    }
  }

  const handleHarvestFile = async (file: File | null) => {
    if (!file || !conversationId) return
    setHarvestBusy(true)
    setHarvest(null)
    setError(null)
    try {
      const dataset = await uploadDatasetFile(file)
      const result = await harvestPassiveLog(conversationId, dataset.dataset_id)
      setHarvest(result)
      await load()
      onCorpusChanged?.()
    } catch (err: any) {
      setError(err?.message || 'Could not harvest this log.')
    } finally {
      setHarvestBusy(false)
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  if (!isOpen) return null

  const verdict = closure?.verdict ?? 'not_preregistered'
  const vs = verdictStyle(verdict)
  const VerdictIcon = vs.icon
  const reasons = closure?.reasons ?? []
  const holdoutRegistered = closure?.holdout_registered ?? closure?.holdout_total ?? 0
  const minHoldout = closure?.min_holdout ?? 100
  const holdoutPct = Math.min(100, Math.round((100 * holdoutRegistered) / Math.max(1, minHoldout)))

  const tabs: Array<{ id: Tab; label: string; badge?: number }> = [
    { id: 'crossfire', label: 'Crossfire', badge: reasons.length || undefined },
    { id: 'corpus', label: 'Corpus', badge: cases.length || undefined },
    { id: 'gate', label: 'Gate', badge: gateHistory.length || undefined },
  ]

  return (
    <div className="fixed inset-0 z-50 flex justify-end">
      <div className="absolute inset-0 bg-black/60" onClick={onClose} />

      <aside className="relative w-[30rem] max-w-full h-full flex flex-col bg-gray-950 border-l border-gray-800 text-gray-200 shadow-2xl">
        {/* Header */}
        <div className="px-4 py-3 border-b border-gray-800 flex items-start justify-between bg-gray-900/60">
          <div>
            <div className="flex items-center space-x-2">
              <ShieldCheck className="h-4 w-4 text-emerald-400" />
              <span className="text-sm font-bold tracking-wide">Trust &amp; Provenance</span>
            </div>
            <p className="text-[10px] text-gray-500 font-mono mt-0.5">
              {conversationId ? conversationId : 'no active conversation'}
            </p>
          </div>
          <div className="flex items-center space-x-1">
            <button
              onClick={() => void load()}
              disabled={isLoading || !conversationId}
              className="p-1.5 rounded-lg text-gray-400 hover:text-gray-200 hover:bg-gray-800 disabled:opacity-40"
              title="Refetch corpus and protocol"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${isLoading ? 'animate-spin' : ''}`} />
            </button>
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg text-gray-400 hover:text-gray-200 hover:bg-gray-800"
              title="Close"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-gray-800 bg-gray-900/30">
          {tabs.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`flex-1 px-3 py-2 text-[11px] font-semibold border-b-2 transition ${
                tab === t.id
                  ? 'border-emerald-500 text-emerald-300'
                  : 'border-transparent text-gray-500 hover:text-gray-300'
              }`}
            >
              {t.label}
              {t.badge !== undefined && (
                <span className="ml-1.5 text-[10px] font-mono text-gray-500">{t.badge}</span>
              )}
            </button>
          ))}
        </div>

        {error && (
          <div className="px-4 py-2 text-[11px] bg-rose-950/40 border-b border-rose-900/50 text-rose-300">
            {error}
          </div>
        )}

        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {/* ---------------------------------------------------------------- */}
          {/* Crossfire                                                        */}
          {/* ---------------------------------------------------------------- */}
          {tab === 'crossfire' && (
            <>
              <div className={`rounded-xl border p-3 space-y-2 ${vs.border} ${vs.bg}`}>
                <div className={`flex items-center space-x-2 ${vs.tone}`}>
                  <VerdictIcon className="h-4 w-4" />
                  <span className="text-xs font-bold uppercase tracking-wider">
                    {verdict.replace(/_/g, ' ')}
                  </span>
                  <span className="text-[10px] font-mono text-gray-500">
                    {closure?.closed ? 'closed' : 'not closed'}
                  </span>
                </div>
                <p className="text-[11px] text-gray-400 leading-snug">
                  {verdict === 'win' &&
                    'Pre-registered rubric, enough frozen judgements and a measured baseline: the claim holds.'}
                  {verdict === 'loss' &&
                    'Pre-registered and scored honestly below target. A loss is a result, not a failure of the protocol.'}
                  {verdict === 'inconclusive' &&
                    'The protocol is sound but the evidence is not yet sufficient. Nothing below is a verdict on the policy.'}
                  {verdict === 'not_preregistered' &&
                    'No falsification protocol exists yet. It is opened automatically when training starts.'}
                </p>
              </div>

              <div>
                <div className="flex items-center justify-between text-[10px] text-gray-500 font-mono mb-1">
                  <span>frozen hold-out</span>
                  <span>
                    {holdoutRegistered} / {minHoldout}
                  </span>
                </div>
                <div className="w-full bg-gray-800 rounded-full h-1.5 overflow-hidden">
                  <div
                    className={`h-full rounded-full ${holdoutPct >= 100 ? 'bg-emerald-500' : 'bg-amber-500'}`}
                    style={{ width: `${holdoutPct}%` }}
                  />
                </div>
                {typeof closure?.holdout_pass_rate_pct === 'number' && closure.holdout_total ? (
                  <p className="text-[10px] text-gray-500 font-mono mt-1">
                    scored {closure.holdout_passed}/{closure.holdout_total} · pass{' '}
                    {closure.holdout_pass_rate_pct}% vs target {closure.target_pass_rate_pct ?? '—'}%
                  </p>
                ) : null}
              </div>

              <div>
                <div className="text-[10px] uppercase tracking-wider text-gray-500 font-bold mb-1.5">
                  Blocking closure
                </div>
                {reasons.length === 0 ? (
                  <p className="text-[11px] text-gray-400">
                    {verdict === 'not_preregistered'
                      ? 'Nothing to block yet: the protocol is opened automatically at the first training round.'
                      : 'No pre-registration gaps — the rubric, the training stamp and the frozen hold-out set all agree.'}
                  </p>
                ) : (
                  <ol className="space-y-1.5">
                    {reasons.map((reason, idx) => (
                      <li
                        key={idx}
                        className="flex items-start space-x-2 text-[11px] text-amber-200/90 bg-amber-950/20 border border-amber-900/40 rounded-lg px-2 py-1.5"
                      >
                        <span className="text-amber-500 font-mono text-[10px] mt-0.5">{idx + 1}</span>
                        <span className="leading-snug">{reason}</span>
                      </li>
                    ))}
                  </ol>
                )}
              </div>

              <div className="rounded-xl border border-gray-800 bg-gray-900/40 p-3 space-y-1.5 text-[10px] font-mono">
                <Row label="policy" value={closure?.policy_metric != null ? `${closure.policy_metric}%` : '—'} />
                <Row
                  label="best baseline"
                  value={closure?.baseline_metric != null ? `${closure.baseline_metric}%` : 'not measured'}
                />
                <Row
                  label="beats baseline"
                  value={closure?.beats_baseline === undefined ? '—' : closure.beats_baseline ? 'yes' : 'no'}
                />
                <Row label="opened" value={closure?.opened_at ?? '—'} />
                <Row label="training stamp" value={closure?.training_started_at ?? 'missing'} />
                <Row
                  label="rubric hash"
                  value={closure?.rubric_hash ? `${closure.rubric_hash.slice(0, 12)}…` : '—'}
                  title={closure?.rubric_hash}
                />
              </div>

              <div className="rounded-xl border border-gray-800 bg-gray-900/30 p-3">
                <div className="text-[10px] uppercase tracking-wider text-gray-500 font-bold mb-1">
                  What earns a win
                </div>
                <ul className="text-[11px] text-gray-400 space-y-1">
                  <li>· Rubric registered before the training stamp (automatic).</li>
                  <li>· At least {minHoldout} corpus judgements frozen at scoring time.</li>
                  <li>· Hold-out pass rate at or above the target, and a measured baseline to beat.</li>
                </ul>
              </div>
            </>
          )}

          {/* ---------------------------------------------------------------- */}
          {/* Corpus                                                           */}
          {/* ---------------------------------------------------------------- */}
          {tab === 'corpus' && (
            <>
              <div className="grid grid-cols-4 gap-1.5">
                <Stat label="scored" value={report?.scored_cases ?? 0} />
                <Stat label="passed" value={report?.passed_cases ?? 0} tone="text-emerald-300" />
                <Stat label="failed" value={report?.failed_cases ?? 0} tone="text-rose-300" />
                <Stat label="waived" value={report?.waived_cases ?? 0} tone="text-gray-400" />
              </div>

              {/* Lock the step you just inspected */}
              <div className="rounded-xl border border-gray-800 bg-gray-900/40 p-3 space-y-2">
                <div className="flex items-center space-x-2 text-[11px] font-semibold text-gray-300">
                  <Lock className="h-3.5 w-3.5 text-indigo-400" />
                  <span>Lock the step you just inspected</span>
                </div>
                {lastStep && lastStep.action !== null ? (
                  <>
                    <p className="text-[10px] text-gray-500 font-mono leading-relaxed">
                      step {lastStep.step ?? '—'} · obs[{lastStep.obs.length}] · policy chose{' '}
                      <span className="text-indigo-300">{actionName(lastStep.action, actionLabels)}</span>
                    </p>
                    <input
                      value={lockRationale}
                      onChange={(e) => setLockRationale(e.target.value)}
                      placeholder="Why (optional)"
                      className="w-full bg-gray-950 border border-gray-800 rounded-lg px-2 py-1.5 text-[11px] text-gray-200 placeholder-gray-600 focus:outline-none focus:border-indigo-700"
                    />
                    <div className="flex space-x-1.5">
                      <button
                        onClick={() => void handleLock('expected')}
                        disabled={lockBusy}
                        className="flex-1 flex items-center justify-center space-x-1 px-2 py-1.5 rounded-lg bg-emerald-900/60 border border-emerald-700/60 text-emerald-200 text-[10px] font-semibold hover:bg-emerald-900 disabled:opacity-50"
                      >
                        <CheckCircle2 className="h-3 w-3" />
                        <span>This is correct</span>
                      </button>
                      <button
                        onClick={() => void handleLock('forbidden')}
                        disabled={lockBusy}
                        className="flex-1 flex items-center justify-center space-x-1 px-2 py-1.5 rounded-lg bg-rose-900/50 border border-rose-800/60 text-rose-200 text-[10px] font-semibold hover:bg-rose-900/70 disabled:opacity-50"
                      >
                        <Ban className="h-3 w-3" />
                        <span>Never do this</span>
                      </button>
                    </div>
                    {lockNote && <p className="text-[10px] text-gray-400">{lockNote}</p>}
                  </>
                ) : (
                  <p className="text-[10px] text-gray-500 leading-snug">
                    Play the arena in <span className="font-mono">Interactive</span> mode, then reopen this
                    drawer. The observation you just saw becomes the assertion.
                  </p>
                )}
              </div>

              {/* Passive harvest */}
              <div className="rounded-xl border border-gray-800 bg-gray-900/40 p-3 space-y-2">
                <div className="flex items-center space-x-2 text-[11px] font-semibold text-gray-300">
                  <Upload className="h-3.5 w-3.5 text-cyan-400" />
                  <span>Import a frontier interaction log</span>
                </div>
                <p className="text-[10px] text-gray-500 leading-snug">
                  CSV with the proposed action, a human verdict and (ideally) the corrected action. Rejections
                  become corpus assertions; uploaded files also land in the dataset store.
                </p>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".csv,.xlsx,.json"
                  disabled={harvestBusy || !conversationId}
                  onChange={(e) => void handleHarvestFile(e.target.files?.[0] ?? null)}
                  className="block w-full text-[10px] text-gray-400 file:mr-2 file:py-1 file:px-2 file:rounded-lg file:border-0 file:bg-gray-800 file:text-gray-200 hover:file:bg-gray-700"
                />
                {harvestBusy && <p className="text-[10px] text-cyan-300">Profiling and harvesting…</p>}
                {harvest && (
                  <div className="text-[10px] font-mono space-y-0.5">
                    {harvest.detected && harvest.harvested ? (
                      <>
                        <p className="text-gray-400">
                          matched {harvest.detection?.usable_rows ?? 0} usable rows · registered{' '}
                          <span className="text-emerald-300">{harvest.harvested.registered}</span>
                          {harvest.harvested.soft_negatives > 0 && (
                            <span className="text-gray-500"> ({harvest.harvested.soft_negatives} soft)</span>
                          )}
                        </p>
                        {harvest.harvested.unmatched_rows > 0 && (
                          <p className="text-amber-300/80">
                            {harvest.harvested.unmatched_rows} rows had no matching record — skipped, not
                            invented.
                          </p>
                        )}
                        {harvest.harvested.skipped_no_env && (
                          <p className="text-amber-300/80">
                            No environment loaded in the server: retry after a training run.
                          </p>
                        )}
                      </>
                    ) : (
                      <p className="text-amber-300/80">
                        Not recognised as an interaction log. Needs an action column plus an accepted/rejected
                        verdict.
                      </p>
                    )}
                  </div>
                )}
              </div>

              {waiveTarget && (
                <div className="rounded-xl border border-amber-800/60 bg-amber-950/30 p-2 space-y-1.5">
                  <p className="text-[10px] text-amber-200 font-mono">
                    Waive {waiveTarget} — it stops being scored but stays on record.
                  </p>
                  <div className="flex space-x-1.5">
                    <input
                      value={waiveReason}
                      onChange={(e) => setWaiveReason(e.target.value)}
                      placeholder="Reason"
                      className="flex-1 bg-gray-950 border border-gray-800 rounded-lg px-2 py-1 text-[10px] text-gray-200 placeholder-gray-600"
                    />
                    <button
                      onClick={() => void handleWaive()}
                      disabled={waiveBusy}
                      className="px-2 py-1 rounded-lg bg-amber-800/70 border border-amber-700 text-amber-100 text-[10px] font-semibold disabled:opacity-50"
                    >
                      Waive
                    </button>
                    <button
                      onClick={() => {
                        setWaiveTarget(null)
                        setWaiveReason('')
                      }}
                      className="px-2 py-1 rounded-lg bg-gray-800 border border-gray-700 text-gray-300 text-[10px]"
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              )}

              <div className="space-y-1.5">
                <div className="flex items-center justify-between">
                  <span className="text-[10px] uppercase tracking-wider text-gray-500 font-bold">
                    Assertions
                  </span>
                  <span className="text-[10px] text-gray-600 font-mono">{cases.length} total</span>
                </div>
                {cases.length === 0 ? (
                  <p className="text-[11px] text-gray-500 leading-snug border border-dashed border-gray-800 rounded-xl p-3">
                    The vault is empty. It seeds itself from held-out rows at the next training round; a
                    conversation trained before this feature needs one retrain or resume.
                  </p>
                ) : (
                  cases.map((c) => (
                    <CorpusRow
                      key={c.case_id}
                      item={c}
                      failing={failingIds.has(c.case_id)}
                      labels={actionLabels.length ? actionLabels : c.action_labels}
                      onWaive={() => setWaiveTarget(c.case_id)}
                    />
                  ))
                )}
              </div>
            </>
          )}

          {/* ---------------------------------------------------------------- */}
          {/* Gate                                                             */}
          {/* ---------------------------------------------------------------- */}
          {tab === 'gate' && (
            <>
              {mutationPending && (
                <div className="rounded-xl border border-amber-800/60 bg-amber-950/30 p-2 text-[11px] text-amber-200">
                  A {mutationPending} change is waiting: the next training round judges it.
                </div>
              )}

              {gateHistory.length === 0 && !newestGate && (
                <p className="text-[11px] text-gray-500 leading-snug border border-dashed border-gray-800 rounded-xl p-3">
                  No gate decisions yet. The verifier gate judges reward, environment and inference changes
                  once a policy exists — before that there is nothing to compare against.
                </p>
              )}

              {gateHistory.length === 0 && newestGate && (
                <GateCard
                  record={{
                    at: '',
                    round: session?.current_round ?? 0,
                    origin: 'last',
                    kind: newestGate.mutation_kind ?? null,
                    passed: newestGate.passed,
                    category: newestGate.triage?.category ?? null,
                    reason: newestGate.reason ?? '',
                    advice: newestGate.triage?.advice ?? '',
                  }}
                />
              )}

              {gateHistory.length > 0 && (
                <div className="space-y-1.5">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] uppercase tracking-wider text-gray-500 font-bold flex items-center space-x-1">
                      <History className="h-3 w-3" />
                      <span>Decisions, oldest first</span>
                    </span>
                    <span className="text-[10px] text-gray-600 font-mono">{gateHistory.length}</span>
                  </div>
                  {[...gateHistory].reverse().map((record, idx) => (
                    <GateCard key={`${record.round}_${record.at}_${idx}`} record={record} />
                  ))}
                </div>
              )}

              <div className="rounded-xl border border-gray-800 bg-gray-900/30 p-3">
                <div className="text-[10px] uppercase tracking-wider text-gray-500 font-bold mb-1 flex items-center space-x-1">
                  <ListChecks className="h-3 w-3" />
                  <span>Triage vocabulary</span>
                </div>
                <ul className="text-[10px] text-gray-400 space-y-0.5 font-mono">
                  <li>compiler_mismatch — the term never fired</li>
                  <li>cross_term_interference — it over-dominated an approved case</li>
                  <li>rl_variance — signal fine, budget or entropy</li>
                  <li>unclassified — inspect the evidence</li>
                </ul>
              </div>
            </>
          )}
        </div>
      </aside>
    </div>
  )
}

const Row: React.FC<{ label: string; value: string; title?: string }> = ({ label, value, title }) => (
  <div className="flex items-center justify-between" title={title}>
    <span className="text-gray-500">{label}</span>
    <span className="text-gray-300 truncate pl-2">{value}</span>
  </div>
)

const Stat: React.FC<{ label: string; value: number; tone?: string }> = ({ label, value, tone }) => (
  <div className="rounded-lg border border-gray-800 bg-gray-900/40 px-2 py-1.5 text-center">
    <div className={`text-sm font-black font-mono ${tone ?? 'text-gray-200'}`}>{value}</div>
    <div className="text-[9px] uppercase tracking-wider text-gray-600">{label}</div>
  </div>
)

const CorpusRow: React.FC<{
  item: EvalCorpusCase
  failing: boolean
  labels: string[]
  onWaive: () => void
}> = ({ item, failing, labels, onWaive }) => {
  const expectation = item.expected_actions.length
    ? `expect ${item.expected_actions.map((a) => actionName(a, labels)).join(', ')}`
    : `forbid ${item.forbidden_actions.map((a) => actionName(a, labels)).join(', ')}`
  const status = item.waived ? 'waived' : failing ? 'failed' : 'passed'
  const tone =
    status === 'waived'
      ? 'border-gray-800 text-gray-500'
      : status === 'failed'
        ? 'border-rose-900/50 text-rose-200'
        : 'border-gray-800 text-gray-300'
  return (
    <div className={`rounded-lg border px-2 py-1.5 ${tone} bg-gray-900/30`}>
      <div className="flex items-center justify-between space-x-2">
        <div className="min-w-0">
          <div className="flex items-center space-x-1.5">
            <span
              className={`h-1.5 w-1.5 rounded-full ${
                status === 'waived'
                  ? 'bg-gray-600'
                  : status === 'failed'
                    ? 'bg-rose-400'
                    : 'bg-emerald-400'
              }`}
            />
            <span className="text-[10px] font-mono text-gray-500">{item.case_id}</span>
            <span className="text-[9px] px-1 rounded bg-gray-800 text-gray-400 font-mono">
              {item.source}
            </span>
          </div>
          <div className="text-[10px] font-mono text-gray-300 truncate">{expectation}</div>
          {item.rationale && (
            <div className="text-[10px] text-gray-500 truncate" title={item.rationale}>
              {item.rationale}
            </div>
          )}
        </div>
        {!item.waived && (
          <button
            onClick={onWaive}
            className="shrink-0 px-1.5 py-0.5 rounded border border-gray-700 text-[9px] text-gray-400 hover:text-amber-300 hover:border-amber-800"
            title="Stop scoring this assertion"
          >
            waive
          </button>
        )}
      </div>
    </div>
  )
}

const GateCard: React.FC<{ record: GateRecord }> = ({ record }) => (
  <div
    className={`rounded-lg border px-2 py-1.5 ${
      record.passed ? 'border-emerald-900/50 bg-emerald-950/20' : 'border-rose-900/50 bg-rose-950/20'
    }`}
  >
    <div className="flex items-center justify-between">
      <div className="flex items-center space-x-1.5">
        {record.passed ? (
          <CheckCircle2 className="h-3 w-3 text-emerald-400" />
        ) : (
          <XCircle className="h-3 w-3 text-rose-400" />
        )}
        <span className="text-[10px] font-mono text-gray-400">
          round {record.round} · {record.kind ?? 'change'}
        </span>
      </div>
      <span className="text-[9px] font-mono text-gray-600">{record.origin}</span>
    </div>
    {record.category && (
      <div className="mt-1 text-[10px] font-mono text-amber-300">{record.category}</div>
    )}
    <p className="text-[10px] text-gray-400 leading-snug mt-0.5">{record.reason}</p>
    {record.advice && (
      <p className="text-[10px] text-gray-500 leading-snug mt-1 flex items-start space-x-1">
        <ChevronRight className="h-3 w-3 mt-0.5 shrink-0 text-gray-600" />
        <span>{record.advice}</span>
      </p>
    )}
  </div>
)
