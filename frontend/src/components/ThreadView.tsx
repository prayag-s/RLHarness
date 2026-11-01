import React, { useEffect, useMemo, useRef } from 'react'
import {
  AlertTriangle,
  ArrowDown,
  Bot,
  CheckCircle2,
  ChevronDown,
  Cog,
  GraduationCap,
  Layers,
  Play,
  ShieldCheck,
  Sparkles,
  User,
  XCircle,
} from 'lucide-react'
import { CoachAction, CoachItem, ThreadEntry, TurnStatus } from '../types'
import { CoachCard } from './CoachCard'

interface ThreadViewProps {
  entries: ThreadEntry[]
  turn: TurnStatus | null
  conversationTitle?: string
  onCoachAction: (action: CoachAction, item: CoachItem) => void
  onAcknowledge: (item: CoachItem) => void
  onOpenTrust: () => void
  onSelectRound: (round: number) => void
  /** Rolls the policy back to a round; absent when there is nothing trained yet. */
  onRollbackRound?: (round: number) => void
  canRollback?: boolean
  /** Entry to highlight and scroll to, e.g. after clicking a scoreboard metric. */
  focusedSeq?: number | null
}

const ACTOR_LABEL: Record<string, string> = {
  user: 'You',
  studio: 'Studio',
  trainer: 'Trainer',
  coach: 'Coach',
  gatekeeper: 'Gatekeeper',
  system: 'System',
}

const STATUS_TONE: Record<string, string> = {
  success: 'border-emerald-800/60 bg-emerald-950/20',
  warning: 'border-amber-800/60 bg-amber-950/20',
  blocked: 'border-rose-800/70 bg-rose-950/25',
  running: 'border-indigo-700/60 bg-indigo-950/25',
  pending: 'border-gray-700 bg-gray-900/50',
  info: 'border-gray-800 bg-gray-900/40',
}

function clock(iso: string): string {
  if (!iso) return ''
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return ''
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

/** Consecutive progress stages collapse into one row: a turn may report ten of them. */
type RenderItem =
  | { type: 'entry'; entry: ThreadEntry }
  | { type: 'stage-group'; key: string; entries: ThreadEntry[] }

function groupTurn(entries: ThreadEntry[]): RenderItem[] {
  const items: RenderItem[] = []
  let stages: ThreadEntry[] = []
  const flush = () => {
    if (stages.length) {
      items.push({ type: 'stage-group', key: `stages_${stages[0].seq}`, entries: stages })
      stages = []
    }
  }
  for (const entry of entries) {
    if (entry.kind === 'stage') {
      stages.push(entry)
      continue
    }
    flush()
    items.push({ type: 'entry', entry })
  }
  flush()
  return items
}

const ActorIcon: React.FC<{ actor: string }> = ({ actor }) => {
  switch (actor) {
    case 'user':
      return <User className="h-3.5 w-3.5" />
    case 'trainer':
      return <GraduationCap className="h-3.5 w-3.5" />
    case 'coach':
      return <Sparkles className="h-3.5 w-3.5" />
    case 'gatekeeper':
      return <ShieldCheck className="h-3.5 w-3.5" />
    case 'system':
      return <Cog className="h-3.5 w-3.5" />
    default:
      return <Bot className="h-3.5 w-3.5" />
  }
}

const StageGroup: React.FC<{ entries: ThreadEntry[]; isLive: boolean }> = ({ entries, isLive }) => {
  const latest = entries[entries.length - 1]
  return (
    <div className="rounded-lg border border-indigo-900/40 bg-indigo-950/15 px-3 py-2">
      <div className="flex items-center justify-between gap-2">
        <span className="flex items-center gap-1.5 text-[11px] text-indigo-300 min-w-0">
          <Layers className="h-3.5 w-3.5 shrink-0" />
          <span className="truncate">{latest.title}</span>
          {isLive && (
            <span className="relative flex h-1.5 w-1.5 shrink-0">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-indigo-400 opacity-75" />
              <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-indigo-500" />
            </span>
          )}
        </span>
        <span className="text-[10px] font-mono text-gray-500 shrink-0">
          {entries.length} step{entries.length === 1 ? '' : 's'}
        </span>
      </div>
      {latest.body && <p className="mt-0.5 text-[11px] text-gray-400 leading-snug">{latest.body}</p>}
      {entries.length > 1 && (
        <details className="group mt-1">
          <summary className="flex cursor-pointer list-none items-center gap-1 text-[10px] text-gray-500 hover:text-gray-300">
            <ChevronDown className="h-3 w-3 transition group-open:rotate-180" />
            Earlier steps
          </summary>
          <ol className="mt-1 space-y-1 border-l border-gray-800 pl-2">
            {entries.slice(0, -1).map((e) => (
              <li key={e.seq} className="text-[10px] text-gray-500">
                <span className="font-mono text-gray-600">{clock(e.at)} </span>
                {e.title}
                {e.body ? <span className="text-gray-600"> — {e.body}</span> : null}
              </li>
            ))}
          </ol>
        </details>
      )}
    </div>
  )
}

const EntryCard: React.FC<{
  entry: ThreadEntry
  onCoachAction: (action: CoachAction, item: CoachItem) => void
  onAcknowledge: (item: CoachItem) => void
  onOpenTrust: () => void
  onSelectRound: (round: number) => void
  onRollbackRound?: (round: number) => void
  canRollback: boolean
  focused: boolean
}> = ({
  entry,
  onCoachAction,
  onAcknowledge,
  onOpenTrust,
  onSelectRound,
  onRollbackRound,
  canRollback,
  focused,
}) => {
  const isUser = entry.actor === 'user'
  const tone = STATUS_TONE[entry.status] || STATUS_TONE.info

  if (entry.kind === 'finding') {
    const item: CoachItem | undefined = entry.data?.item
    if (item) {
      return (
        <div className="pl-7" data-seq={entry.seq}>
          <div className={focused ? 'rounded-xl ring-2 ring-cyan-500/60' : ''}>
            <CoachCard
              report={{
                stage: (item as any).stage,
                status:
                  item.severity === 'blocker'
                    ? 'blocked'
                    : item.severity === 'warning'
                      ? 'attention'
                      : 'ok',
                items: [item],
                conversation_id: null,
                evaluated_at: entry.at,
                llm_used: false,
              }}
              onAction={onCoachAction}
              onAcknowledge={onAcknowledge}
              maxItems={1}
            />
          </div>
        </div>
      )
    }
  }

  return (
    <div
      data-seq={entry.seq}
      className={`rounded-xl border px-3 py-2.5 transition ${tone} ${
        isUser ? 'border-indigo-700/60 bg-indigo-950/25' : ''
      } ${focused ? 'ring-2 ring-cyan-500/60' : ''}`}
    >
      <div className="flex items-center justify-between gap-2 mb-1">
        <span
          className={`flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wide ${
            isUser ? 'text-indigo-300' : 'text-gray-400'
          }`}
        >
          <ActorIcon actor={entry.actor} />
          {ACTOR_LABEL[entry.actor] || entry.actor}
          {entry.status === 'success' && entry.actor === 'trainer' && (
            <CheckCircle2 className="h-3 w-3 text-emerald-400" />
          )}
          {entry.status === 'warning' && <AlertTriangle className="h-3 w-3 text-amber-400" />}
          {entry.status === 'blocked' && <XCircle className="h-3 w-3 text-rose-400" />}
        </span>
        <span className="text-[10px] text-gray-500 font-mono shrink-0">{clock(entry.at)}</span>
      </div>

      <p className="text-xs font-medium text-gray-100 leading-snug">{entry.title}</p>
      {entry.body && (
        <p className="mt-1 text-[11.5px] text-gray-300 leading-relaxed whitespace-pre-wrap break-words">
          {entry.body}
        </p>
      )}

      <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
        {entry.anchor?.round !== undefined && (
          <button
            onClick={() => onSelectRound(Number(entry.anchor?.round))}
            className="flex items-center gap-1 rounded-md border border-gray-700 bg-gray-950/70 px-2 py-0.5 text-[10px] text-gray-300 hover:border-gray-500 transition"
          >
            <Play className="h-2.5 w-2.5" /> Round {entry.anchor.round}
          </button>
        )}
        {entry.actor === 'trainer' &&
          (entry.kind === 'round' || entry.kind === 'tune') &&
          entry.anchor?.round !== undefined &&
          onRollbackRound &&
          canRollback && (
            <button
              onClick={() => onRollbackRound(Number(entry.anchor?.round))}
              className="rounded-md border border-gray-700 bg-gray-950/70 px-2 py-0.5 text-[10px] text-gray-400 hover:border-amber-600 hover:text-amber-300 transition"
              title="Restore the policy and environment code from this round"
            >
              Roll back here
            </button>
          )}
        {(entry.actions || []).map((action, idx) =>
          action.kind === 'open_trust' ? (
            <button
              key={`${entry.seq}-act-${idx}`}
              onClick={onOpenTrust}
              className="rounded-md border border-gray-700 bg-gray-950/70 px-2 py-0.5 text-[10px] text-gray-300 hover:border-gray-500 transition"
            >
              {action.label}
            </button>
          ) : null,
        )}
      </div>
    </div>
  )
}

export const ThreadView: React.FC<ThreadViewProps> = ({
  entries,
  turn,
  conversationTitle,
  onCoachAction,
  onAcknowledge,
  onOpenTrust,
  onSelectRound,
  onRollbackRound,
  canRollback = false,
  focusedSeq,
}) => {
  const bottomRef = useRef<HTMLDivElement>(null)
  const isRunning = turn?.state === 'running'

  const turns = useMemo(() => {
    const groups: { turnId: string; entries: ThreadEntry[] }[] = []
    for (const entry of entries) {
      const last = groups[groups.length - 1]
      if (last && last.turnId === entry.turn_id) {
        last.entries.push(entry)
      } else {
        groups.push({ turnId: entry.turn_id, entries: [entry] })
      }
    }
    return groups
  }, [entries])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [entries.length, isRunning])

  useEffect(() => {
    if (focusedSeq == null) return
    const node = document.querySelector(`[data-seq="${focusedSeq}"]`)
    node?.scrollIntoView({ behavior: 'smooth', block: 'center' })
  }, [focusedSeq])

  if (!entries.length) {
    return (
      <div className="flex-1 flex items-center justify-center p-8">
        <div className="max-w-md text-center space-y-3">
          <div className="mx-auto h-10 w-10 rounded-xl bg-gradient-to-tr from-cyan-500 via-indigo-600 to-fuchsia-500 flex items-center justify-center">
            <Sparkles className="h-5 w-5 text-white" />
          </div>
          <p className="text-sm font-semibold text-gray-200">
            {conversationTitle ? conversationTitle : 'Describe the agent you want'}
          </p>
          <p className="text-xs text-gray-500 leading-relaxed">
            Everything happens here, in order: your instruction, the studio's plan, the arena, the
            coach, every training round and every tuning decision.
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
      {turns.map((group, groupIdx) => {
        const items = groupTurn(group.entries)
        const last = group.entries[group.entries.length - 1]
        const live = isRunning && groupIdx === turns.length - 1
        return (
          <section key={group.turnId} className="space-y-2">
            {groupIdx > 0 && (
              <div className="flex items-center gap-2 pt-1">
                <div className="h-px flex-1 bg-gray-800" />
                <span className="text-[10px] uppercase tracking-wider text-gray-600">
                  {group.entries[0].actor === 'user' ? 'You' : ACTOR_LABEL[group.entries[0].actor]}
                </span>
                <div className="h-px flex-1 bg-gray-800" />
              </div>
            )}
            {items.map((item) =>
              item.type === 'stage-group' ? (
                <StageGroup
                  key={item.key}
                  entries={item.entries}
                  isLive={live && item === items[items.length - 1]}
                />
              ) : (
                <EntryCard
                  key={item.entry.seq}
                  entry={item.entry}
                  onCoachAction={onCoachAction}
                  onAcknowledge={onAcknowledge}
                  onOpenTrust={onOpenTrust}
                  onSelectRound={onSelectRound}
                  onRollbackRound={onRollbackRound}
                  canRollback={canRollback}
                  focused={focusedSeq === item.entry.seq}
                />
              ),
            )}
            {live && last.kind !== 'stage' && (
              <div className="flex items-center gap-2 pl-7 text-[11px] text-indigo-300">
                <span className="relative flex h-1.5 w-1.5">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-indigo-400 opacity-75" />
                  <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-indigo-500" />
                </span>
                {ACTOR_LABEL[turn?.actor || 'studio']} is working — you'll get the prompt back when
                this turn ends
              </div>
            )}
          </section>
        )
      })}
      <div ref={bottomRef} />
      {turn?.state === 'decision_required' && (
        <div className="flex items-center gap-1.5 text-[11px] text-amber-300">
          <ArrowDown className="h-3 w-3" /> Your turn — answer below.
        </div>
      )}
    </div>
  )
}
