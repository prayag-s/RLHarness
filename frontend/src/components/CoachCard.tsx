import React from 'react'
import {
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  Lightbulb,
  ShieldAlert,
  Sparkles,
} from 'lucide-react'
import { CoachAction, CoachItem, CoachReport } from '../types'

interface CoachCardProps {
  report?: CoachReport | null
  title?: string
  onAction: (action: CoachAction, item: CoachItem) => void
  onAcknowledge?: (item: CoachItem) => void
  className?: string
  maxItems?: number
}

const SEVERITY_TONE: Record<string, { box: string; icon: React.ReactNode; label: string }> = {
  blocker: {
    box: 'border-rose-800/70 bg-rose-950/30',
    icon: <ShieldAlert className="h-4 w-4 text-rose-400 shrink-0" />,
    label: 'Blocked',
  },
  warning: {
    box: 'border-amber-800/60 bg-amber-950/20',
    icon: <AlertTriangle className="h-4 w-4 text-amber-400 shrink-0" />,
    label: 'Warning',
  },
  tip: {
    box: 'border-sky-800/60 bg-sky-950/20',
    icon: <Lightbulb className="h-4 w-4 text-sky-400 shrink-0" />,
    label: 'Tip',
  },
}

const STAGE_TITLE: Record<string, string> = {
  objective: 'Before we build',
  data: 'Data check',
  parameters: 'Training budget check',
  preview: 'Arena check',
  results: 'What this result means',
  export: 'Export check',
}

export const CoachCard: React.FC<CoachCardProps> = ({
  report,
  title,
  onAction,
  onAcknowledge,
  className = '',
  maxItems = 3,
}) => {
  if (!report || !report.items?.length) return null

  const items = report.items.filter((item) => item.status !== 'resolved')
  if (!items.length) return null

  const visible = items.slice(0, maxItems)
  const hidden = items.length - visible.length
  const heading = title || (report.stage ? STAGE_TITLE[report.stage] : 'Coach findings')

  const statusPill =
    report.status === 'blocked'
      ? 'bg-rose-950/80 border-rose-700/60 text-rose-300'
      : report.status === 'attention'
        ? 'bg-amber-950/80 border-amber-700/60 text-amber-300'
        : 'bg-emerald-950/80 border-emerald-700/60 text-emerald-300'

  return (
    <div
      className={`rounded-xl border border-gray-800 bg-gray-900/70 text-gray-200 ${className}`}
    >
      <div className="flex items-center justify-between gap-2 px-3 py-2 border-b border-gray-800">
        <div className="flex items-center gap-1.5 min-w-0">
          <Sparkles className="h-3.5 w-3.5 text-indigo-400 shrink-0" />
          <span className="text-[11px] font-semibold uppercase tracking-wide text-gray-400 truncate">
            {heading}
          </span>
        </div>
        <span className={`px-1.5 py-0.5 rounded-full border text-[10px] font-mono ${statusPill}`}>
          {report.status === 'blocked' ? 'blocked' : report.status === 'attention' ? 'needs a look' : 'clear'}
        </span>
      </div>

      <ul className="divide-y divide-gray-800/70">
        {visible.map((item) => {
          const tone = SEVERITY_TONE[item.severity] || SEVERITY_TONE.tip
          const acknowledged = item.status === 'acknowledged'
          return (
            <li
              key={`${item.rule_id}:${item.fingerprint || item.title}`}
              className={`px-3 py-2.5 border-l-2 ${
                acknowledged ? 'opacity-60 border-gray-700' : tone.box
              }`}
            >
              <div className="flex items-start gap-2">
                {tone.icon}
                <div className="min-w-0 flex-1">
                  <p className="text-xs font-medium text-gray-100 leading-snug">{item.title}</p>
                  <p className="mt-0.5 text-[11px] text-gray-400 leading-snug">{item.why}</p>

                  {item.actions.length > 0 && !acknowledged && (
                    <div className="mt-2 flex flex-wrap gap-1.5">
                      {item.actions.map((action, idx) => (
                        <button
                          key={`${item.rule_id}-action-${idx}`}
                          onClick={() => onAction(action, item)}
                          className="px-2 py-1 rounded-md text-[10px] font-medium bg-indigo-950/70 border border-indigo-700/60 text-indigo-200 hover:bg-indigo-900/70 transition"
                        >
                          {action.label}
                        </button>
                      ))}
                      {item.severity !== 'blocker' && item.status === 'open' && onAcknowledge && (
                        <button
                          onClick={() => onAcknowledge(item)}
                          className="px-2 py-1 rounded-md text-[10px] font-medium bg-gray-950 border border-gray-700 text-gray-400 hover:text-gray-200 transition"
                        >
                          Acknowledge
                        </button>
                      )}
                    </div>
                  )}

                  {acknowledged && (
                    <p className="mt-1 flex items-center gap-1 text-[10px] text-gray-500">
                      <CheckCircle2 className="h-3 w-3" /> Acknowledged
                    </p>
                  )}

                  {Object.keys(item.evidence || {}).length > 0 && (
                    <details className="mt-1.5 group">
                      <summary className="flex cursor-pointer list-none items-center gap-1 text-[10px] text-gray-500 hover:text-gray-300">
                        <ChevronDown className="h-3 w-3 transition group-open:rotate-180" />
                        Evidence
                      </summary>
                      <pre className="mt-1 max-h-32 overflow-auto rounded bg-gray-950/80 p-2 text-[10px] text-gray-400 whitespace-pre-wrap break-words">
                        {JSON.stringify(item.evidence, null, 2)}
                      </pre>
                    </details>
                  )}
                </div>
              </div>
            </li>
          )
        })}
      </ul>

      {hidden > 0 && (
        <div className="px-3 py-1.5 text-[10px] text-gray-500">
          +{hidden} more finding{hidden === 1 ? '' : 's'} in the Coach panel
        </div>
      )}
    </div>
  )
}
