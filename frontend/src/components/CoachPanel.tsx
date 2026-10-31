import React, { useEffect, useMemo, useState } from 'react'
import { RefreshCw, Sparkles, X } from 'lucide-react'
import { CoachAction, CoachItem, CoachListResponse } from '../types'
import { acknowledgeGuidance, fetchGuidance } from '../services/api'
import { CoachCard } from './CoachCard'

interface CoachPanelProps {
  isOpen: boolean
  onClose: () => void
  conversationId?: string
  onAction: (action: CoachAction, item: CoachItem) => void
  onChanged?: () => void
}

const STAGE_ORDER = ['objective', 'data', 'parameters', 'preview', 'results', 'export']

const STAGE_LABEL: Record<string, string> = {
  objective: 'Objective',
  data: 'Data',
  parameters: 'Training budget',
  preview: 'Arena',
  results: 'Results',
  export: 'Export',
}

export const CoachPanel: React.FC<CoachPanelProps> = ({
  isOpen,
  onClose,
  conversationId,
  onAction,
  onChanged,
}) => {
  const [data, setData] = useState<CoachListResponse | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const load = async () => {
    if (!conversationId) {
      setData(null)
      return
    }
    setIsLoading(true)
    setError(null)
    try {
      setData(await fetchGuidance(conversationId))
    } catch (err: any) {
      setError(err?.message || 'Could not load coach findings')
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    if (isOpen) {
      void load()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen, conversationId])

  const grouped = useMemo(() => {
    const groups: Record<string, CoachItem[]> = {}
    for (const item of data?.items || []) {
      if (!groups[item.stage]) groups[item.stage] = []
      groups[item.stage].push(item)
    }
    return groups
  }, [data])

  if (!isOpen) return null

  const handleAcknowledge = async (item: CoachItem) => {
    if (!conversationId) return
    try {
      setData(await acknowledgeGuidance(conversationId, item.rule_id, item.fingerprint))
      onChanged?.()
    } catch (err: any) {
      setError(err?.message || 'Could not acknowledge the finding')
    }
  }

  const stages = STAGE_ORDER.filter((stage) => grouped[stage]?.length)

  return (
    <div className="fixed inset-0 z-40 flex justify-end">
      <div className="absolute inset-0 bg-black/50" onClick={onClose} />
      <aside className="relative z-10 h-full w-full max-w-[30rem] bg-gray-950 border-l border-gray-800 shadow-2xl flex flex-col">
        <header className="flex items-center justify-between px-4 py-3 border-b border-gray-800">
          <div className="flex items-center gap-2">
            <Sparkles className="h-4 w-4 text-indigo-400" />
            <div>
              <h2 className="text-sm font-semibold text-gray-100">Journey Coach</h2>
              <p className="text-[10px] text-gray-500">
                {data
                  ? `${data.counts.blocker || 0} blocker(s) · ${data.counts.warning || 0} warning(s) · mode ${data.blocker_mode}`
                  : 'Findings for this conversation'}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-1">
            <button
              onClick={() => void load()}
              className="p-1.5 rounded-md bg-gray-900 border border-gray-700 text-gray-300 hover:border-gray-600 transition"
              title="Refresh"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${isLoading ? 'animate-spin' : ''}`} />
            </button>
            <button
              onClick={onClose}
              className="p-1.5 rounded-md bg-gray-900 border border-gray-700 text-gray-300 hover:border-gray-600 transition"
              title="Close"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          </div>
        </header>

        <div className="flex-1 overflow-y-auto p-3 space-y-3">
          {!conversationId && (
            <p className="text-xs text-gray-500">
              Start a conversation to see coach findings for its objective, data and budget.
            </p>
          )}
          {error && (
            <p className="text-xs text-rose-300 bg-rose-950/40 border border-rose-900/60 rounded-lg px-3 py-2">
              {error}
            </p>
          )}
          {conversationId && !isLoading && !error && stages.length === 0 && (
            <p className="text-xs text-gray-500">
              Nothing outstanding. Findings appear here as the objective, data, budget, arena and
              results are evaluated.
            </p>
          )}

          {stages.map((stage) => (
            <section key={stage} className="space-y-1.5">
              <h3 className="text-[10px] font-semibold uppercase tracking-wider text-gray-500 px-1">
                {STAGE_LABEL[stage] || stage}
              </h3>
              <CoachCard
                report={{
                  stage: stage as any,
                  status: grouped[stage].some(
                    (i) => i.severity === 'blocker' && i.status !== 'acknowledged',
                  )
                    ? 'blocked'
                    : grouped[stage].some(
                          (i) => i.severity === 'warning' && i.status !== 'acknowledged',
                        )
                      ? 'attention'
                      : 'ok',
                  items: grouped[stage],
                  evaluated_at: new Date().toISOString(),
                  llm_used: false,
                }}
                onAction={onAction}
                onAcknowledge={handleAcknowledge}
                maxItems={20}
              />
            </section>
          ))}
        </div>

        <footer className="px-4 py-2 border-t border-gray-800 text-[10px] text-gray-500">
          Rules are deterministic and free. The judge is cached by content hash
          {data ? ` · ${data.llm_calls} judge call(s) this conversation` : ''}.
        </footer>
      </aside>
    </div>
  )
}
