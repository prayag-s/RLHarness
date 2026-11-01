import React, { useEffect, useRef, useState } from 'react'
import {
  AlertCircle,
  Clock,
  Database,
  FileSpreadsheet,
  Lock,
  Play,
  Send,
  SlidersHorizontal,
  Square,
  Upload,
  X,
} from 'lucide-react'
import { IngestedDatasetSummary, TrainingParams, TurnStatus } from '../types'

interface TurnComposerProps {
  turn: TurnStatus | null
  isLLMVerified: boolean
  onOpenSettings: () => void
  onSend: (message: string) => void
  onStop: () => void
  isStopping?: boolean
  /** Text held for the next turn because the other party currently owns this one. */
  queued: string | null
  onQueue: (message: string) => void
  onCancelQueued: () => void
  attachedDataset: IngestedDatasetSummary | null
  onAttachFile: (file: File) => void
  onOpenSQLModal: () => void
  onRemoveDataset: () => void
  trainingParams: TrainingParams
  onUpdateTrainingParams: (params: TrainingParams) => void
  quickActions: string[]
  onQuickAction: (label: string) => void
  inputRef?: React.RefObject<HTMLInputElement>
  hasPolicy?: boolean
}

const STATE_LABEL: Record<string, string> = {
  idle: 'Your turn',
  running: 'Studio has the turn',
  decision_required: 'Your turn — a decision is waiting',
  blocked: 'Blocked — the coach needs a fix',
}

export const TurnComposer: React.FC<TurnComposerProps> = ({
  turn,
  isLLMVerified,
  onOpenSettings,
  onSend,
  onStop,
  isStopping = false,
  queued,
  onQueue,
  onCancelQueued,
  attachedDataset,
  onAttachFile,
  onOpenSQLModal,
  onRemoveDataset,
  trainingParams,
  onUpdateTrainingParams,
  quickActions,
  onQuickAction,
  inputRef,
  hasPolicy = false,
}) => {
  const [text, setText] = useState('')
  const [showParams, setShowParams] = useState(false)
  const [isDragging, setIsDragging] = useState(false)
  const localRef = useRef<HTMLInputElement>(null)
  const effectiveRef = (inputRef || localRef) as React.RefObject<HTMLInputElement>
  const fileRef = useRef<HTMLInputElement>(null)

  const state = turn?.state || 'idle'
  const isRunning = state === 'running'

  useEffect(() => {
    if (!isRunning) {
      effectiveRef.current?.focus()
    }
  }, [isRunning, effectiveRef])

  const submit = (e?: React.FormEvent) => {
    e?.preventDefault()
    const message = text.trim()
    if (!message) return
    if (!isLLMVerified) {
      onOpenSettings()
      return
    }
    if (isRunning) {
      onQueue(message)
    } else {
      onSend(message)
    }
    setText('')
  }

  const onKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Escape' && isRunning && !isStopping) {
      e.preventDefault()
      onStop()
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    const file = e.dataTransfer.files?.[0]
    if (file) onAttachFile(file)
  }

  return (
    <div className="border-t border-gray-800 bg-gray-950/95 px-4 py-3 space-y-2">
      {!isLLMVerified && (
        <button
          onClick={onOpenSettings}
          className="w-full flex items-center gap-2 rounded-lg border border-amber-700/60 bg-amber-950/70 px-3 py-2 text-left text-[11px] text-amber-100 hover:bg-amber-900/70 transition"
        >
          <AlertCircle className="h-3.5 w-3.5 text-amber-400 shrink-0" />
          <span>
            <strong className="text-amber-300">Explore mode — no model connected.</strong> Connect a
            provider to synthesize, chat or train.
          </span>
        </button>
      )}

      {queued && (
        <div className="flex items-center justify-between gap-2 rounded-lg border border-amber-800/60 bg-amber-950/30 px-3 py-1.5 text-[11px] text-amber-200">
          <span className="flex items-center gap-1.5 min-w-0">
            <Clock className="h-3 w-3 shrink-0" />
            <span className="truncate">Queued for the next turn: “{queued}”</span>
          </span>
          <button
            onClick={onCancelQueued}
            className="p-0.5 text-amber-400 hover:text-amber-200 transition shrink-0"
            title="Discard the queued message"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        </div>
      )}

      {queued && isRunning && (
        <div className="flex items-center justify-end">
          <button
            onClick={() => {
              onCancelQueued()
              onStop()
            }}
            className="flex items-center gap-1 text-[10px] text-gray-400 hover:text-rose-300 transition"
            title="Interrupt this turn so the queued message can be sent now"
          >
            <Square className="h-2.5 w-2.5" /> Interrupt instead
          </button>
        </div>
      )}

      {quickActions.length > 0 && !isRunning && (
        <div className="flex items-center gap-1.5 overflow-x-auto pb-0.5">
          {quickActions.map((chip, idx) => (
            <button
              key={`chip-${idx}`}
              onClick={() => onQuickAction(chip)}
              className={`shrink-0 rounded-full px-2.5 py-1 text-[11px] font-medium transition ${
                chip.includes('Lock') || chip.includes('Start Training')
                  ? 'bg-emerald-600 text-white hover:bg-emerald-500'
                  : 'border border-gray-700 bg-gray-900 text-gray-300 hover:border-gray-500 hover:text-white'
              }`}
            >
              {chip}
            </button>
          ))}
        </div>
      )}

      <div className="flex items-center justify-between text-[11px]">
        <span className="flex items-center gap-1.5 text-gray-400">
          {attachedDataset ? (
            <>
              <FileSpreadsheet className="h-3 w-3 text-indigo-400" />
              <span className="text-indigo-200 truncate max-w-[220px]">{attachedDataset.name}</span>
              <span className="text-gray-500 font-mono text-[10px]">
                {attachedDataset.num_rows} rows
              </span>
              <button
                onClick={onRemoveDataset}
                className="text-gray-500 hover:text-rose-400 transition"
                title="Detach the dataset"
              >
                <X className="h-3 w-3" />
              </button>
            </>
          ) : (
            <span className="text-gray-500">No dataset attached — optional</span>
          )}
        </span>
        <span
          className={`font-mono text-[10px] ${
            isRunning
              ? 'text-indigo-300'
              : state === 'blocked'
                ? 'text-rose-300'
                : state === 'decision_required'
                  ? 'text-amber-300'
                  : 'text-gray-500'
          }`}
        >
          {STATE_LABEL[state]}
        </span>
      </div>

      <form onSubmit={submit} className="space-y-1.5">
        <div
          onDragOver={(e) => {
            e.preventDefault()
            setIsDragging(true)
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          className={`flex items-center gap-1.5 rounded-xl border px-1.5 py-1.5 transition ${
            isDragging
              ? 'border-indigo-400 bg-indigo-950/40'
              : 'border-gray-700 bg-gray-900 focus-within:border-indigo-500'
          }`}
        >
          <input
            ref={fileRef}
            type="file"
            className="hidden"
            accept=".csv,.xlsx,.pdf,.docx,.sqlite,.json,.txt"
            onChange={(e) => {
              const file = e.target.files?.[0]
              if (file) onAttachFile(file)
              e.target.value = ''
            }}
          />
          <button
            type="button"
            onClick={() => fileRef.current?.click()}
            className="p-1.5 rounded-lg text-gray-400 hover:text-indigo-300 hover:bg-gray-800 transition"
            title="Attach a CSV, XLSX, PDF, DOCX, JSON or SQLite file"
          >
            <Upload className="h-4 w-4" />
          </button>
          <button
            type="button"
            onClick={onOpenSQLModal}
            className="p-1.5 rounded-lg text-gray-400 hover:text-cyan-300 hover:bg-gray-800 transition"
            title="Connect SQLite, PostgreSQL or MySQL"
          >
            <Database className="h-4 w-4" />
          </button>
          <button
            type="button"
            onClick={() => setShowParams((v) => !v)}
            className={`p-1.5 rounded-lg transition ${
              showParams ? 'text-indigo-300 bg-gray-800' : 'text-gray-400 hover:text-indigo-300 hover:bg-gray-800'
            }`}
            title="Training budget"
          >
            <SlidersHorizontal className="h-4 w-4" />
          </button>

          <input
            ref={effectiveRef}
            type="text"
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder={
              isRunning
                ? 'Studio has the turn — type now and it will be queued for the next turn…'
                : hasPolicy
                  ? "Instruct the studio — e.g. 'train 5 more rounds', 'make it more aggressive'…"
                  : "Describe the agent you want — or attach a dataset…"
            }
            className="flex-1 bg-transparent px-1 py-1.5 text-xs text-gray-100 placeholder-gray-500 focus:outline-none"
          />

          {isRunning ? (
            <button
              type="button"
              onClick={onStop}
              disabled={isStopping}
              className="flex items-center gap-1 rounded-lg border border-rose-800/70 bg-rose-950/50 px-2.5 py-1.5 text-[11px] font-semibold text-rose-200 hover:bg-rose-900/60 transition disabled:opacity-50"
              title="Stop this turn (Esc)"
            >
              <Square className="h-3 w-3" />
              {isStopping ? 'Stopping…' : 'Stop'}
            </button>
          ) : null}

          <button
            type="submit"
            disabled={!text.trim()}
            className={`flex items-center gap-1 rounded-lg px-2.5 py-1.5 text-[11px] font-semibold text-white transition disabled:opacity-40 ${
              isRunning ? 'bg-amber-600 hover:bg-amber-500' : 'bg-indigo-600 hover:bg-indigo-500'
            }`}
            title={isRunning ? 'Queue this message for the next turn' : 'Send (Enter)'}
          >
            {isRunning ? <Clock className="h-3.5 w-3.5" /> : <Send className="h-3.5 w-3.5" />}
            {isRunning ? 'Queue' : 'Send'}
          </button>
        </div>
      </form>

      {showParams && (
        <div className="rounded-xl border border-gray-800 bg-gray-900/80 p-3 space-y-2.5">
          <div className="grid grid-cols-3 gap-2 text-[11px]">
            <label className="space-y-1">
              <span className="block text-[10px] text-gray-400">Rounds (1-100)</span>
              <input
                type="number"
                min={1}
                max={100}
                value={trainingParams.max_auto_tune_rounds}
                onChange={(e) =>
                  onUpdateTrainingParams({
                    ...trainingParams,
                    max_auto_tune_rounds: Math.max(1, Math.min(100, parseInt(e.target.value) || 1)),
                  })
                }
                className="w-full rounded border border-gray-700 bg-gray-900 px-2 py-1 text-center font-mono text-gray-200 focus:border-indigo-500 focus:outline-none"
              />
            </label>
            <label className="space-y-1">
              <span className="block text-[10px] text-gray-400">Episodes / round</span>
              <select
                value={trainingParams.episodes_per_round}
                onChange={(e) =>
                  onUpdateTrainingParams({
                    ...trainingParams,
                    episodes_per_round: parseInt(e.target.value),
                  })
                }
                className="w-full rounded border border-gray-700 bg-gray-900 px-1.5 py-1 text-center font-mono text-gray-200 focus:border-indigo-500 focus:outline-none"
              >
                {[50, 100, 200, 250, 500, 1000, 2000].map((n) => (
                  <option key={n} value={n}>
                    {n}
                  </option>
                ))}
              </select>
            </label>
            <label className="space-y-1">
              <span className="block text-[10px] text-gray-400">Target (%)</span>
              <input
                type="number"
                min={50}
                max={100}
                value={trainingParams.target_metric_threshold}
                onChange={(e) =>
                  onUpdateTrainingParams({
                    ...trainingParams,
                    target_metric_threshold: Math.max(
                      50,
                      Math.min(100, parseFloat(e.target.value) || 90),
                    ),
                  })
                }
                className="w-full rounded border border-gray-700 bg-gray-900 px-1.5 py-1 text-center font-mono text-gray-200 focus:border-indigo-500 focus:outline-none"
              />
            </label>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-gray-500">
              Changing the budget is a turn: say it in the thread or send it now.
            </span>
            <button
              type="button"
              onClick={() => {
                const p = trainingParams
                setText(
                  `Train for ${p.max_auto_tune_rounds} rounds with ${p.episodes_per_round} episodes per round and target ${p.target_metric_threshold}%`,
                )
                setShowParams(false)
                effectiveRef.current?.focus()
              }}
              className="flex items-center gap-1 rounded bg-indigo-600 px-2 py-1 text-[10px] font-bold text-white hover:bg-indigo-500 transition"
            >
              <Play className="h-3 w-3" /> Use these params
            </button>
          </div>
        </div>
      )}

      {hasPolicy && (
        <div className="flex items-center gap-1 text-[10px] text-gray-600">
          <Lock className="h-2.5 w-2.5" /> Arena locked — the policy keeps training from here
        </div>
      )}
    </div>
  )
}
