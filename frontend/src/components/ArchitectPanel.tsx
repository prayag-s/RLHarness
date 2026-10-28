import React, { useState, useRef, useEffect } from 'react'
import {
  MessageSquare,
  Sparkles,
  Zap,
  Upload,
  Database,
  CheckCircle2,
  AlertCircle,
  Cpu,
  RefreshCw,
  Send,
  FileSpreadsheet,
  X,
  Play,
  RotateCcw,
  Palette,
  Target,
} from 'lucide-react'
import {
  ClarifyingQuestion,
  IntakeResponse,
  AgentBlueprintSpec,
  IngestedDatasetSummary,
  ActivityTimelineItem,
} from '../types'

interface ArchitectPanelProps {
  onStartSession: (blueprint: AgentBlueprintSpec, datasetId?: string) => Promise<void>
  onChatWithArchitect: (
    prompt: string,
    answers?: Record<string, string>,
    forceSmartDefaults?: boolean,
  ) => Promise<IntakeResponse>
  onNudge: (feedbackText: string) => Promise<void>
  onOpenSQLModal: () => void
  onFileUpload: (file: File) => Promise<void>
  attachedDataset: IngestedDatasetSummary | null
  onRemoveDataset: () => void
  timeline: ActivityTimelineItem[]
  isTraining: boolean
  isSynthesizing: boolean
  activeSessionId?: string
}

export const ArchitectPanel: React.FC<ArchitectPanelProps> = ({
  onStartSession,
  onChatWithArchitect,
  onNudge,
  onOpenSQLModal,
  onFileUpload,
  attachedDataset,
  onRemoveDataset,
  timeline,
  isTraining,
  isSynthesizing,
  activeSessionId,
}) => {
  // Intake Chat State
  const [userPrompt, setUserPrompt] = useState('')
  const [isAskingArchitect, setIsAskingArchitect] = useState(false)
  const [clarifyingQuestions, setClarifyingQuestions] = useState<ClarifyingQuestion[]>([])
  const [selectedAnswers, setSelectedAnswers] = useState<Record<string, string>>({})
  const [stagedBlueprint, setStagedBlueprint] = useState<AgentBlueprintSpec | null>(null)
  const [architectMessage, setArchitectMessage] = useState<string>(
    'Describe the agent you want to synthesize or drop a dataset to get started.',
  )

  // Nudge State
  const [nudgeText, setNudgeText] = useState('')
  const [isNudging, setIsNudging] = useState(false)

  // Drag and Drop
  const [isDragging, setIsDragging] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const chatBottomRef = useRef<HTMLDivElement>(null)

  // Auto-scroll timeline
  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [timeline, clarifyingQuestions])

  const handleSendPrompt = async (e?: React.FormEvent) => {
    if (e) e.preventDefault()
    if (!userPrompt.trim() || isAskingArchitect) return

    setIsAskingArchitect(true)
    try {
      const resp = await onChatWithArchitect(userPrompt, selectedAnswers, false)
      setArchitectMessage(resp.message)
      if (resp.status === 'needs_clarification') {
        setClarifyingQuestions(resp.clarifying_questions)
        // Initialize default answers
        const initAnswers: Record<string, string> = {}
        resp.clarifying_questions.forEach((q) => {
          initAnswers[q.question_id] = q.default_option
        })
        setSelectedAnswers(initAnswers)
        setStagedBlueprint(null)
      } else if (resp.status === 'ready_to_build' && resp.blueprint) {
        setClarifyingQuestions([])
        setStagedBlueprint(resp.blueprint)
        // Auto-launch synthesis
        await onStartSession(resp.blueprint, attachedDataset?.dataset_id)
      }
    } catch (err: any) {
      setArchitectMessage(`Error: ${err.message || 'Architect intake failed'}`)
    } finally {
      setIsAskingArchitect(false)
    }
  }

  const handleOptionSelect = (questionId: string, option: string) => {
    setSelectedAnswers((prev) => ({
      ...prev,
      [questionId]: option,
    }))
  }

  const handleUseSmartDefaults = async () => {
    setIsAskingArchitect(true)
    try {
      const promptToUse = userPrompt.trim() || (attachedDataset ? `Train agent on ${attachedDataset.name}` : 'Train an unbeatable Tic-Tac-Toe agent')
      const resp = await onChatWithArchitect(promptToUse, selectedAnswers, true)
      setArchitectMessage(resp.message)
      if (resp.blueprint) {
        setClarifyingQuestions([])
        setStagedBlueprint(resp.blueprint)
        await onStartSession(resp.blueprint, attachedDataset?.dataset_id)
      }
    } catch (err: any) {
      setArchitectMessage(`Error: ${err.message || 'Failed with smart defaults'}`)
    } finally {
      setIsAskingArchitect(false)
    }
  }

  const handleConfirmAndBuild = async () => {
    setIsAskingArchitect(true)
    try {
      const resp = await onChatWithArchitect(userPrompt, selectedAnswers, false)
      if (resp.blueprint) {
        setClarifyingQuestions([])
        setStagedBlueprint(resp.blueprint)
        await onStartSession(resp.blueprint, attachedDataset?.dataset_id)
      }
    } catch (err: any) {
      setArchitectMessage(`Error: ${err.message || 'Failed to build agent'}`)
    } finally {
      setIsAskingArchitect(false)
    }
  }

  const handleNudgeSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!nudgeText.trim() || isNudging || !activeSessionId) return
    setIsNudging(true)
    try {
      await onNudge(nudgeText.trim())
      setNudgeText('')
    } catch (err: any) {
      console.error('Nudge failed', err)
    } finally {
      setIsNudging(false)
    }
  }

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0]
      await onFileUpload(file)
    }
  }

  return (
    <aside className="w-96 flex flex-col h-full bg-gray-900 border-r border-gray-800 text-gray-200 select-none">
      {/* 1. Attachment Bar: Files & SQL */}
      <div className="p-3 border-b border-gray-800 bg-gray-950/70">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-bold uppercase tracking-wider text-gray-400 flex items-center space-x-1.5">
            <Upload className="h-3.5 w-3.5 text-indigo-400" />
            <span>Universal Data Intake</span>
          </span>
          <button
            onClick={onOpenSQLModal}
            className="flex items-center space-x-1 text-[11px] text-cyan-400 hover:text-cyan-300 font-medium px-2 py-0.5 rounded bg-cyan-950/50 border border-cyan-800/40 transition"
            title="Connect SQLite, PostgreSQL, or MySQL"
          >
            <Database className="h-3 w-3" />
            <span>SQL DB</span>
          </button>
        </div>

        {/* Drag and Drop Zone or Attached Dataset Card */}
        {attachedDataset ? (
          <div className="flex items-center justify-between p-2 rounded-lg bg-indigo-950/40 border border-indigo-700/50 text-xs">
            <div className="flex items-center space-x-2 truncate">
              <FileSpreadsheet className="h-4 w-4 text-indigo-400 shrink-0" />
              <div className="truncate">
                <p className="font-semibold text-indigo-200 truncate">{attachedDataset.name}</p>
                <p className="text-[10px] text-gray-400">
                  {attachedDataset.num_rows} rows • {attachedDataset.columns.length} columns ({attachedDataset.format})
                </p>
              </div>
            </div>
            <button
              onClick={onRemoveDataset}
              className="p-1 text-gray-400 hover:text-rose-400 transition"
              title="Remove dataset"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          </div>
        ) : (
          <div
            onDragOver={(e) => {
              e.preventDefault()
              setIsDragging(true)
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`border border-dashed rounded-lg p-2.5 text-center cursor-pointer transition ${
              isDragging
                ? 'border-indigo-400 bg-indigo-950/50 text-indigo-200'
                : 'border-gray-700 hover:border-gray-500 bg-gray-900/60 text-gray-400'
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              className="hidden"
              accept=".csv,.xlsx,.pdf,.docx,.sqlite,.json,.txt"
              onChange={(e) => {
                if (e.target.files && e.target.files[0]) {
                  onFileUpload(e.target.files[0])
                }
              }}
            />
            <p className="text-xs font-medium">
              Drop file here (<span className="text-gray-300">CSV, XLSX, PDF, DOCX, JSON</span>)
            </p>
            <p className="text-[10px] text-gray-500 mt-0.5">or click to browse files</p>
          </div>
        )}
      </div>

      {/* 2. Conversational Intake & Autonomous Activity Feed */}
      <div className="flex-1 overflow-y-auto p-3 space-y-3.5">
        {/* Architect Introduction / Message Card */}
        <div className="p-3 rounded-xl bg-gray-950/80 border border-gray-800 text-xs shadow-sm">
          <div className="flex items-center space-x-2 text-indigo-400 font-bold mb-1.5">
            <Sparkles className="h-4 w-4" />
            <span>Conversational Intake Architect</span>
          </div>
          <p className="text-gray-300 leading-relaxed">{architectMessage}</p>
        </div>

        {/* Clarifying Questions Card (Rendered when needs_clarification) */}
        {clarifyingQuestions.length > 0 && (
          <div className="p-3 rounded-xl bg-gradient-to-b from-indigo-950/40 to-purple-950/30 border border-indigo-700/50 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-indigo-300 flex items-center space-x-1.5">
                <Target className="h-3.5 w-3.5" />
                <span>Pillars Alignment ({clarifyingQuestions.length})</span>
              </span>
            </div>

            {clarifyingQuestions.map((q) => (
              <div key={q.question_id} className="space-y-1.5">
                <p className="text-xs font-semibold text-gray-200">{q.question}</p>
                <div className="flex flex-wrap gap-1.5">
                  {q.suggested_options.map((opt) => {
                    const isSelected = selectedAnswers[q.question_id] === opt
                    return (
                      <button
                        key={opt}
                        onClick={() => handleOptionSelect(q.question_id, opt)}
                        className={`text-[11px] px-2.5 py-1 rounded-full font-medium transition border ${
                          isSelected
                            ? 'bg-indigo-600 border-indigo-400 text-white shadow-sm'
                            : 'bg-gray-800/90 border-gray-700 text-gray-300 hover:border-gray-500 hover:text-white'
                        }`}
                      >
                        {opt}
                      </button>
                    )
                  })}
                </div>
              </div>
            ))}

            {/* Smart Defaults & Confirm Buttons */}
            <div className="pt-2 flex items-center space-x-2">
              <button
                onClick={handleUseSmartDefaults}
                disabled={isAskingArchitect}
                className="flex-1 py-1.5 px-3 rounded-lg bg-gradient-to-r from-amber-500 to-orange-500 hover:from-amber-400 hover:to-orange-400 text-gray-950 text-xs font-extrabold flex items-center justify-center space-x-1.5 shadow-md transition"
                title="Immediately build with optimal smart defaults"
              >
                <Zap className="h-3.5 w-3.5 fill-current" />
                <span>⚡ Use Smart Defaults & Build</span>
              </button>
              <button
                onClick={handleConfirmAndBuild}
                disabled={isAskingArchitect}
                className="py-1.5 px-3 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-bold flex items-center space-x-1 transition"
                title="Build with selected options"
              >
                <Play className="h-3.5 w-3.5" />
                <span>Build</span>
              </button>
            </div>
          </div>
        )}

        {/* Autonomous Loop Activity Timeline Stream */}
        {timeline.length > 0 && (
          <div className="space-y-2 pt-1">
            <div className="text-[11px] font-bold uppercase tracking-wider text-gray-400 flex items-center space-x-1.5">
              <Cpu className="h-3.5 w-3.5 text-indigo-400" />
              <span>Autonomous Activity Stream</span>
              {isTraining && (
                <span className="flex h-2 w-2 relative ml-1">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                </span>
              )}
            </div>

            <div className="space-y-2">
              {timeline.map((item) => {
                const getStatusStyle = () => {
                  switch (item.status) {
                    case 'success':
                      return 'border-emerald-800/60 bg-emerald-950/20 text-emerald-300'
                    case 'warning':
                      return 'border-amber-800/60 bg-amber-950/20 text-amber-300'
                    case 'info':
                    default:
                      return 'border-indigo-900/50 bg-indigo-950/20 text-indigo-300'
                  }
                }

                return (
                  <div
                    key={item.id}
                    className={`p-2.5 rounded-lg border text-xs shadow-sm transition ${getStatusStyle()}`}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-semibold text-gray-200">{item.title}</span>
                      <span className="text-[10px] text-gray-500 font-mono">{item.timestamp}</span>
                    </div>
                    <p className="text-[11px] text-gray-300 leading-snug">{item.description}</p>
                  </div>
                )
              })}
            </div>
          </div>
        )}

        <div ref={chatBottomRef} />
      </div>

      {/* 3. Bottom Controls: Prompt Input & Mid-Flight Nudge Bar */}
      <div className="p-3 border-t border-gray-800 bg-gray-950/90 space-y-2">
        {/* If session is active/trained, show Mid-Flight Nudge Input */}
        {activeSessionId ? (
          <form onSubmit={handleNudgeSubmit} className="space-y-1.5">
            <div className="flex items-center justify-between text-[11px] text-amber-400 font-semibold">
              <span className="flex items-center space-x-1">
                <Sparkles className="h-3 w-3" />
                <span>Nudge Active Agent</span>
              </span>
              <span className="text-[10px] text-gray-400">Triggers targeted tuning round</span>
            </div>
            <div className="flex items-center space-x-1.5">
              <input
                type="text"
                value={nudgeText}
                onChange={(e) => setNudgeText(e.target.value)}
                placeholder="e.g. 'Block opponent wins more aggressively'..."
                disabled={isNudging}
                className="flex-1 bg-gray-900 border border-gray-700 focus:border-amber-500 rounded-lg px-2.5 py-1.5 text-xs text-gray-200 focus:outline-none placeholder-gray-500"
              />
              <button
                type="submit"
                disabled={isNudging || !nudgeText.trim()}
                className="p-1.5 rounded-lg bg-amber-500 hover:bg-amber-400 text-gray-950 font-bold transition disabled:opacity-50"
                title="Send guidance nudge"
              >
                <Send className="h-3.5 w-3.5" />
              </button>
            </div>
          </form>
        ) : (
          /* Intake Prompt Input */
          <form onSubmit={handleSendPrompt} className="space-y-1.5">
            <div className="flex items-center justify-between text-[11px] text-gray-400 font-medium">
              <span>Agent Design Prompt</span>
              <span className="text-[10px] text-indigo-400">Plain English</span>
            </div>
            <div className="flex items-center space-x-1.5">
              <input
                type="text"
                value={userPrompt}
                onChange={(e) => setUserPrompt(e.target.value)}
                placeholder="e.g. 'Train an unbeatable Tic-Tac-Toe agent'..."
                disabled={isAskingArchitect}
                className="flex-1 bg-gray-900 border border-gray-700 focus:border-indigo-500 rounded-lg px-2.5 py-1.5 text-xs text-gray-200 focus:outline-none placeholder-gray-500"
              />
              <button
                type="submit"
                disabled={isAskingArchitect || !userPrompt.trim()}
                className="p-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white transition disabled:opacity-50"
                title="Send prompt to Conversational Architect"
              >
                {isAskingArchitect ? <RefreshCw className="h-3.5 w-3.5 animate-spin" /> : <Send className="h-3.5 w-3.5" />}
              </button>
            </div>
          </form>
        )}
      </div>
    </aside>
  )
}
