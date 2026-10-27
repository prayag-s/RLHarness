import React, { useState } from 'react'
import {
  MessageSquare,
  Send,
  Sparkles,
  TrendingUp,
  RotateCcw,
  Bot,
  User,
  Loader2,
  CheckCircle2,
} from 'lucide-react'

interface CoachMessage {
  id: string
  sender: 'user' | 'coach'
  text: string
  masteryDelta?: number
  afterMastery?: number
  appliedDiff?: Array<{ term_name: string; action: string }>
  timestamp: string
}

interface VibeCoachPanelProps {
  messages: CoachMessage[]
  onSendCritique: (text: string) => void
  isCoaching: boolean
  onUndoLast?: () => void
}

const SUGGESTIONS = [
  'Take the center square first when open',
  'Stay further away from obstacles and minimize collisions',
  'Flag any invoice over $5,000 missing a PO',
  'Never route spam tickets to urgent queue',
  'Penalize harsh thruster direction changes',
]

export const VibeCoachPanel: React.FC<VibeCoachPanelProps> = ({
  messages,
  onSendCritique,
  isCoaching,
  onUndoLast,
}) => {
  const [inputText, setInputText] = useState('')

  const handleSend = () => {
    if (!inputText.trim() || isCoaching) return
    onSendCritique(inputText.trim())
    setInputText('')
  }

  return (
    <div className="flex flex-col h-full bg-panel border-l border-panel-border overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-panel-border bg-panel-subtle flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Bot className="h-4 w-4 text-purple-400" />
          <h2 className="text-xs font-bold uppercase tracking-wider text-gray-200">
            Conversational AI Coach
          </h2>
        </div>
        {onUndoLast && (
          <button
            onClick={onUndoLast}
            title="Undo last coach adjustment"
            className="flex items-center space-x-1 text-[11px] text-gray-400 hover:text-gray-200 transition"
          >
            <RotateCcw className="h-3 w-3" />
            <span>Undo</span>
          </button>
        )}
      </div>

      {/* Chat Thread */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3.5">
        {messages.length === 0 ? (
          <div className="p-6 text-center text-xs text-gray-400 space-y-3">
            <div className="h-10 w-10 mx-auto rounded-full bg-purple-950/60 border border-purple-800/40 flex items-center justify-center text-purple-400">
              <Sparkles className="h-5 w-5" />
            </div>
            <p className="font-semibold text-gray-300">Coach your agent in plain English</p>
            <p className="text-[11px] text-gray-400">
              Type what you want improved, or click a suggestion below. The AI Coach converts your feedback into AST reward diffs, warms up the critic, and micro-burst updates policy weights in seconds.
            </p>
          </div>
        ) : (
          messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex items-start space-x-2.5 ${
                msg.sender === 'user' ? 'justify-end' : 'justify-start'
              }`}
            >
              {msg.sender === 'coach' && (
                <div className="p-1 rounded-lg bg-purple-950/80 border border-purple-800 text-purple-400 shrink-0 mt-0.5">
                  <Bot className="h-3.5 w-3.5" />
                </div>
              )}

              <div
                className={`max-w-[85%] rounded-2xl p-3 text-xs space-y-2 shadow-sm ${
                  msg.sender === 'user'
                    ? 'bg-blue-600 text-white rounded-tr-none'
                    : 'bg-gray-900 border border-gray-800 text-gray-200 rounded-tl-none'
                }`}
              >
                <p className="leading-relaxed">{msg.text}</p>

                {/* Score delta & Diff summary */}
                {msg.sender === 'coach' && msg.masteryDelta !== undefined && (
                  <div className="pt-2 border-t border-gray-800 flex items-center justify-between text-[11px]">
                    <span className="flex items-center space-x-1 text-emerald-400 font-bold">
                      <TrendingUp className="h-3.5 w-3.5" />
                      <span>
                        {msg.masteryDelta >= 0 ? `+${msg.masteryDelta}%` : `${msg.masteryDelta}%`} Mastery
                      </span>
                    </span>
                    {msg.afterMastery && (
                      <span className="text-gray-400 font-mono">Now: {msg.afterMastery}%</span>
                    )}
                  </div>
                )}

                {msg.appliedDiff && msg.appliedDiff.length > 0 && (
                  <div className="pt-1 text-[10px] text-gray-400 space-y-0.5">
                    {msg.appliedDiff.map((d, i) => (
                      <div key={i} className="flex items-center space-x-1 text-purple-300 font-mono">
                        <CheckCircle2 className="h-2.5 w-2.5 text-purple-400" />
                        <span>Rule '{d.term_name}' adjusted</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {msg.sender === 'user' && (
                <div className="p-1 rounded-lg bg-blue-900 border border-blue-700 text-white shrink-0 mt-0.5">
                  <User className="h-3.5 w-3.5" />
                </div>
              )}
            </div>
          ))
        )}
      </div>

      {/* Suggestion Chips */}
      <div className="px-4 py-2 border-t border-gray-800/80 bg-panel-subtle/50 flex items-center space-x-1.5 overflow-x-auto no-scrollbar text-[11px]">
        <span className="text-gray-400 font-semibold shrink-0">💡 Ideas:</span>
        {SUGGESTIONS.map((sug, i) => (
          <button
            key={i}
            onClick={() => onSendCritique(sug)}
            disabled={isCoaching}
            className="px-2.5 py-1 rounded-full bg-gray-900 border border-gray-800 hover:border-purple-600/60 text-gray-300 hover:text-white transition shrink-0 whitespace-nowrap"
          >
            {sug}
          </button>
        ))}
      </div>

      {/* Input bar */}
      <div className="p-3 border-t border-panel-border bg-panel flex items-center space-x-2">
        <input
          type="text"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') handleSend()
          }}
          disabled={isCoaching}
          placeholder="Type feedback: e.g. 'Keep a wider safety buffer around obstacles'..."
          className="flex-1 bg-gray-900 border border-gray-700 rounded-xl px-3 py-2 text-xs text-gray-100 placeholder-gray-500 focus:outline-none focus:border-purple-500"
        />
        <button
          onClick={handleSend}
          disabled={isCoaching || !inputText.trim()}
          className="p-2 rounded-xl bg-purple-600 hover:bg-purple-500 text-white disabled:opacity-40 transition shadow-lg shadow-purple-600/25"
        >
          {isCoaching ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <Send className="h-4 w-4" />
          )}
        </button>
      </div>
    </div>
  )
}
