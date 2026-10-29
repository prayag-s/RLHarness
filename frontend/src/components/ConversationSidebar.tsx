import React from 'react'
import {
  MessageSquare,
  Plus,
  Trash2,
  X,
  Sparkles,
  CheckCircle2,
  Clock,
  Target,
  Layers,
  ChevronRight,
  RefreshCw,
} from 'lucide-react'
import { ConversationSummary } from '../types'

interface ConversationSidebarProps {
  isOpen: boolean
  onClose: () => void
  conversations: ConversationSummary[]
  activeConversationId?: string
  onSelectConversation: (conversationId: string) => void
  onNewAgent: () => void
  onDeleteConversation: (conversationId: string) => void
  onClearAll?: () => void
  isLoading?: boolean
}

export const ConversationSidebar: React.FC<ConversationSidebarProps> = ({
  isOpen,
  onClose,
  conversations,
  activeConversationId,
  onSelectConversation,
  onNewAgent,
  onDeleteConversation,
  onClearAll,
  isLoading = false,
}) => {
  if (!isOpen) return null

  const getStatusBadge = (status: string, metric: number, target: number) => {
    if (metric >= target && target > 0) {
      return (
        <span className="px-1.5 py-0.5 rounded-full text-[10px] font-bold bg-emerald-950/80 border border-emerald-600/60 text-emerald-300 flex items-center space-x-1">
          <CheckCircle2 className="h-2.5 w-2.5" />
          <span>100% Met</span>
        </span>
      )
    }

    switch (status) {
      case 'target_achieved':
        return (
          <span className="px-1.5 py-0.5 rounded-full text-[10px] font-bold bg-emerald-950/80 border border-emerald-600/60 text-emerald-300">
            Mastery Met
          </span>
        )
      case 'training':
        return (
          <span className="px-1.5 py-0.5 rounded-full text-[10px] font-bold bg-amber-950/80 border border-amber-600/60 text-amber-300 flex items-center space-x-1">
            <span className="h-1.5 w-1.5 rounded-full bg-amber-400 animate-pulse" />
            <span>Training</span>
          </span>
        )
      case 'completed':
        return (
          <span className="px-1.5 py-0.5 rounded-full text-[10px] font-bold bg-cyan-950/80 border border-cyan-600/60 text-cyan-300">
            Concluded
          </span>
        )
      case 'ready_to_build':
        return (
          <span className="px-1.5 py-0.5 rounded-full text-[10px] font-bold bg-indigo-950/80 border border-indigo-600/60 text-indigo-300">
            Ready to Build
          </span>
        )
      case 'clarifying':
      default:
        return (
          <span className="px-1.5 py-0.5 rounded-full text-[10px] font-bold bg-purple-950/80 border border-purple-600/60 text-purple-300">
            Intake
          </span>
        )
    }
  }

  const formatTimestamp = (isoString?: string) => {
    if (!isoString) return ''
    try {
      const date = new Date(isoString)
      return date.toLocaleDateString(undefined, {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      })
    } catch {
      return isoString.slice(0, 16)
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex">
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/60 backdrop-blur-sm transition-opacity"
        onClick={onClose}
      />

      {/* Drawer Container */}
      <div className="relative w-84 max-w-full bg-gray-950 border-r border-gray-800 flex flex-col h-full shadow-2xl z-10 select-none animate-in slide-in-from-left duration-200">
        {/* Drawer Header */}
        <div className="p-4 border-b border-gray-800 flex items-center justify-between bg-gray-900/80">
          <div className="flex items-center space-x-2">
            <Layers className="h-5 w-5 text-indigo-400" />
            <div>
              <h2 className="text-sm font-bold text-gray-100">Past Conversations</h2>
              <p className="text-[10px] text-gray-400">Persistent agent synthesis history</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-lg text-gray-400 hover:text-white hover:bg-gray-800 transition"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* [+ New Agent] & Clear Actions */}
        <div className="p-3 border-b border-gray-800/80 bg-gray-950/90 flex items-center gap-2">
          <button
            onClick={() => {
              onNewAgent()
              onClose()
            }}
            className="flex-1 py-2 px-3 rounded-lg bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white text-xs font-bold flex items-center justify-center space-x-2 shadow-md transition active:scale-[0.99]"
          >
            <Plus className="h-4 w-4" />
            <span>+ New Agent Synthesis</span>
          </button>
          {onClearAll && conversations.length > 0 && (
            <button
              type="button"
              onClick={() => {
                if (window.confirm(`Delete all ${conversations.length} saved past conversations? This cannot be undone.`)) {
                  onClearAll()
                }
              }}
              className="py-2 px-2.5 rounded-lg border border-rose-900/60 bg-rose-950/30 hover:bg-rose-950/80 hover:border-rose-700 text-rose-300 text-xs font-semibold flex items-center justify-center space-x-1.5 transition active:scale-[0.99]"
              title="Delete all past conversations"
            >
              <Trash2 className="h-3.5 w-3.5" />
              <span>Clear All</span>
            </button>
          )}
        </div>

        {/* Conversations List */}
        <div className="flex-1 overflow-y-auto p-3 space-y-2">
          {isLoading && (
            <div className="p-4 text-center text-xs text-gray-400 flex items-center justify-center space-x-2">
              <RefreshCw className="h-4 w-4 animate-spin text-indigo-400" />
              <span>Loading saved agents...</span>
            </div>
          )}

          {!isLoading && conversations.length === 0 && (
            <div className="p-6 text-center space-y-2 text-gray-500">
              <MessageSquare className="h-8 w-8 mx-auto opacity-30 text-gray-400" />
              <p className="text-xs font-medium">No saved conversations yet.</p>
              <p className="text-[11px] text-gray-600">
                Your agent synthesis runs, chat intake, and trained models will appear here automatically.
              </p>
            </div>
          )}

          {conversations.map((conv) => {
            const isActive = activeConversationId === conv.conversation_id
            return (
              <div
                key={conv.conversation_id}
                className={`p-3 rounded-xl border transition group relative cursor-pointer ${
                  isActive
                    ? 'bg-indigo-950/60 border-indigo-600 text-gray-100 shadow-md ring-1 ring-indigo-500/50'
                    : 'bg-gray-900/70 border-gray-800 text-gray-300 hover:border-gray-700 hover:bg-gray-900'
                }`}
                onClick={() => {
                  onSelectConversation(conv.conversation_id)
                  onClose()
                }}
              >
                {/* Title and Delete Button */}
                <div className="flex items-start justify-between gap-2 mb-1.5">
                  <span className="font-semibold text-xs leading-snug line-clamp-2 text-gray-200">
                    {conv.title}
                  </span>
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation()
                      if (window.confirm(`Delete "${conv.title}"?`)) {
                        onDeleteConversation(conv.conversation_id)
                      }
                    }}
                    className="opacity-0 group-hover:opacity-100 p-1 rounded text-gray-500 hover:text-rose-400 hover:bg-gray-800 transition shrink-0"
                    title="Delete conversation"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                </div>

                {/* Status Badge & Score */}
                <div className="flex items-center justify-between text-[11px] pt-1">
                  <div>
                    {getStatusBadge(conv.status, conv.best_metric, conv.target_metric_threshold)}
                  </div>
                  <div className="flex items-center space-x-1 font-mono text-[11px]">
                    <span className="text-gray-400">Score:</span>
                    <span className="font-bold text-indigo-300">
                      {conv.best_metric > 0 ? `${conv.best_metric.toFixed(1)}%` : '—'}
                    </span>
                  </div>
                </div>

                {/* Timestamp & Round */}
                <div className="flex items-center justify-between text-[10px] text-gray-500 pt-2 border-t border-gray-800/60 mt-2 font-mono">
                  <span>Round {conv.current_round}</span>
                  <span>{formatTimestamp(conv.updated_at)}</span>
                </div>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
