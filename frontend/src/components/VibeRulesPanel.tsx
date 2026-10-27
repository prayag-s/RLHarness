import React from 'react'
import {
  ShieldCheck,
  Zap,
  Star,
  AlertTriangle,
  Slash,
  ChevronUp,
  ChevronDown,
  Info,
  CheckCircle,
  Sparkles,
} from 'lucide-react'
import { PlainRuleCard, SkillSummary } from '../types'

interface VibeRulesPanelProps {
  skillSummary: SkillSummary | null
  rules: PlainRuleCard[]
  onAdjustRuleImportance: (termName: string, newImportance: string) => void
  isAdjusting: boolean
}

const IMPORTANCE_SEQUENCE = ['Strict Ban', 'Avoid', 'Bonus', 'High Priority', 'Must Do']

export const VibeRulesPanel: React.FC<VibeRulesPanelProps> = ({
  skillSummary,
  rules,
  onAdjustRuleImportance,
  isAdjusting,
}) => {
  const score = skillSummary?.mastery_score ?? 50
  const winRate = skillSummary ? Math.round(skillSummary.win_rate * 100) : 50
  const badge = skillSummary?.status_badge ?? 'Learning'

  // SVG circular gauge calculation
  const radius = 38
  const circumference = 2 * Math.PI * radius
  const strokeDashoffset = circumference - (score / 100) * circumference

  const getBadgeStyle = (importance: string) => {
    switch (importance) {
      case 'Must Do':
        return 'bg-emerald-950/70 border-emerald-700/60 text-emerald-300'
      case 'High Priority':
        return 'bg-blue-950/70 border-blue-700/60 text-blue-300'
      case 'Bonus':
        return 'bg-indigo-950/70 border-indigo-700/60 text-indigo-300'
      case 'Avoid':
        return 'bg-amber-950/70 border-amber-700/60 text-amber-300'
      case 'Strict Ban':
        return 'bg-rose-950/70 border-rose-700/60 text-rose-300'
      default:
        return 'bg-gray-800 border-gray-700 text-gray-300'
    }
  }

  const getIcon = (importance: string) => {
    switch (importance) {
      case 'Must Do':
        return <ShieldCheck className="h-4 w-4 text-emerald-400" />
      case 'High Priority':
        return <Zap className="h-4 w-4 text-blue-400" />
      case 'Bonus':
        return <Star className="h-4 w-4 text-indigo-400" />
      case 'Avoid':
        return <AlertTriangle className="h-4 w-4 text-amber-400" />
      case 'Strict Ban':
        return <Slash className="h-4 w-4 text-rose-400" />
      default:
        return <Sparkles className="h-4 w-4 text-purple-400" />
    }
  }

  const handleStepImportance = (card: PlainRuleCard, direction: 'up' | 'down') => {
    const currentIndex = IMPORTANCE_SEQUENCE.indexOf(card.importance)
    if (currentIndex === -1) return
    const nextIndex = direction === 'up' ? currentIndex + 1 : currentIndex - 1
    if (nextIndex >= 0 && nextIndex < IMPORTANCE_SEQUENCE.length) {
      onAdjustRuleImportance(card.id, IMPORTANCE_SEQUENCE[nextIndex])
    }
  }

  return (
    <div className="flex flex-col h-full bg-panel border-r border-panel-border overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-panel-border bg-panel-subtle flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Sparkles className="h-4 w-4 text-purple-400" />
          <h2 className="text-xs font-bold uppercase tracking-wider text-gray-200">
            Agent Mastery & Rules
          </h2>
        </div>
        <span className="text-[10px] text-gray-400 font-mono">0-Jargon RL</span>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {/* Mastery Gauge Card */}
        <div className="p-4 rounded-xl bg-gradient-to-br from-gray-900 via-gray-900/90 to-purple-950/30 border border-gray-800 shadow-md flex items-center space-x-4">
          {/* Circular Progress Gauge */}
          <div className="relative w-24 h-24 shrink-0 flex items-center justify-center">
            <svg className="w-24 h-24 transform -rotate-90">
              <circle
                cx="48"
                cy="48"
                r={radius}
                stroke="#1f2937"
                strokeWidth="7"
                fill="transparent"
              />
              <circle
                cx="48"
                cy="48"
                r={radius}
                stroke="url(#masteryGrad)"
                strokeWidth="7"
                strokeDasharray={circumference}
                strokeDashoffset={strokeDashoffset}
                strokeLinecap="round"
                fill="transparent"
                className="transition-all duration-700 ease-out"
              />
              <defs>
                <linearGradient id="masteryGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stopColor="#a855f7" />
                  <stop offset="50%" stopColor="#3b82f6" />
                  <stop offset="100%" stopColor="#10b981" />
                </linearGradient>
              </defs>
            </svg>
            <div className="absolute flex flex-col items-center justify-center text-center">
              <span className="text-xl font-extrabold text-white font-mono leading-none">
                {score}%
              </span>
              <span className="text-[9px] uppercase tracking-wider font-semibold text-gray-400 mt-0.5">
                Mastery
              </span>
            </div>
          </div>

          {/* Details */}
          <div className="flex-1 space-y-1.5">
            <div className="flex items-center space-x-2">
              <span className="px-2 py-0.5 rounded text-[11px] font-bold bg-emerald-950 border border-emerald-800 text-emerald-300">
                {badge}
              </span>
              <span className="text-[11px] font-medium text-gray-400">
                Win/Success: <strong className="text-gray-200">{winRate}%</strong>
              </span>
            </div>
            {skillSummary?.key_insights && skillSummary.key_insights.length > 0 && (
              <p className="text-[11px] text-gray-300 leading-tight">
                {skillSummary.key_insights[0]}
              </p>
            )}
            <div className="text-[10px] text-gray-400">
              Evaluated over {skillSummary?.total_episodes || 0} episodes
            </div>
          </div>
        </div>

        {/* Plain-English Active Rules */}
        <div className="space-y-2">
          <div className="flex items-center justify-between px-1">
            <span className="text-xs font-semibold text-gray-300">Behavioral Rules ({rules.length})</span>
            <span className="text-[10px] text-gray-400">Click ▲▼ to adjust priority</span>
          </div>

          {rules.length === 0 ? (
            <div className="p-6 text-center text-xs text-gray-400 border border-dashed border-gray-800 rounded-xl">
              No rules active. Enter a task prompt above or pick a demo to synthesize an agent.
            </div>
          ) : (
            rules.map((rule) => (
              <div
                key={rule.id}
                className="p-3 rounded-xl bg-gray-900/90 border border-gray-800 hover:border-gray-700/80 transition shadow-sm space-y-2"
              >
                <div className="flex items-start justify-between">
                  <div className="flex items-center space-x-2">
                    <div className="p-1.5 rounded-lg bg-gray-800 border border-gray-700/60">
                      {getIcon(rule.importance)}
                    </div>
                    <div>
                      <h4 className="text-xs font-bold text-gray-100">{rule.title}</h4>
                      <span className="text-[10px] text-gray-400 uppercase tracking-wider font-semibold">
                        {rule.category}
                      </span>
                    </div>
                  </div>

                  {/* Priority Pill & Adjust Buttons */}
                  <div className="flex items-center space-x-1.5">
                    <span className={`px-2 py-0.5 rounded-full text-[10px] font-semibold border ${getBadgeStyle(rule.importance)}`}>
                      {rule.importance}
                    </span>
                    <div className="flex flex-col space-y-0.5">
                      <button
                        onClick={() => handleStepImportance(rule, 'up')}
                        disabled={isAdjusting || rule.importance === 'Must Do'}
                        title="Increase priority"
                        className="p-0.5 rounded hover:bg-gray-800 text-gray-400 hover:text-white disabled:opacity-30 transition"
                      >
                        <ChevronUp className="h-3 w-3" />
                      </button>
                      <button
                        onClick={() => handleStepImportance(rule, 'down')}
                        disabled={isAdjusting || rule.importance === 'Strict Ban'}
                        title="Decrease priority"
                        className="p-0.5 rounded hover:bg-gray-800 text-gray-400 hover:text-white disabled:opacity-30 transition"
                      >
                        <ChevronDown className="h-3 w-3" />
                      </button>
                    </div>
                  </div>
                </div>

                <p className="text-xs text-gray-300 leading-snug">{rule.description}</p>

                {/* Mathematical rule debug hint */}
                <div className="flex items-center justify-between text-[10px] text-gray-400 pt-1 border-t border-gray-800/60 font-mono">
                  <span>Weight: {rule.weight > 0 ? `+${rule.weight}` : rule.weight}</span>
                  <span className="truncate max-w-[160px]" title={rule.expression}>
                    {rule.expression}
                  </span>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  )
}
