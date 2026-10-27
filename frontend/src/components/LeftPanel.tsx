import React, { useState } from 'react'
import {
  Sparkles,
  Code2,
  ListFilter,
  History,
  RotateCcw,
  Sliders,
  CheckCircle2,
  Copy,
  Check,
} from 'lucide-react'
import { RewardDSL, VersionRecord } from '../types'

interface LeftPanelProps {
  prompt: string
  onPromptChange: (val: string) => void
  onCompile: () => void
  isCompiling: boolean
  rewardDsl: RewardDSL | null
  onUpdateDsl?: (dsl: RewardDSL) => void
  versionHistory: VersionRecord[]
  currentVersion: number
  onRollback: (version: number) => void
}

export const LeftPanel: React.FC<LeftPanelProps> = ({
  prompt,
  onPromptChange,
  onCompile,
  isCompiling,
  rewardDsl,
  versionHistory,
  currentVersion,
  onRollback,
}) => {
  const [activeTab, setActiveTab] = useState<'terms' | 'yaml' | 'history'>('terms')
  const [copied, setCopied] = useState(false)

  const dslYaml = rewardDsl
    ? `# RewardDSL Specification (v${currentVersion})
version: "${rewardDsl.version || '1.0'}"
horizon:
  gamma: ${rewardDsl.horizon?.gamma ?? 0.99}
  critic_warmup_epochs_on_diff: ${rewardDsl.horizon?.critic_warmup_epochs_on_diff ?? 15}
  kl_coef: ${rewardDsl.horizon?.kl_coef ?? 0.2}

reward:
  aggregate: "${rewardDsl.reward?.aggregate || 'sum'}"
  clamp: [${rewardDsl.reward?.clamp?.[0] ?? -50.0}, ${rewardDsl.reward?.clamp?.[1] ?? 50.0}]
  terms:
${(rewardDsl.reward?.terms || [])
  .map(
    (t) => `    - name: "${t.name}"
      when: "${t.when}"
      value: ${t.value}`,
  )
  .join('\n')}
${
  rewardDsl.reward?.pinned_episodes?.length
    ? `  pinned_episodes:
${rewardDsl.reward.pinned_episodes
  .map(
    (p) => `    - episode_id: ${p.episode_id}
      forbid_action: "${p.forbid_action}"
      penalty: ${p.penalty}`,
  )
  .join('\n')}`
    : ''
}`
    : '# No Task Compiled Yet.\n# Enter natural language task above and click [Compile & Warm-Start].'

  const handleCopyYaml = () => {
    navigator.clipboard.writeText(dslYaml)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="w-[28%] min-w-[320px] max-w-[420px] h-full flex flex-col border-r border-panel-border bg-panel overflow-hidden">
      {/* Task Authoring Header */}
      <div className="p-3 border-b border-panel-border bg-gray-900/60">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center space-x-1.5">
            <Sparkles className="h-4 w-4 text-blue-400" />
            <h2 className="text-xs font-bold text-gray-200 uppercase tracking-wider">
              Task Specification
            </h2>
          </div>
          <span className="text-[10px] text-gray-400 font-mono">NL → AST Compiler</span>
        </div>

        {/* Prompt Input Area */}
        <div className="space-y-2">
          <textarea
            value={prompt}
            onChange={(e) => onPromptChange(e.target.value)}
            placeholder="Describe the desired behavior in natural language..."
            rows={3}
            className="w-full text-xs bg-gray-950/80 border border-gray-800 rounded-lg p-2.5 text-gray-200 placeholder-gray-500 focus:outline-none focus:border-blue-500 transition resize-none leading-relaxed"
          />

          <button
            onClick={onCompile}
            disabled={isCompiling || !prompt.trim()}
            className="w-full flex items-center justify-center space-x-2 py-2 px-3 rounded-lg bg-blue-600 hover:bg-blue-500 text-white font-medium text-xs shadow-md shadow-blue-600/20 transition disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {isCompiling ? (
              <>
                <div className="h-3.5 w-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                <span>Compiling & Warm-Starting...</span>
              </>
            ) : (
              <>
                <Sparkles className="h-3.5 w-3.5" />
                <span>Compile & Warm-Start</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Tabs Bar */}
      <div className="flex border-b border-panel-border bg-gray-950/40 text-xs px-2 pt-1">
        <button
          onClick={() => setActiveTab('terms')}
          className={`flex items-center space-x-1.5 px-3 py-1.5 border-b-2 font-medium transition ${
            activeTab === 'terms'
              ? 'border-blue-500 text-blue-400 bg-gray-900/40'
              : 'border-transparent text-gray-400 hover:text-gray-200'
          }`}
        >
          <ListFilter className="h-3.5 w-3.5" />
          <span>Active Terms ({rewardDsl?.reward?.terms?.length || 0})</span>
        </button>

        <button
          onClick={() => setActiveTab('yaml')}
          className={`flex items-center space-x-1.5 px-3 py-1.5 border-b-2 font-medium transition ${
            activeTab === 'yaml'
              ? 'border-blue-500 text-blue-400 bg-gray-900/40'
              : 'border-transparent text-gray-400 hover:text-gray-200'
          }`}
        >
          <Code2 className="h-3.5 w-3.5" />
          <span>YAML Inspector</span>
        </button>

        <button
          onClick={() => setActiveTab('history')}
          className={`flex items-center space-x-1.5 px-3 py-1.5 border-b-2 font-medium transition ${
            activeTab === 'history'
              ? 'border-blue-500 text-blue-400 bg-gray-900/40'
              : 'border-transparent text-gray-400 hover:text-gray-200'
          }`}
        >
          <History className="h-3.5 w-3.5" />
          <span>History ({versionHistory.length})</span>
        </button>
      </div>

      {/* Tab Content */}
      <div className="flex-1 overflow-y-auto p-3">
        {activeTab === 'terms' && (
          <div className="space-y-2.5">
            {rewardDsl?.reward?.terms && rewardDsl.reward.terms.length > 0 ? (
              rewardDsl.reward.terms.map((term, idx) => {
                const isPositive = Number(term.value) > 0
                return (
                  <div
                    key={idx}
                    className="p-2.5 rounded-lg bg-gray-900/70 border border-gray-800/80 hover:border-gray-700 transition"
                  >
                    <div className="flex items-center justify-between mb-1.5">
                      <span className="font-mono font-semibold text-xs text-blue-300">
                        {term.name}
                      </span>
                      <span
                        className={`font-mono text-xs font-bold px-2 py-0.5 rounded ${
                          isPositive
                            ? 'bg-emerald-950/80 text-emerald-400 border border-emerald-800/40'
                            : 'bg-rose-950/80 text-rose-400 border border-rose-800/40'
                        }`}
                      >
                        {isPositive ? `+${term.value}` : term.value}
                      </span>
                    </div>

                    <div className="text-[11px] text-gray-400 font-mono bg-gray-950/60 p-2 rounded border border-gray-800/50 break-words leading-relaxed">
                      <span className="text-gray-500">when: </span>
                      <span className="text-amber-300/90">{String(term.when)}</span>
                    </div>
                  </div>
                )
              })
            ) : (
              <div className="text-center py-10 text-gray-500 text-xs">
                No active terms. Compile a task above.
              </div>
            )}

            {/* Horizon & Hyperparameters Summary */}
            {rewardDsl && (
              <div className="mt-4 p-2.5 rounded-lg bg-gray-950/40 border border-gray-800 text-[11px] space-y-1 text-gray-400">
                <div className="flex items-center space-x-1.5 text-gray-300 font-semibold mb-1">
                  <Sliders className="h-3.5 w-3.5 text-gray-400" />
                  <span>Horizon & Warmup Config</span>
                </div>
                <div className="flex justify-between">
                  <span>Discount factor (γ):</span>
                  <span className="font-mono text-gray-200">
                    {rewardDsl.horizon?.gamma ?? 0.99}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Critic Warmup Epochs:</span>
                  <span className="font-mono text-gray-200">
                    {rewardDsl.horizon?.critic_warmup_epochs_on_diff ?? 15}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Reward Clamping:</span>
                  <span className="font-mono text-gray-200">
                    [{rewardDsl.reward?.clamp?.[0] ?? -50},{' '}
                    {rewardDsl.reward?.clamp?.[1] ?? 50}]
                  </span>
                </div>
                {rewardDsl.reward?.pinned_episodes &&
                  rewardDsl.reward.pinned_episodes.length > 0 && (
                    <div className="flex justify-between pt-1 border-t border-gray-800 text-amber-400">
                      <span>Locked Golden Episodes:</span>
                      <span className="font-mono font-bold">
                        {rewardDsl.reward.pinned_episodes.length}
                      </span>
                    </div>
                  )}
              </div>
            )}
          </div>
        )}

        {activeTab === 'yaml' && (
          <div className="relative h-full flex flex-col">
            <div className="flex justify-end mb-2">
              <button
                onClick={handleCopyYaml}
                className="flex items-center space-x-1 text-[11px] bg-gray-800 hover:bg-gray-700 text-gray-300 px-2 py-1 rounded transition"
              >
                {copied ? (
                  <>
                    <Check className="h-3 w-3 text-emerald-400" />
                    <span className="text-emerald-400">Copied</span>
                  </>
                ) : (
                  <>
                    <Copy className="h-3 w-3" />
                    <span>Copy YAML</span>
                  </>
                )}
              </button>
            </div>
            <pre className="flex-1 p-3 rounded-lg bg-gray-950 border border-gray-800 text-[11px] text-gray-300 font-mono overflow-auto leading-relaxed whitespace-pre-wrap select-all">
              {dslYaml}
            </pre>
          </div>
        )}

        {activeTab === 'history' && (
          <div className="space-y-2">
            {versionHistory.length > 0 ? (
              versionHistory.map((item) => {
                const isCurrent = item.version === currentVersion
                return (
                  <div
                    key={item.version}
                    className={`p-2.5 rounded-lg border transition ${
                      isCurrent
                        ? 'bg-blue-950/30 border-blue-800/60'
                        : 'bg-gray-900/60 border-gray-800 hover:border-gray-700'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <div className="flex items-center space-x-1.5">
                        <span className="font-mono text-xs font-bold text-blue-400">
                          v{item.version}
                        </span>
                        {isCurrent && (
                          <span className="text-[10px] bg-blue-900/60 text-blue-300 px-1.5 py-0.2 rounded border border-blue-700/50">
                            Active
                          </span>
                        )}
                      </div>
                      <span className="text-[10px] text-gray-500 font-mono">
                        {item.timestamp}
                      </span>
                    </div>

                    <p className="text-xs text-gray-300 mb-2 italic">
                      "{item.description || item.critique_text || 'Initial compilation'}"
                    </p>

                    {!isCurrent && (
                      <button
                        onClick={() => onRollback(item.version)}
                        className="flex items-center space-x-1 text-[11px] text-gray-400 hover:text-white bg-gray-800 hover:bg-gray-700 px-2 py-0.5 rounded transition"
                      >
                        <RotateCcw className="h-3 w-3" />
                        <span>Rollback to v{item.version}</span>
                      </button>
                    )}
                  </div>
                )
              })
            ) : (
              <div className="text-center py-10 text-gray-500 text-xs">
                No versions recorded yet.
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
