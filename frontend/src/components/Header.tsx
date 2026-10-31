import React from 'react'
import {
  Sparkles,
  Settings,
  Archive,
  Download,
  CheckCircle2,
  Activity,
  AlertTriangle,
  Zap,
  History,
  ShieldCheck,
} from 'lucide-react'
import { LLMSettings } from '../types'

interface HeaderProps {
  llmSettings: LLMSettings
  isLLMVerified?: boolean
  onOpenSettings: () => void
  onOpenRegistry: () => void
  registryCount: number
  onExport: () => void
  isExporting: boolean
  hasActivePolicy: boolean
  driftStatus?: string
  driftDetail?: string
  driftSamples?: number
  onOpenConversations: () => void
  conversationCount?: number
  usageCostUsd?: number
  usageCalls?: number
  onOpenTrust: () => void
  trustVerdict?: string
  trustBlockers?: number
}

export const Header: React.FC<HeaderProps> = ({
  llmSettings,
  isLLMVerified = false,
  onOpenSettings,
  onOpenRegistry,
  registryCount,
  onExport,
  isExporting,
  hasActivePolicy,
  driftStatus = 'unmonitored',
  driftDetail,
  driftSamples = 0,
  onOpenConversations,
  conversationCount = 0,
  usageCostUsd = 0,
  usageCalls = 0,
  onOpenTrust,
  trustVerdict,
  trustBlockers = 0,
}) => {
  const getProviderLabel = () => {
    switch (llmSettings.provider) {
      case 'deepseek':
        return { name: 'DeepSeek V3 / R1', color: 'bg-blue-950/80 text-blue-300 border-blue-700/60' }
      case 'anthropic':
        return { name: 'Claude 3.7 Sonnet', color: 'bg-amber-950/80 text-amber-300 border-amber-700/60' }
      case 'openai':
        return { name: 'GPT-4o', color: 'bg-emerald-950/80 text-emerald-300 border-emerald-700/60' }
      case 'azure_openai':
        return { name: 'Azure OpenAI', color: 'bg-sky-950/80 text-sky-300 border-sky-700/60' }
      case 'openai_compatible':
        return { name: 'Self-hosted LLM', color: 'bg-teal-950/80 text-teal-300 border-teal-700/60' }
      case 'ollama':
        return { name: 'Ollama (Local)', color: 'bg-purple-950/80 text-purple-300 border-purple-700/60' }
      case 'fallback':
      default:
        return { name: 'No Key Configured', color: 'bg-red-950/80 text-red-300 border-red-700/60' }
    }
  }

  const provider = getProviderLabel()

  const driftTitle =
    driftDetail ||
    (driftStatus === 'healthy'
      ? 'Drift monitor: live observations still match the training baseline'
      : driftStatus === 'unmonitored'
        ? 'No training baseline yet — train the policy to monitor drift'
        : driftStatus === 'warming_up'
          ? 'Collecting live observations before judging drift'
          : 'Drift detected: the live inputs or actions moved away from the training baseline')

  return (
    <header className="h-14 border-b border-gray-800 bg-gray-950/95 backdrop-blur px-4 flex items-center justify-between z-20">
      {/* Brand & Subtitle */}
      <div className="flex items-center space-x-3">
        <div className="h-9 w-9 rounded-xl bg-gradient-to-tr from-cyan-500 via-indigo-600 to-fuchsia-500 flex items-center justify-center shadow-lg shadow-indigo-500/25 ring-1 ring-white/10">
          <Zap className="h-5 w-5 text-white" />
        </div>
        <div>
          <div className="flex items-center space-x-2">
            <span className="font-extrabold text-sm tracking-wide bg-gradient-to-r from-white via-gray-200 to-gray-400 bg-clip-text text-transparent">
              RL-NATIVE AGENT STUDIO
            </span>
            <span className="text-[11px] font-mono font-bold px-2 py-0.5 rounded-full bg-gradient-to-r from-indigo-500/20 to-purple-500/20 border border-indigo-500/30 text-indigo-300">
              v4.1 Autonomous JIT
            </span>
          </div>
          <div className="text-[11px] text-gray-400">
            Zero-Adapter Code Synthesis • Custom Arenas • Autonomous Self-Tuning
          </div>
        </div>
      </div>

      {/* Center / Right Control Cluster */}
      <div className="flex items-center space-x-3">
        {/* Past Conversations / Agents Switcher Button */}
        <button
          onClick={onOpenConversations}
          className="flex items-center space-x-1.5 px-3 py-1 rounded-lg bg-gray-900 border border-gray-700 hover:border-gray-500 text-gray-200 text-xs font-semibold transition"
          title="Browse and switch between past conversations and trained agents"
        >
          <History className="h-3.5 w-3.5 text-indigo-400" />
          <span>Agents</span>
          {conversationCount > 0 && (
            <span className="px-1.5 py-0.2 rounded-full bg-indigo-900/80 border border-indigo-700/60 text-indigo-300 text-[10px] font-mono">
              {conversationCount}
            </span>
          )}
        </button>
        {/* Spend so far: what this run has cost in provider tokens */}
        {usageCalls > 0 && (
          <div
            className="flex items-center space-x-1.5 px-2.5 py-1 rounded-lg text-xs font-medium border bg-gray-900 border-gray-700 text-gray-300"
            title={`${usageCalls} provider call${usageCalls === 1 ? '' : 's'} recorded for this conversation`}
          >
            <span className="font-mono">
              ${usageCostUsd < 0.01 ? usageCostUsd.toFixed(4) : usageCostUsd.toFixed(2)}
            </span>
            <span className="text-gray-500 text-[10px]">spent</span>
          </div>
        )}

        {/* Drift Status Indicator: scored live against the training baseline on every interact. */}
        <div
          className={`flex items-center space-x-1.5 px-2.5 py-1 rounded-lg text-xs font-medium border ${
            driftStatus === 'healthy'
              ? 'bg-emerald-950/40 border-emerald-800/50 text-emerald-400'
              : driftStatus === 'unmonitored'
                ? 'bg-slate-900 border-slate-700 text-slate-400'
                : driftStatus === 'warming_up'
                  ? 'bg-slate-900 border-slate-700 text-slate-300'
                  : 'bg-amber-950/40 border-amber-800/50 text-amber-300'
          }`}
          title={driftTitle}
        >
          {driftStatus === 'healthy' ? (
            <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" />
          ) : driftStatus === 'unmonitored' ? (
            <Activity className="h-3.5 w-3.5 text-slate-500" />
          ) : driftStatus === 'warming_up' ? (
            <Activity className="h-3.5 w-3.5 text-slate-400 animate-pulse" />
          ) : (
            <AlertTriangle className="h-3.5 w-3.5 text-amber-400" />
          )}
          <span className="capitalize font-mono">{driftStatus.replace('_', ' ')}</span>
          {driftSamples > 0 && driftStatus !== 'unmonitored' && (
            <span className="text-[10px] text-gray-500 font-mono">{driftSamples}</span>
          )}
        </div>

        {/* LLM Engine Badge + Settings Button */}
        <button
          onClick={onOpenSettings}
          className={`flex items-center space-x-1.5 px-3 py-1 rounded-lg text-xs font-semibold border transition hover:opacity-90 ${
            isLLMVerified
              ? provider.color
              : 'bg-amber-950/80 border-amber-600/70 text-amber-300'
          }`}
          title={isLLMVerified ? 'LLM key verified and active. Click to modify.' : 'No model connected. Exploration works; connect a provider to synthesize.'}
        >
          {isLLMVerified ? (
            <>
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
              </span>
              <Sparkles className="h-3.5 w-3.5" />
              <span>{provider.name}</span>
              <span className="text-[10px] uppercase font-bold px-1.5 py-0.2 rounded bg-emerald-950/80 border border-emerald-500/50 text-emerald-300 ml-1">
                Verified
              </span>
            </>
          ) : (
            <>
              <AlertTriangle className="h-3.5 w-3.5 text-amber-400" />
              <span>No Model</span>
              <span className="text-[10px] uppercase font-bold px-1.5 py-0.2 rounded bg-amber-900/80 border border-amber-500/50 text-amber-200 ml-1">
                Explore mode
              </span>
            </>
          )}
          <Settings className="h-3.5 w-3.5 ml-1 opacity-70" />
        </button>

        {/* Trust & Provenance: the corpus, the pre-registered verdict, the gate history */}
        <button
          onClick={onOpenTrust}
          className={`flex items-center space-x-1.5 px-3 py-1 rounded-lg text-xs font-medium border transition ${
            trustVerdict === 'win'
              ? 'bg-emerald-950/50 border-emerald-800/60 text-emerald-300 hover:border-emerald-700'
              : trustVerdict === 'loss'
                ? 'bg-rose-950/40 border-rose-900/60 text-rose-300 hover:border-rose-800'
                : 'bg-gray-900 border-gray-700 text-gray-200 hover:border-gray-600'
          }`}
          title={
            trustVerdict
              ? `Crossfire verdict: ${trustVerdict.replace(/_/g, ' ')}${
                  trustBlockers ? ` · ${trustBlockers} blocker(s)` : ''
                }`
              : 'Golden corpus, Crossfire protocol and verifier-gate history'
          }
        >
          <ShieldCheck className="h-3.5 w-3.5 opacity-80" />
          <span>Trust</span>
          {trustBlockers > 0 && (
            <span className="px-1.5 py-0.2 rounded-full bg-amber-900/80 border border-amber-700/60 text-amber-300 text-[10px] font-mono">
              {trustBlockers}
            </span>
          )}
        </button>

        {/* Policy Registry Button */}
        <button
          onClick={onOpenRegistry}
          className="flex items-center space-x-1.5 px-3 py-1 rounded-lg bg-gray-900 border border-gray-700 hover:border-gray-600 text-gray-200 text-xs font-medium transition"
          title="Open Catalog of Packaged .runbundle Policies"
        >
          <Archive className="h-3.5 w-3.5 text-gray-400" />
          <span>Policy Registry</span>
          {registryCount > 0 && (
            <span className="px-1.5 py-0.2 rounded-full bg-indigo-900/80 border border-indigo-700/60 text-indigo-300 text-[10px] font-mono">
              {registryCount}
            </span>
          )}
        </button>

        {/* Export Owned Policy (.onnx) CTA Button */}
        <button
          onClick={onExport}
          disabled={isExporting || !hasActivePolicy}
          className={`flex items-center space-x-2 px-3.5 py-1.5 rounded-lg text-xs font-bold transition shadow-md ${
            hasActivePolicy
              ? 'bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white shadow-emerald-950/40 ring-1 ring-emerald-400/30'
              : 'bg-gray-800 text-gray-500 border border-gray-700 cursor-not-allowed opacity-60'
          }`}
          title={hasActivePolicy ? 'Package policy to standalone ONNX .runbundle' : 'Train an agent first to export'}
        >
          <Download className={`h-4 w-4 ${isExporting ? 'animate-bounce' : ''}`} />
          <span>{isExporting ? 'Packaging .runbundle...' : 'Export Owned Policy (.onnx)'}</span>
        </button>
      </div>
    </header>
  )
}
