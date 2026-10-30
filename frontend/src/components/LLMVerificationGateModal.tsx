import React, { useState } from 'react'
import {
  Key,
  Server,
  Cpu,
  Check,
  Sparkles,
  AlertCircle,
  RefreshCw,
  Eye,
  EyeOff,
  ShieldAlert,
  ArrowRight,
} from 'lucide-react'
import { LLMSettings } from '../types'
import { saveAndVerifyLLMSettings } from '../services/api'

interface LLMVerificationGateModalProps {
  isOpen: boolean
  currentSettings: LLMSettings
  onVerified: (settings: LLMSettings) => void
  /** Exploration is not blocked on a key, so the dialog can be closed (G13). */
  canDismiss?: boolean
  onClose?: () => void
}

export const LLMVerificationGateModal: React.FC<LLMVerificationGateModalProps> = ({
  isOpen,
  currentSettings,
  onVerified,
  canDismiss = false,
  onClose,
}) => {
  const [provider, setProvider] = useState<'deepseek' | 'openai' | 'anthropic' | 'ollama'>(
    currentSettings.provider !== 'fallback' ? (currentSettings.provider as any) : 'deepseek',
  )
  const [apiKey, setApiKey] = useState<string>(currentSettings.apiKey || '')
  const [baseUrl, setBaseUrl] = useState<string>(
    currentSettings.baseUrl || 'https://api.deepseek.com/v1',
  )
  const [modelName, setModelName] = useState<string>(
    currentSettings.modelName || 'deepseek-chat',
  )
  const [showKey, setShowKey] = useState(false)
  const [isValidating, setIsValidating] = useState(false)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)
  const [successMsg, setSuccessMsg] = useState<string | null>(null)

  if (!isOpen) return null

  const providers = [
    {
      id: 'deepseek' as const,
      name: 'DeepSeek V3 / R1',
      defaultBaseUrl: 'https://api.deepseek.com/v1',
      defaultModel: 'deepseek-chat',
      badge: 'Recommended',
      keyPlaceholder: 'sk-...',
    },
    {
      id: 'openai' as const,
      name: 'OpenAI (GPT-4o)',
      defaultBaseUrl: 'https://api.openai.com/v1',
      defaultModel: 'gpt-4o',
      badge: 'Popular',
      keyPlaceholder: 'sk-proj-...',
    },
    {
      id: 'anthropic' as const,
      name: 'Anthropic (Claude 3.7)',
      defaultBaseUrl: 'https://api.anthropic.com/v1',
      defaultModel: 'claude-3-7-sonnet-20250219',
      badge: 'Smart & Fast',
      keyPlaceholder: 'sk-ant-...',
    },
    {
      id: 'ollama' as const,
      name: 'Ollama (Local / vLLM)',
      defaultBaseUrl: 'http://localhost:11434/v1',
      defaultModel: 'llama3.1',
      badge: 'Local Endpoint',
      keyPlaceholder: 'Optional for local Ollama',
    },
  ]

  const handleProviderSelect = (p: (typeof providers)[0]) => {
    setProvider(p.id)
    setBaseUrl(p.defaultBaseUrl)
    setModelName(p.defaultModel)
    setErrorMsg(null)
    setSuccessMsg(null)
  }

  const handleVerifyAndUnlock = async (e: React.FormEvent) => {
    e.preventDefault()

    if (provider !== 'ollama' && !apiKey.trim()) {
      setErrorMsg(`API Key is required for ${provider.toUpperCase()}.`)
      return
    }

    setIsValidating(true)
    setErrorMsg(null)
    setSuccessMsg(null)

    const payload: LLMSettings = {
      provider,
      apiKey: apiKey.trim(),
      baseUrl: baseUrl.trim(),
      modelName: modelName.trim(),
    }

    try {
      const res = await saveAndVerifyLLMSettings(payload)

      if (res.is_verified || res.success) {
        setSuccessMsg(
          `Connection verified successfully to ${provider.toUpperCase()} (${res.model_name || modelName})! Studio unlocked.`,
        )
        // Store in localStorage
        localStorage.setItem('rl_ide_v4_llm_settings', JSON.stringify(payload))
        setTimeout(() => {
          onVerified(payload)
        }, 800)
      } else {
        setErrorMsg(
          res.error ||
            'LLM connectivity test failed. Please verify your API key, model name, and network connection.',
        )
      }
    } catch (err: any) {
      setErrorMsg(
        err.message || 'Verification request failed. Could not reach backend validation service.',
      )
    } finally {
      setIsValidating(false)
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/85 backdrop-blur-md p-4 animate-in fade-in duration-200">
      <div className="w-full max-w-xl bg-gray-900 border border-indigo-500/40 rounded-2xl shadow-2xl shadow-indigo-950/50 overflow-hidden flex flex-col text-gray-200">
        {/* Header Banner */}
        <div className="px-6 py-5 bg-gradient-to-r from-indigo-950/70 via-gray-950 to-indigo-950/70 border-b border-gray-800 flex items-start space-x-3.5">
          <div className="h-10 w-10 rounded-xl bg-indigo-900/60 border border-indigo-500/50 flex items-center justify-center text-indigo-300 shrink-0 shadow-lg">
            <ShieldAlert className="h-5 w-5" />
          </div>
          <div className="flex-1">
            <div className="flex items-center space-x-2">
              <h2 className="text-sm font-extrabold text-white tracking-wide">
                CONNECT A MODEL
              </h2>
              <span className="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded-full bg-indigo-950 border border-indigo-600/60 text-indigo-300">
                Required to synthesize
              </span>
            </div>
            <p className="text-xs text-gray-300 mt-1 leading-relaxed">
              Synthesizing environments, chatting with the architect and training need a verified
              provider. Uploading data, browsing existing agents and the arena preview already work
              without one.
            </p>
          </div>
          {canDismiss && onClose && (
            <button
              type="button"
              onClick={onClose}
              className="text-gray-400 hover:text-white text-xs px-2 py-1 rounded border border-gray-700 hover:border-gray-500 transition shrink-0"
              title="Keep exploring without a key"
            >
              Explore first
            </button>
          )}
        </div>

        {/* Form Body */}
        <form onSubmit={handleVerifyAndUnlock} className="p-6 space-y-5 text-xs">
          {/* Provider Selection */}
          <div className="space-y-2">
            <label className="text-gray-300 font-semibold flex items-center space-x-1.5">
              <Sparkles className="h-3.5 w-3.5 text-indigo-400" />
              <span>Select LLM Provider:</span>
            </label>
            <div className="grid grid-cols-2 gap-2.5">
              {providers.map((p) => {
                const isSelected = provider === p.id
                return (
                  <button
                    type="button"
                    key={p.id}
                    onClick={() => handleProviderSelect(p)}
                    className={`p-3 rounded-xl border text-left transition flex flex-col justify-between ${
                      isSelected
                        ? 'bg-indigo-950/80 border-indigo-500 text-white shadow-lg ring-1 ring-indigo-500/30'
                        : 'bg-gray-950/70 border-gray-800 text-gray-400 hover:border-gray-700 hover:text-gray-200'
                    }`}
                  >
                    <span className="font-bold text-xs text-gray-200">{p.name}</span>
                    <span className="text-[10px] text-indigo-400 font-mono mt-1">{p.badge}</span>
                  </button>
                )
              })}
            </div>
          </div>

          {/* API Key Input */}
          <div className="space-y-1.5">
            <label className="text-gray-300 font-semibold flex items-center space-x-1.5">
              <Key className="h-3.5 w-3.5 text-cyan-400" />
              <span>{provider.toUpperCase()} API Key:</span>
              <span className="text-red-400 font-bold">*</span>
            </label>
            <div className="relative">
              <input
                type={showKey ? 'text' : 'password'}
                value={apiKey}
                onChange={(e) => {
                  setApiKey(e.target.value)
                  setErrorMsg(null)
                  setSuccessMsg(null)
                }}
                placeholder={
                  providers.find((p) => p.id === provider)?.keyPlaceholder || 'Enter your API Key'
                }
                required={provider !== 'ollama'}
                className="w-full bg-gray-950 border border-gray-700 focus:border-indigo-500 rounded-xl pl-3.5 pr-10 py-2.5 text-xs font-mono text-gray-100 placeholder-gray-600 focus:outline-none focus:ring-1 focus:ring-indigo-500 transition"
              />
              <button
                type="button"
                onClick={() => setShowKey(!showKey)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-200 transition p-1"
                title={showKey ? 'Hide API Key' : 'Show API Key'}
              >
                {showKey ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              </button>
            </div>
          </div>

          {/* Advanced Endpoint Config: Base URL & Model */}
          <div className="grid grid-cols-2 gap-3 pt-1">
            <div className="space-y-1">
              <label className="text-gray-400 font-medium flex items-center space-x-1">
                <Server className="h-3 w-3 text-cyan-400" />
                <span>Base URL:</span>
              </label>
              <input
                type="text"
                value={baseUrl}
                onChange={(e) => setBaseUrl(e.target.value)}
                placeholder="https://api.deepseek.com/v1"
                className="w-full bg-gray-950 border border-gray-800 focus:border-indigo-500 rounded-lg px-3 py-2 text-xs font-mono text-gray-300 focus:outline-none"
              />
            </div>
            <div className="space-y-1">
              <label className="text-gray-400 font-medium flex items-center space-x-1">
                <Cpu className="h-3 w-3 text-cyan-400" />
                <span>Model Identifier:</span>
              </label>
              <input
                type="text"
                value={modelName}
                onChange={(e) => setModelName(e.target.value)}
                placeholder="deepseek-chat"
                className="w-full bg-gray-950 border border-gray-800 focus:border-indigo-500 rounded-lg px-3 py-2 text-xs font-mono text-gray-300 focus:outline-none"
              />
            </div>
          </div>

          {/* Validation Feedback Banner */}
          {errorMsg && (
            <div className="p-3.5 rounded-xl border bg-rose-950/50 border-rose-800/80 text-rose-200 flex items-start space-x-2.5 text-xs shadow-inner">
              <AlertCircle className="h-4 w-4 shrink-0 text-rose-400 mt-0.5" />
              <div className="flex-1 leading-snug">
                <span className="font-bold block text-rose-300">Verification Failed:</span>
                <span>{errorMsg}</span>
              </div>
            </div>
          )}

          {successMsg && (
            <div className="p-3.5 rounded-xl border bg-emerald-950/50 border-emerald-800/80 text-emerald-200 flex items-start space-x-2.5 text-xs shadow-inner">
              <Check className="h-4 w-4 shrink-0 text-emerald-400 mt-0.5" />
              <div className="flex-1 leading-snug">
                <span className="font-bold block text-emerald-300">Success:</span>
                <span>{successMsg}</span>
              </div>
            </div>
          )}

          {/* Footer CTA */}
          <div className="pt-2">
            <button
              type="submit"
              disabled={isValidating || (provider !== 'ollama' && !apiKey.trim())}
              className="w-full py-3 px-4 rounded-xl bg-gradient-to-r from-indigo-600 via-indigo-500 to-cyan-600 hover:from-indigo-500 hover:to-cyan-500 text-white font-bold text-xs shadow-lg shadow-indigo-600/30 flex items-center justify-center space-x-2 transition disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isValidating ? (
                <>
                  <RefreshCw className="h-4 w-4 animate-spin text-cyan-200" />
                  <span>Verifying API Key Connectivity with Health Probe...</span>
                </>
              ) : (
                <>
                  <Check className="h-4 w-4 text-cyan-200" />
                  <span>Verify Key & Unlock Studio</span>
                  <ArrowRight className="h-4 w-4 ml-1 opacity-80" />
                </>
              )}
            </button>
            <p className="text-[11px] text-gray-500 text-center mt-2.5">
              The studio will run a live probe to confirm your key works before enabling environment
              synthesis.
            </p>
          </div>
        </form>
      </div>
    </div>
  )
}
