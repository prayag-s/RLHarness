import React, { useState, useEffect } from 'react'
import { X, Key, Server, Cpu, Check, Save, Sparkles, AlertCircle, RefreshCw } from 'lucide-react'
import { LLMSettings } from '../types'
import { validateLLMSettings, saveAndVerifyLLMSettings } from '../services/api'

interface SettingsModalProps {
  isOpen: boolean
  onClose: () => void
  settings: LLMSettings
  onSaveSettings: (settings: LLMSettings) => void
}

export const SettingsModal: React.FC<SettingsModalProps> = ({
  isOpen,
  onClose,
  settings,
  onSaveSettings,
}) => {
  const [localSettings, setLocalSettings] = useState<LLMSettings>(settings)
  const [saved, setSaved] = useState(false)
  const [isValidating, setIsValidating] = useState(false)
  const [valResult, setValResult] = useState<{ valid: boolean; message: string } | null>(null)

  useEffect(() => {
    setLocalSettings(settings)
    setValResult(null)
  }, [settings, isOpen])

  if (!isOpen) return null

  const handleTestConnection = async () => {
    setIsValidating(true)
    setValResult(null)
    try {
      const res = await validateLLMSettings(localSettings)
      setValResult({
        valid: res.valid || res.success,
        message: res.message || (res.success ? 'Connectivity verified!' : 'Connection failed'),
      })
    } catch (e: any) {
      setValResult({ valid: false, message: e.message || 'Validation request failed' })
    } finally {
      setIsValidating(false)
    }
  }

  const handleSave = async () => {
    setIsValidating(true)
    setValResult(null)
    try {
      const res = await saveAndVerifyLLMSettings(localSettings)
      if (res.is_verified || res.success) {
        onSaveSettings(localSettings)
        setSaved(true)
        setTimeout(() => {
          setSaved(false)
          onClose()
        }, 700)
      } else {
        setValResult({
          valid: false,
          message: res.error || 'LLM connection check failed. Settings not applied.',
        })
      }
    } catch (err: any) {
      setValResult({
        valid: false,
        message: err.message || 'Failed to verify and activate LLM settings.',
      })
    } finally {
      setIsValidating(false)
    }
  }

  const providers = [
    {
      id: 'deepseek' as const,
      name: 'DeepSeek V3 / R1 (Official)',
      defaultBaseUrl: 'https://api.deepseek.com/v1',
      defaultModel: 'deepseek-chat',
      badge: 'Recommended',
    },
    {
      id: 'anthropic' as const,
      name: 'Anthropic (Claude 3.7)',
      defaultBaseUrl: 'https://api.anthropic.com/v1',
      defaultModel: 'claude-3-7-sonnet-20250219',
      badge: 'Fast & Smart',
    },
    {
      id: 'openai' as const,
      name: 'OpenAI (GPT-4o)',
      defaultBaseUrl: 'https://api.openai.com/v1',
      defaultModel: 'gpt-4o',
      badge: 'Popular',
    },
    {
      id: 'ollama' as const,
      name: 'Ollama (Local / vLLM)',
      defaultBaseUrl: 'http://localhost:11434/v1',
      defaultModel: 'llama3.1',
      badge: 'Private',
    },
  ]

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-sm p-4">
      <div className="w-full max-w-lg bg-gray-900 border border-gray-800 rounded-2xl shadow-2xl overflow-hidden flex flex-col text-gray-200">
        {/* Header */}
        <div className="px-5 py-4 border-b border-gray-800 flex items-center justify-between bg-gray-950">
          <div className="flex items-center space-x-2.5">
            <div className="h-8 w-8 rounded-lg bg-indigo-950/80 border border-indigo-700/50 flex items-center justify-center text-indigo-400">
              <Sparkles className="h-4 w-4" />
            </div>
            <div>
              <h2 className="text-sm font-bold text-gray-100">LLM Synthesis & Auto-Tuner Settings</h2>
              <p className="text-[11px] text-gray-400">Configure BYOK multi-provider credentials</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-lg text-gray-400 hover:text-white hover:bg-gray-800 transition"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Content */}
        <div className="p-5 space-y-4 text-xs">
          {/* Provider Selector Grid */}
          <div className="space-y-1.5">
            <label className="text-gray-400 font-semibold block">Select Active Engine:</label>
            <div className="grid grid-cols-2 gap-2">
              {providers.map((p) => {
                const isSelected = localSettings.provider === p.id
                return (
                  <button
                    key={p.id}
                    onClick={() => {
                      setLocalSettings({
                        ...localSettings,
                        provider: p.id,
                        baseUrl: p.defaultBaseUrl,
                        modelName: p.defaultModel,
                      })
                      setValResult(null)
                    }}
                    className={`p-2.5 rounded-xl border text-left transition flex flex-col justify-between ${
                      isSelected
                        ? 'bg-indigo-950/70 border-indigo-500 text-white shadow-md'
                        : 'bg-gray-950/60 border-gray-800 text-gray-400 hover:border-gray-700 hover:text-gray-200'
                    }`}
                  >
                    <span className="font-semibold text-xs text-gray-200">{p.name}</span>
                    <span className="text-[10px] text-indigo-400 font-mono mt-1">{p.badge}</span>
                  </button>
                )
              })}
            </div>
          </div>

          {/* API Key Input (if not fallback) */}
          {localSettings.provider !== 'fallback' && (
            <div className="space-y-1">
              <label className="text-gray-300 font-medium flex items-center space-x-1">
                <Key className="h-3 w-3 text-cyan-400" />
                <span>API Key:</span>
              </label>
              <input
                type="password"
                value={localSettings.apiKey || ''}
                onChange={(e) => setLocalSettings({ ...localSettings, apiKey: e.target.value })}
                placeholder={
                  localSettings.provider === 'deepseek'
                    ? 'sk-...'
                    : localSettings.provider === 'anthropic'
                    ? 'sk-ant-...'
                    : localSettings.provider === 'openai'
                    ? 'sk-...'
                    : 'Optional for local Ollama'
                }
                className="w-full bg-gray-950 border border-gray-800 focus:border-indigo-500 rounded-lg px-3 py-2 text-xs font-mono text-gray-200 focus:outline-none"
              />
            </div>
          )}

          {/* Base URL (DeepSeek / Ollama) */}
          {localSettings.provider !== 'fallback' && (
            <div className="space-y-1">
              <label className="text-gray-300 font-medium flex items-center space-x-1">
                <Server className="h-3 w-3 text-cyan-400" />
                <span>Base URL:</span>
              </label>
              <input
                type="text"
                value={localSettings.baseUrl || ''}
                onChange={(e) => setLocalSettings({ ...localSettings, baseUrl: e.target.value })}
                placeholder="https://api.deepseek.com/v1"
                className="w-full bg-gray-950 border border-gray-800 focus:border-indigo-500 rounded-lg px-3 py-2 text-xs font-mono text-gray-200 focus:outline-none"
              />
            </div>
          )}

          {/* Model Name */}
          <div className="space-y-1">
            <label className="text-gray-300 font-medium flex items-center space-x-1">
              <Cpu className="h-3 w-3 text-cyan-400" />
              <span>Model Identifier:</span>
            </label>
            <input
              type="text"
              value={localSettings.modelName || ''}
              onChange={(e) => setLocalSettings({ ...localSettings, modelName: e.target.value })}
              placeholder="e.g. deepseek-chat, gpt-4o, claude-3-7-sonnet"
              className="w-full bg-gray-950 border border-gray-800 focus:border-indigo-500 rounded-lg px-3 py-2 text-xs font-mono text-gray-200 focus:outline-none"
            />
          </div>

          {/* Validation Feedback */}
          {valResult && (
            <div
              className={`p-3 rounded-lg border flex items-start space-x-2 text-xs ${
                valResult.valid
                  ? 'bg-emerald-950/40 border-emerald-800/60 text-emerald-300'
                  : 'bg-rose-950/40 border-rose-800/60 text-rose-300'
              }`}
            >
              {valResult.valid ? (
                <Check className="h-4 w-4 shrink-0 text-emerald-400 mt-0.5" />
              ) : (
                <AlertCircle className="h-4 w-4 shrink-0 text-rose-400 mt-0.5" />
              )}
              <span className="leading-snug">{valResult.message}</span>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-5 py-3 border-t border-gray-800 bg-gray-950 flex items-center justify-between">
          <button
            onClick={handleTestConnection}
            disabled={isValidating}
            className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg border border-gray-700 bg-gray-800 hover:bg-gray-700 text-gray-300 text-xs font-medium transition disabled:opacity-50"
          >
            {isValidating ? (
              <RefreshCw className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <Check className="h-3.5 w-3.5 text-cyan-400" />
            )}
            <span>{isValidating ? 'Testing...' : 'Test Connection'}</span>
          </button>

          <div className="flex items-center space-x-2">
            <button
              onClick={onClose}
              className="px-3 py-1.5 rounded-lg text-gray-400 hover:text-gray-200 text-xs font-medium transition"
            >
              Cancel
            </button>
            <button
              onClick={handleSave}
              className="flex items-center space-x-1.5 px-4 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-bold transition shadow-md"
            >
              <Save className="h-3.5 w-3.5" />
              <span>{saved ? 'Saved!' : 'Save & Apply'}</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
