import React, { useState, useEffect } from 'react'
import { X, Key, Server, Cpu, Check, Save, Zap, AlertCircle, RefreshCw } from 'lucide-react'
import { LLMSettings } from '../types'
import { validateLLMSettings } from '../services/api'

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
      setValResult(res)
    } catch (e: any) {
      setValResult({ valid: false, message: e.message || 'Validation request failed' })
    } finally {
      setIsValidating(false)
    }
  }

  const handleSave = () => {
    onSaveSettings(localSettings)
    setSaved(true)
    setTimeout(() => {
      setSaved(false)
      onClose()
    }, 800)
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-sm p-4">
      <div className="w-full max-w-md bg-panel border border-panel-border rounded-2xl shadow-2xl overflow-hidden flex flex-col">
        {/* Header */}
        <div className="px-5 py-4 border-b border-panel-border flex items-center justify-between bg-gray-900/60">
          <div className="flex items-center space-x-2">
            <Key className="h-5 w-5 text-blue-400" />
            <h2 className="text-sm font-bold text-gray-100">
              BYOK & Model Settings
            </h2>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-lg text-gray-400 hover:text-white hover:bg-gray-800 transition"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-5 space-y-4 text-xs">
          {/* Provider Selector */}
          <div className="space-y-1">
            <label className="text-gray-400 font-medium block">
              Active LLM Compiler Provider:
            </label>
            <div className="grid grid-cols-2 gap-2">
              {[
                { id: 'mock', name: 'Mock Rules (Offline)' },
                { id: 'deepseek', name: 'DeepSeek (Official)' },
                { id: 'openai', name: 'OpenAI (GPT-4o)' },
                { id: 'anthropic', name: 'Anthropic (Claude)' },
                { id: 'ollama', name: 'Local Ollama / vLLM' },
              ].map((p) => (
                <button
                  key={p.id}
                  onClick={() => {
                    const defaultModel =
                      p.id === 'deepseek'
                        ? 'deepseek-chat'
                        : p.id === 'openai'
                        ? 'gpt-4o'
                        : p.id === 'anthropic'
                        ? 'claude-3-5-sonnet-20241022'
                        : p.id === 'ollama'
                        ? 'llama3'
                        : 'mock-rules'
                    setLocalSettings({
                      ...localSettings,
                      provider: p.id as any,
                      modelName: defaultModel,
                    })
                    setValResult(null)
                  }}
                  className={`p-2 rounded-lg border text-left font-medium transition text-xs ${
                    localSettings.provider === p.id
                      ? 'bg-blue-950/60 border-blue-500 text-blue-200'
                      : 'bg-gray-950 border-gray-800 text-gray-400 hover:text-gray-200'
                  }`}
                >
                  {p.name}
                </button>
              ))}
            </div>
          </div>

          {/* DeepSeek API Key */}
          {localSettings.provider === 'deepseek' && (
            <div className="space-y-1">
              <label className="text-gray-400 font-medium block">DeepSeek API Key:</label>
              <input
                type="password"
                value={localSettings.deepseekApiKey || ''}
                onChange={(e) => {
                  setLocalSettings({ ...localSettings, deepseekApiKey: e.target.value })
                  setValResult(null)
                }}
                placeholder="sk-..."
                className="w-full bg-gray-950 border border-gray-800 rounded-lg p-2 text-xs font-mono text-gray-200 focus:outline-none focus:border-blue-500"
              />
            </div>
          )}

          {/* OpenAI API Key */}
          {localSettings.provider === 'openai' && (
            <div className="space-y-1">
              <label className="text-gray-400 font-medium block">OpenAI API Key:</label>
              <input
                type="password"
                value={localSettings.openaiApiKey}
                onChange={(e) => {
                  setLocalSettings({ ...localSettings, openaiApiKey: e.target.value })
                  setValResult(null)
                }}
                placeholder="sk-proj-..."
                className="w-full bg-gray-950 border border-gray-800 rounded-lg p-2 text-xs font-mono text-gray-200 focus:outline-none focus:border-blue-500"
              />
            </div>
          )}

          {/* Anthropic API Key */}
          {localSettings.provider === 'anthropic' && (
            <div className="space-y-1">
              <label className="text-gray-400 font-medium block">Anthropic API Key:</label>
              <input
                type="password"
                value={localSettings.anthropicApiKey}
                onChange={(e) => {
                  setLocalSettings({ ...localSettings, anthropicApiKey: e.target.value })
                  setValResult(null)
                }}
                placeholder="sk-ant-..."
                className="w-full bg-gray-950 border border-gray-800 rounded-lg p-2 text-xs font-mono text-gray-200 focus:outline-none focus:border-blue-500"
              />
            </div>
          )}

          {/* Ollama Base URL */}
          {localSettings.provider === 'ollama' && (
            <div className="space-y-1">
              <label className="text-gray-400 font-medium block">Ollama / vLLM Base URL:</label>
              <input
                type="text"
                value={localSettings.ollamaBaseUrl}
                onChange={(e) => {
                  setLocalSettings({ ...localSettings, ollamaBaseUrl: e.target.value })
                  setValResult(null)
                }}
                placeholder="http://localhost:11434/v1"
                className="w-full bg-gray-950 border border-gray-800 rounded-lg p-2 text-xs font-mono text-gray-200 focus:outline-none focus:border-blue-500"
              />
            </div>
          )}

          {/* Model Name */}
          <div className="space-y-1">
            <label className="text-gray-400 font-medium block">Model Identifier:</label>
            <input
              type="text"
              value={localSettings.modelName}
              onChange={(e) => {
                setLocalSettings({ ...localSettings, modelName: e.target.value })
                setValResult(null)
              }}
              placeholder="gpt-4o / claude-3-5-sonnet-20241022 / llama3"
              className="w-full bg-gray-950 border border-gray-800 rounded-lg p-2 text-xs font-mono text-gray-200 focus:outline-none focus:border-blue-500"
            />
          </div>

          {/* Test Connection Button & Result */}
          <div className="space-y-2 pt-1">
            <div className="flex items-center justify-between">
              <button
                type="button"
                onClick={handleTestConnection}
                disabled={isValidating}
                className="flex items-center space-x-1.5 px-3 py-1.5 bg-gray-900 hover:bg-gray-800 border border-gray-700 text-gray-200 rounded-lg font-medium transition text-xs disabled:opacity-50"
              >
                {isValidating ? (
                  <RefreshCw className="h-3 w-3 animate-spin text-blue-400" />
                ) : (
                  <Zap className="h-3 w-3 text-amber-400" />
                )}
                <span>{isValidating ? 'Testing...' : 'Test Connection'}</span>
              </button>

              <span className="text-[11px] text-gray-500">
                {localSettings.provider.toUpperCase()} Mode
              </span>
            </div>

            {valResult && (
              <div
                className={`p-2.5 rounded-lg border flex items-start space-x-2 text-[11px] ${
                  valResult.valid
                    ? 'bg-emerald-950/50 border-emerald-800 text-emerald-300'
                    : 'bg-rose-950/50 border-rose-800 text-rose-300'
                }`}
              >
                {valResult.valid ? (
                  <Check className="h-4 w-4 text-emerald-400 flex-shrink-0 mt-0.5" />
                ) : (
                  <AlertCircle className="h-4 w-4 text-rose-400 flex-shrink-0 mt-0.5" />
                )}
                <span>{valResult.message}</span>
              </div>
            )}
          </div>

          <div className="p-2.5 rounded-lg bg-gray-950/60 border border-gray-800 text-[11px] text-gray-400">
            Keys are stored locally in your browser session. Antigravity never touches or meters
            your tokens.
          </div>
        </div>

        {/* Footer */}
        <div className="px-5 py-3 border-t border-panel-border bg-gray-900/60 flex items-center justify-between">
          <button
            onClick={onClose}
            className="px-3 py-1.5 rounded-lg text-gray-400 hover:text-white transition text-xs"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            className="flex items-center space-x-1.5 px-4 py-1.5 bg-blue-600 hover:bg-blue-500 text-white font-medium rounded-lg text-xs transition"
          >
            {saved ? (
              <>
                <Check className="h-3.5 w-3.5 text-emerald-300" />
                <span>Saved!</span>
              </>
            ) : (
              <>
                <Save className="h-3.5 w-3.5" />
                <span>Save Settings</span>
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  )
}
