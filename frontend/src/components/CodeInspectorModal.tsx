import React, { useState } from 'react'
import { X, Code, Globe, Copy, Check, FileCode } from 'lucide-react'
import { copyText } from '../utils/clipboard'

interface CodeInspectorModalProps {
  isOpen: boolean
  onClose: () => void
  envModuleCode?: string
  arenaHtml?: string
}

export const CodeInspectorModal: React.FC<CodeInspectorModalProps> = ({
  isOpen,
  onClose,
  envModuleCode = '# No env_module.py synthesized yet.',
  arenaHtml = '<!-- No arena.html synthesized yet. -->',
}) => {
  const [activeTab, setActiveTab] = useState<'env' | 'arena'>('env')
  const [copied, setCopied] = useState(false)

  if (!isOpen) return null

  const currentCode = activeTab === 'env' ? envModuleCode : arenaHtml

  const handleCopy = () => {
    void copyText(currentCode).then((ok) => {
      if (!ok) return
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-sm p-4">
      <div className="w-full max-w-5xl h-[85vh] bg-gray-900 border border-gray-800 rounded-2xl shadow-2xl flex flex-col overflow-hidden">
        {/* Modal Header */}
        <div className="h-14 border-b border-gray-800 bg-gray-950 px-5 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="h-8 w-8 rounded-lg bg-indigo-950/80 border border-indigo-700/50 flex items-center justify-center text-indigo-400">
              <Code className="h-4 w-4" />
            </div>
            <div>
              <h2 className="text-sm font-bold text-gray-200">Synthesized JIT Artifact Inspector</h2>
              <p className="text-[11px] text-gray-400">Zero-Adapter Generated Code & Arena Document</p>
            </div>
          </div>

          {/* Tab Selector & Copy Button */}
          <div className="flex items-center space-x-3">
            <div className="flex items-center bg-gray-900 border border-gray-800 rounded-lg p-0.5 text-xs">
              <button
                onClick={() => setActiveTab('env')}
                className={`flex items-center space-x-1.5 px-3 py-1 rounded-md font-medium transition ${
                  activeTab === 'env'
                    ? 'bg-indigo-600 text-white shadow-sm'
                    : 'text-gray-400 hover:text-gray-200'
                }`}
              >
                <FileCode className="h-3.5 w-3.5 text-cyan-400" />
                <span>env_module.py</span>
              </button>
              <button
                onClick={() => setActiveTab('arena')}
                className={`flex items-center space-x-1.5 px-3 py-1 rounded-md font-medium transition ${
                  activeTab === 'arena'
                    ? 'bg-indigo-600 text-white shadow-sm'
                    : 'text-gray-400 hover:text-gray-200'
                }`}
              >
                <Globe className="h-3.5 w-3.5 text-amber-400" />
                <span>arena.html</span>
              </button>
            </div>

            <button
              onClick={handleCopy}
              className="flex items-center space-x-1 text-xs px-2.5 py-1 rounded-lg bg-gray-800 hover:bg-gray-700 border border-gray-700 text-gray-300 transition"
              title="Copy current code to clipboard"
            >
              {copied ? (
                <>
                  <Check className="h-3.5 w-3.5 text-emerald-400" />
                  <span className="text-emerald-400">Copied</span>
                </>
              ) : (
                <>
                  <Copy className="h-3.5 w-3.5" />
                  <span>Copy</span>
                </>
              )}
            </button>

            <button
              onClick={onClose}
              className="p-1.5 text-gray-400 hover:text-gray-200 rounded-lg hover:bg-gray-800 transition"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>

        {/* Code Content */}
        <div className="flex-1 overflow-auto bg-gray-950 p-4 font-mono text-xs text-gray-300 leading-relaxed scrollbar-thin">
          <pre className="whitespace-pre">{currentCode}</pre>
        </div>
      </div>
    </div>
  )
}
