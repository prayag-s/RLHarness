import React, { useState, useEffect } from 'react'
import {
  X,
  Archive,
  RefreshCw,
  Download,
  Copy,
  Check,
  Terminal,
  FileCode,
  Layers,
  Sparkles,
} from 'lucide-react'
import { BundleMetadata } from '../types'
import { fetchBundles } from '../services/api'
import { copyText } from '../utils/clipboard'

interface RegistryModalProps {
  isOpen: boolean
  onClose: () => void
}

export const RegistryModal: React.FC<RegistryModalProps> = ({ isOpen, onClose }) => {
  const [bundles, setBundles] = useState<BundleMetadata[]>([])
  const [loading, setLoading] = useState(false)
  const [copiedKey, setCopiedKey] = useState<string | null>(null)
  const [selectedBundle, setSelectedBundle] = useState<BundleMetadata | null>(null)

  const loadBundles = async () => {
    setLoading(true)
    try {
      const data = await fetchBundles()
      setBundles(data)
      if (data.length > 0 && !selectedBundle) {
        setSelectedBundle(data[0])
      }
    } catch (e) {
      console.error('Failed to load bundles', e)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (isOpen) {
      loadBundles()
    }
  }, [isOpen])

  if (!isOpen) return null

  const handleCopy = (text: string, key: string) => {
    void copyText(text).then((ok) => {
      if (!ok) return
      setCopiedKey(key)
      setTimeout(() => setCopiedKey(null), 2000)
    })
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-sm p-4">
      <div className="w-full max-w-4xl h-[80vh] bg-gray-900 border border-gray-800 rounded-2xl shadow-2xl flex flex-col overflow-hidden text-gray-200">
        {/* Header */}
        <div className="h-14 border-b border-gray-800 bg-gray-950 px-5 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="h-8 w-8 rounded-lg bg-indigo-950/80 border border-indigo-700/50 flex items-center justify-center text-indigo-400">
              <Archive className="h-4 w-4" />
            </div>
            <div>
              <h2 className="text-sm font-bold text-gray-100">Standalone RunBundle Registry</h2>
              <p className="text-[11px] text-gray-400">Catalog of self-contained ONNX policy distributions</p>
            </div>
          </div>

          <div className="flex items-center space-x-2">
            <button
              onClick={loadBundles}
              className="p-1.5 rounded-lg text-gray-400 hover:text-white hover:bg-gray-800 transition"
              title="Refresh Registry"
            >
              <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
            </button>
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg text-gray-400 hover:text-white hover:bg-gray-800 transition"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>

        {/* Content Body */}
        <div className="flex-1 flex overflow-hidden">
          {/* Left: Bundle List */}
          <div className="w-80 border-r border-gray-800 bg-gray-950/50 overflow-y-auto p-3 space-y-2">
            <div className="text-[11px] font-bold uppercase tracking-wider text-gray-500 mb-2">
              Packaged Bundles ({bundles.length})
            </div>

            {bundles.length === 0 ? (
              <div className="p-4 text-center text-xs text-gray-500 border border-dashed border-gray-800 rounded-xl">
                No exported bundles cataloged yet. Train and export a policy first!
              </div>
            ) : (
              bundles.map((b) => {
                const isSelected = selectedBundle?.bundle_id === b.bundle_id
                const sizeMb = (b.onnx_size_bytes / 1024).toFixed(1)
                return (
                  <button
                    key={b.bundle_id}
                    onClick={() => setSelectedBundle(b)}
                    className={`w-full text-left p-3 rounded-xl border transition flex flex-col space-y-1 ${
                      isSelected
                        ? 'bg-indigo-950/60 border-indigo-500 text-white shadow-md'
                        : 'bg-gray-900/60 border-gray-800 text-gray-400 hover:border-gray-700 hover:text-gray-200'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-semibold text-xs text-gray-200 truncate">{b.name}</span>
                      <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-emerald-950 border border-emerald-800/40 text-emerald-400">
                        {b.status}
                      </span>
                    </div>
                    <div className="text-[10px] text-gray-500 font-mono truncate">
                      {b.task_id} • {sizeMb} KB ONNX
                    </div>
                    <div className="text-[10px] text-gray-500">{new Date(b.created_at).toLocaleString()}</div>
                  </button>
                )
              })
            )}
          </div>

          {/* Right: Selected Bundle Inspector */}
          <div className="flex-1 p-5 overflow-y-auto bg-gray-900 space-y-4">
            {selectedBundle ? (
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-base font-bold text-gray-100">{selectedBundle.name}</h3>
                    <p className="text-xs font-mono text-indigo-400">{selectedBundle.task_id}</p>
                  </div>

                  <a
                    href={`/api/v4/sessions/${selectedBundle.task_id}/download`}
                    download
                    className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-bold transition shadow-md"
                  >
                    <Download className="h-3.5 w-3.5" />
                    <span>Download .zip</span>
                  </a>
                </div>

                {/* Bundle Spec Cards */}
                <div className="grid grid-cols-3 gap-3">
                  <div className="p-3 rounded-xl bg-gray-950 border border-gray-800">
                    <span className="text-[11px] text-gray-500 block">ONNX Runtime Size</span>
                    <span className="text-sm font-bold font-mono text-cyan-400">
                      {(selectedBundle.onnx_size_bytes / 1024).toFixed(1)} KB
                    </span>
                  </div>
                  <div className="p-3 rounded-xl bg-gray-950 border border-gray-800">
                    <span className="text-[11px] text-gray-500 block">Distribution Format</span>
                    <span className="text-sm font-bold font-mono text-purple-400">
                      .runbundle ({selectedBundle.archive_type})
                    </span>
                  </div>
                  <div className="p-3 rounded-xl bg-gray-950 border border-gray-800">
                    <span className="text-[11px] text-gray-500 block">Created At</span>
                    <span className="text-xs font-mono text-gray-300">
                      {new Date(selectedBundle.created_at).toLocaleDateString()}
                    </span>
                  </div>
                </div>

                {/* Standalone Daemon CLI Commands */}
                <div className="p-4 rounded-xl bg-gray-950 border border-gray-800 space-y-2">
                  <div className="flex items-center space-x-2 text-xs font-bold text-gray-300">
                    <Terminal className="h-4 w-4 text-emerald-400" />
                    <span>Local Zero-Dependency Execution</span>
                  </div>
                  <p className="text-xs text-gray-400">
                    Extract the runbundle and run inference without Python virtualenv or backend server:
                  </p>

                  <div className="p-2.5 rounded-lg bg-gray-900 border border-gray-800 flex items-center justify-between font-mono text-xs text-gray-300">
                    <code>python run_local.py --sample</code>
                    <button
                      onClick={() => handleCopy('python run_local.py --sample', 'sample')}
                      className="p-1 text-gray-400 hover:text-white"
                      title="Copy command"
                    >
                      {copiedKey === 'sample' ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
                    </button>
                  </div>

                  <div className="p-2.5 rounded-lg bg-gray-900 border border-gray-800 flex items-center justify-between font-mono text-xs text-gray-300">
                    <code>python run_local.py --serve --port 9000</code>
                    <button
                      onClick={() => handleCopy('python run_local.py --serve --port 9000', 'serve')}
                      className="p-1 text-gray-400 hover:text-white"
                      title="Copy command"
                    >
                      {copiedKey === 'serve' ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
                    </button>
                  </div>
                </div>
              </div>
            ) : (
              <div className="h-full flex items-center justify-center text-xs text-gray-500">
                Select a bundle from the list to view its architecture and CLI runner.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
