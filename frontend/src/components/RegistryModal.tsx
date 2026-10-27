import React, { useState, useEffect } from 'react'
import {
  X,
  Archive,
  RefreshCw,
  UploadCloud,
  Check,
  Copy,
  Terminal,
  Activity,
  Layers,
  FileCode,
  Trash2,
  ExternalLink,
  ChevronRight,
  ChevronDown,
} from 'lucide-react'
import { BundleMetadata, BundleDetails } from '../types'
import {
  fetchBundles,
  fetchBundleDetails,
  importBundle,
  deleteBundle,
} from '../services/api'

interface RegistryModalProps {
  isOpen: boolean
  onClose: () => void
  onLoadBundle: (taskConfig: any) => void
}

export const RegistryModal: React.FC<RegistryModalProps> = ({
  isOpen,
  onClose,
  onLoadBundle,
}) => {
  const [bundles, setBundles] = useState<BundleMetadata[]>([])
  const [loading, setLoading] = useState(false)
  const [selectedDetails, setSelectedDetails] = useState<BundleDetails | null>(null)
  const [inspectingId, setInspectingId] = useState<string | null>(null)
  const [importPath, setImportPath] = useState('')
  const [isImporting, setIsImporting] = useState(false)
  const [copiedKey, setCopiedKey] = useState<string | null>(null)
  const [feedbackMsg, setFeedbackMsg] = useState<{ type: 'success' | 'error'; text: string } | null>(null)

  const loadBundles = async () => {
    setLoading(true)
    try {
      const data = await fetchBundles()
      setBundles(data)
    } catch (e: any) {
      setFeedbackMsg({ type: 'error', text: 'Failed to load bundles: ' + e.message })
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (isOpen) {
      loadBundles()
      setSelectedDetails(null)
      setInspectingId(null)
      setFeedbackMsg(null)
    }
  }, [isOpen])

  if (!isOpen) return null

  const handleInspect = async (b: BundleMetadata) => {
    if (inspectingId === b.bundle_id) {
      setInspectingId(null)
      setSelectedDetails(null)
      return
    }
    setInspectingId(b.bundle_id)
    try {
      const details = await fetchBundleDetails(b.bundle_id)
      setSelectedDetails(details)
    } catch (e: any) {
      setFeedbackMsg({ type: 'error', text: `Failed inspecting ${b.bundle_id}: ${e.message}` })
    }
  }

  const handleImport = async () => {
    if (!importPath.trim()) return
    setIsImporting(true)
    setFeedbackMsg(null)
    try {
      const res = await importBundle(importPath.trim(), false)
      setFeedbackMsg({ type: 'success', text: `Successfully imported ${res.task_config.name || res.bundle_dir} into registry!` })
      setImportPath('')
      await loadBundles()
    } catch (e: any) {
      setFeedbackMsg({ type: 'error', text: `Import failed: ${e.message}` })
    } finally {
      setIsImporting(false)
    }
  }

  const handleActivate = async (b: BundleMetadata) => {
    try {
      const res = await importBundle(b.bundle_path, true)
      onLoadBundle(res.task_config)
      onClose()
    } catch (e: any) {
      setFeedbackMsg({ type: 'error', text: `Failed to activate: ${e.message}` })
    }
  }

  const handleDelete = async (bundleId: string, e: React.MouseEvent) => {
    e.stopPropagation()
    if (!confirm(`Are you sure you want to delete bundle '${bundleId}'?`)) return
    try {
      await deleteBundle(bundleId)
      setBundles(bundles.filter((b) => b.bundle_id !== bundleId))
      if (inspectingId === bundleId) {
        setInspectingId(null)
        setSelectedDetails(null)
      }
      setFeedbackMsg({ type: 'success', text: `Bundle '${bundleId}' deleted.` })
    } catch (err: any) {
      setFeedbackMsg({ type: 'error', text: `Delete failed: ${err.message}` })
    }
  }

  const copyToClipboard = (text: string, key: string) => {
    navigator.clipboard.writeText(text)
    setCopiedKey(key)
    setTimeout(() => setCopiedKey(null), 1500)
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
      <div className="w-full max-w-4xl max-h-[90vh] bg-panel border border-panel-border rounded-2xl shadow-2xl overflow-hidden flex flex-col">
        {/* Header */}
        <div className="px-6 py-4 border-b border-panel-border flex items-center justify-between bg-gray-900/70">
          <div className="flex items-center space-x-3">
            <div className="p-2 rounded-xl bg-blue-950/80 border border-blue-800/50">
              <Archive className="h-5 w-5 text-blue-400" />
            </div>
            <div>
              <h2 className="text-sm font-bold text-gray-100 flex items-center space-x-2">
                <span>Team Policy Registry & RunBundle Hub</span>
                <span className="text-[10px] px-2 py-0.5 rounded-full bg-blue-900/50 text-blue-300 font-mono border border-blue-700/40">
                  {bundles.length} Registered
                </span>
              </h2>
              <p className="text-xs text-gray-400">
                Shareable, zero-dependency policy archives with embedded ONNX runtime and live drift monitoring.
              </p>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <button
              onClick={loadBundles}
              disabled={loading}
              title="Refresh Registry"
              className="p-1.5 rounded-lg text-gray-400 hover:text-white hover:bg-gray-800 transition disabled:opacity-50"
            >
              <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
            </button>
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg text-gray-400 hover:text-white hover:bg-gray-800 transition"
            >
              <X className="h-5 w-5" />
            </button>
          </div>
        </div>

        {/* Feedback message banner */}
        {feedbackMsg && (
          <div
            className={`px-6 py-2.5 text-xs font-medium flex items-center justify-between ${
              feedbackMsg.type === 'success'
                ? 'bg-emerald-950/60 border-b border-emerald-800/50 text-emerald-300'
                : 'bg-rose-950/60 border-b border-rose-800/50 text-rose-300'
            }`}
          >
            <span>{feedbackMsg.text}</span>
            <button
              onClick={() => setFeedbackMsg(null)}
              className="text-gray-400 hover:text-white"
            >
              &times;
            </button>
          </div>
        )}

        {/* Import Bar */}
        <div className="px-6 py-3 border-b border-panel-border bg-gray-950/60 flex items-center space-x-3">
          <UploadCloud className="h-4 w-4 text-gray-400 flex-shrink-0" />
          <input
            type="text"
            value={importPath}
            onChange={(e) => setImportPath(e.target.value)}
            placeholder="Import archive path (e.g. exported_bundles/csv_task.runbundle.zip or absolute path)..."
            className="flex-1 bg-gray-900 border border-gray-800 rounded-lg px-3 py-1.5 text-xs text-gray-200 placeholder-gray-500 font-mono focus:outline-none focus:border-blue-500"
          />
          <button
            onClick={handleImport}
            disabled={isImporting || !importPath.trim()}
            className="flex items-center space-x-1.5 px-3 py-1.5 bg-blue-600 hover:bg-blue-500 text-white rounded-lg text-xs font-medium transition disabled:opacity-50"
          >
            {isImporting ? (
              <RefreshCw className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <UploadCloud className="h-3.5 w-3.5" />
            )}
            <span>Import</span>
          </button>
        </div>

        {/* Bundle List */}
        <div className="flex-1 overflow-y-auto p-6 space-y-3">
          {bundles.length === 0 ? (
            <div className="p-12 text-center text-gray-500 space-y-2">
              <Archive className="h-10 w-10 mx-auto text-gray-600 stroke-1" />
              <p className="text-sm">No exported bundles found in registry search paths.</p>
              <p className="text-xs text-gray-600">
                Click "Export Owned Policy" in the top bar to package your first .runbundle.
              </p>
            </div>
          ) : (
            bundles.map((b) => {
              const isSelected = inspectingId === b.bundle_id
              return (
                <div
                  key={b.bundle_id}
                  className={`rounded-xl border transition overflow-hidden ${
                    isSelected
                      ? 'border-blue-500 bg-blue-950/20'
                      : 'border-panel-border bg-gray-950/40 hover:border-gray-700'
                  }`}
                >
                  {/* Summary Row */}
                  <div className="p-4 flex items-center justify-between">
                    <div className="space-y-1">
                      <div className="flex items-center space-x-2.5">
                        <span className="text-sm font-bold text-gray-100">{b.name}</span>
                        <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-gray-800 text-gray-300 border border-gray-700">
                          {b.task_id}
                        </span>
                        <span
                          className={`text-[10px] px-2 py-0.5 rounded-full font-medium ${
                            b.adapter_name.includes('Text')
                              ? 'bg-purple-950/60 text-purple-300 border border-purple-800/40'
                              : 'bg-cyan-950/60 text-cyan-300 border border-cyan-800/40'
                          }`}
                        >
                          {b.adapter_name}
                        </span>
                        <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-gray-900 text-gray-400 border border-gray-800">
                          {b.archive_type}
                        </span>
                      </div>
                      <div className="flex items-center space-x-4 text-xs text-gray-400">
                        <span>Created: {b.created_at.slice(0, 19).replace('T', ' ')}</span>
                        <span>•</span>
                        <span>ONNX: {(b.onnx_size_bytes / 1024).toFixed(1)} KB</span>
                        <span>•</span>
                        <span>DSL Terms: {b.dsl_terms_count}</span>
                        <span>•</span>
                        <span>Features: {b.feature_count}</span>
                      </div>
                    </div>

                    <div className="flex items-center space-x-2">
                      <button
                        onClick={() => handleInspect(b)}
                        className="flex items-center space-x-1 px-3 py-1.5 rounded-lg border border-gray-800 bg-gray-900 hover:bg-gray-800 text-xs text-gray-300 transition"
                      >
                        {isSelected ? (
                          <ChevronDown className="h-3.5 w-3.5 text-blue-400" />
                        ) : (
                          <ChevronRight className="h-3.5 w-3.5 text-gray-400" />
                        )}
                        <span>{isSelected ? 'Collapse' : 'Inspect'}</span>
                      </button>

                      <button
                        onClick={() => handleActivate(b)}
                        className="flex items-center space-x-1.5 px-3 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg text-xs font-semibold shadow-md shadow-emerald-600/20 transition"
                      >
                        <ExternalLink className="h-3.5 w-3.5" />
                        <span>Load into Workbench</span>
                      </button>

                      <button
                        onClick={(e) => handleDelete(b.bundle_id, e)}
                        title="Delete Bundle"
                        className="p-1.5 text-gray-500 hover:text-rose-400 hover:bg-rose-950/30 rounded-lg transition"
                      >
                        <Trash2 className="h-4 w-4" />
                      </button>
                    </div>
                  </div>

                  {/* Expanded Inspection Drawer */}
                  {isSelected && selectedDetails && (
                    <div className="border-t border-panel-border bg-gray-950/80 p-5 space-y-4 text-xs">
                      <div className="grid grid-cols-2 gap-4">
                        {/* Task Config & DSL Summary */}
                        <div className="space-y-2 p-3 bg-gray-900/60 rounded-xl border border-gray-800">
                          <div className="flex items-center space-x-2 text-gray-300 font-semibold">
                            <FileCode className="h-4 w-4 text-indigo-400" />
                            <span>Compiled Reward DSL Terms ({selectedDetails.metadata.dsl_terms_count})</span>
                          </div>
                          <div className="space-y-1.5 max-h-36 overflow-y-auto pr-1">
                            {((selectedDetails.reward_dsl as any)?.reward?.terms || []).map((term: any, idx: number) => (
                              <div key={idx} className="p-2 rounded bg-gray-950 border border-gray-800/80 text-[11px] font-mono">
                                <div className="flex justify-between items-center text-gray-300">
                                  <span className="font-semibold text-blue-300">{term.name}</span>
                                  <span className={term.value >= 0 ? 'text-emerald-400 font-bold' : 'text-rose-400 font-bold'}>
                                    {term.value > 0 ? `+${term.value}` : term.value}
                                  </span>
                                </div>
                                <div className="text-gray-400 text-[10px] truncate mt-0.5">
                                  when: {String(term.when)}
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>

                        {/* Drift Baseline Summary */}
                        <div className="space-y-2 p-3 bg-gray-900/60 rounded-xl border border-gray-800">
                          <div className="flex items-center space-x-2 text-gray-300 font-semibold">
                            <Activity className="h-4 w-4 text-emerald-400" />
                            <span>Unsupervised Drift Baseline</span>
                          </div>
                          <div className="text-[11px] space-y-1.5 text-gray-300 font-mono">
                            <div className="flex justify-between">
                              <span className="text-gray-500">Baseline Entropy Mean:</span>
                              <span>{(selectedDetails.drift_baseline as any)?.entropy_mean?.toFixed(4) || 'N/A'}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-500">Entropy Std:</span>
                              <span>{(selectedDetails.drift_baseline as any)?.entropy_std?.toFixed(4) || 'N/A'}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-500">Quantile Features:</span>
                              <span>{selectedDetails.metadata.baseline_quantiles_count} tracked</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-500">Standalone Daemon Support:</span>
                              <span className="text-emerald-400 font-semibold">Ready (--serve)</span>
                            </div>
                          </div>
                        </div>
                      </div>

                      {/* Standalone Daemon Serve Command */}
                      <div className="space-y-1.5 p-3 rounded-xl bg-gray-900/90 border border-gray-800">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center space-x-2 text-xs font-semibold text-gray-300">
                            <Terminal className="h-3.5 w-3.5 text-amber-400" />
                            <span>Production Zero-SB3 Microservice Command</span>
                          </div>
                          <button
                            onClick={() =>
                              copyToClipboard(
                                `python run_local.py --serve --port 9000`,
                                `daemon-${b.bundle_id}`
                              )
                            }
                            className="flex items-center space-x-1 text-[11px] text-gray-400 hover:text-white"
                          >
                            {copiedKey === `daemon-${b.bundle_id}` ? (
                              <Check className="h-3.5 w-3.5 text-emerald-400" />
                            ) : (
                              <Copy className="h-3.5 w-3.5" />
                            )}
                            <span>{copiedKey === `daemon-${b.bundle_id}` ? 'Copied' : 'Copy'}</span>
                          </button>
                        </div>
                        <div className="p-2 rounded bg-black border border-gray-800 font-mono text-emerald-400 text-xs">
                          python run_local.py --serve --port 9000
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )
            })
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-3 border-t border-panel-border bg-gray-900/70 flex items-center justify-between text-xs text-gray-400">
          <span>
            Policies run standalone with pure Python + ONNX Runtime. Zero token fees.
          </span>
          <button
            onClick={onClose}
            className="px-4 py-1.5 rounded-lg bg-gray-800 hover:bg-gray-700 text-gray-200 transition"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  )
}
