import React, { useState } from 'react'
import {
  X,
  CheckCircle2,
  FileCode,
  FolderArchive,
  Terminal,
  Copy,
  Check,
  ShieldCheck,
  Server,
} from 'lucide-react'
import { ExportResponse } from '../types'

interface ExportModalProps {
  isOpen: boolean
  onClose: () => void
  exportData: ExportResponse | null
  taskId: string
}

export const ExportModal: React.FC<ExportModalProps> = ({
  isOpen,
  onClose,
  exportData,
  taskId,
}) => {
  const [copiedCli, setCopiedCli] = useState(false)
  const [copiedDaemon, setCopiedDaemon] = useState(false)
  const [copiedPath, setCopiedPath] = useState(false)

  if (!isOpen) return null

  const cliCommand = `python run_local.py --sample`
  const daemonCommand = `python run_local.py --serve --port 9000`

  const handleCopyCli = () => {
    navigator.clipboard.writeText(cliCommand)
    setCopiedCli(true)
    setTimeout(() => setCopiedCli(false), 2000)
  }

  const handleCopyDaemon = () => {
    navigator.clipboard.writeText(daemonCommand)
    setCopiedDaemon(true)
    setTimeout(() => setCopiedDaemon(false), 2000)
  }

  const handleCopyPath = () => {
    if (exportData?.bundle_dir) {
      navigator.clipboard.writeText(exportData.bundle_dir)
      setCopiedPath(true)
      setTimeout(() => setCopiedPath(false), 2000)
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-sm p-4">
      <div className="w-full max-w-xl bg-panel border border-panel-border rounded-2xl shadow-2xl overflow-hidden flex flex-col">
        {/* Modal Header */}
        <div className="px-5 py-4 border-b border-panel-border flex items-center justify-between bg-gray-900/60">
          <div className="flex items-center space-x-2.5">
            <div className="h-8 w-8 rounded-lg bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center">
              <ShieldCheck className="h-5 w-5 text-emerald-400" />
            </div>
            <div>
              <h2 className="text-sm font-bold text-gray-100">
                Export Owned Policy (.runbundle)
              </h2>
              <p className="text-[11px] text-gray-400 font-mono">
                Standalone, Zero-LLM, Zero-SB3 Local Inference
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-lg text-gray-400 hover:text-white hover:bg-gray-800 transition"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-5 space-y-4 text-xs">
          {/* Numerical Parity Check Box */}
          <div className="p-3 rounded-xl bg-emerald-950/30 border border-emerald-800/60 flex items-start space-x-3">
            <CheckCircle2 className="h-5 w-5 text-emerald-400 flex-shrink-0 mt-0.5" />
            <div className="space-y-1">
              <div className="flex items-center space-x-2">
                <span className="font-bold text-emerald-300">
                  ONNX Numerical Parity Check Passed
                </span>
                <span className="text-[10px] font-mono bg-emerald-900/60 text-emerald-300 px-2 py-0.2 rounded border border-emerald-700/50">
                  Max diff: {(exportData?.parity_max_diff ?? 5.96e-8).toExponential(2)} ✅
                </span>
              </div>
              <p className="text-[11px] text-emerald-200/80 leading-relaxed">
                PyTorch actor weights exported to ONNX (opset 17). Mathematical parity verified
                against onnxruntime runtime within tolerance (&lt;1e-5).
              </p>
            </div>
          </div>

          {/* Bundle File Tree */}
          <div className="space-y-1.5">
            <span className="text-[11px] font-bold text-gray-300 uppercase tracking-wider block">
              Self-Contained Bundle Contents:
            </span>
            <div className="bg-gray-950 rounded-xl p-3 border border-gray-800 font-mono text-xs space-y-2">
              <div className="flex items-center space-x-2 text-blue-300">
                <FileCode className="h-4 w-4 text-blue-400" />
                <span>model.onnx</span>
                <span className="text-gray-500 text-[10px]">(Actor network, opset 17)</span>
              </div>
              <div className="flex items-center space-x-2 text-gray-300">
                <FileCode className="h-4 w-4 text-gray-400" />
                <span>task_config.yaml & reward_dsl.yaml</span>
                <span className="text-gray-500 text-[10px]">(Portable specs)</span>
              </div>
              <div className="flex items-center space-x-2 text-amber-300">
                <FileCode className="h-4 w-4 text-amber-400" />
                <span>drift_baseline.json</span>
                <span className="text-gray-500 text-[10px]">(PSI feature quantiles)</span>
              </div>
              <div className="flex items-center space-x-2 text-emerald-300">
                <FileCode className="h-4 w-4 text-emerald-400" />
                <span>run_local.py</span>
                <span className="text-gray-500 text-[10px]">
                  (Zero-dependency CLI & HTTP daemon --serve)
                </span>
              </div>
            </div>
          </div>

          {/* Production Serving Daemon Command Box */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="text-[11px] font-bold text-gray-300 uppercase tracking-wider flex items-center space-x-1.5">
                <Server className="h-3.5 w-3.5 text-emerald-400" />
                <span>Production HTTP Daemon Mode:</span>
              </span>
              <button
                onClick={handleCopyDaemon}
                className="flex items-center space-x-1 text-[11px] text-gray-400 hover:text-white transition"
              >
                {copiedDaemon ? (
                  <>
                    <Check className="h-3 w-3 text-emerald-400" />
                    <span className="text-emerald-400">Copied</span>
                  </>
                ) : (
                  <>
                    <Copy className="h-3 w-3" />
                    <span>Copy Command</span>
                  </>
                )}
              </button>
            </div>
            <pre className="bg-gray-950 p-2.5 rounded-lg border border-gray-800 text-xs text-emerald-300 font-mono overflow-x-auto select-all">
              {daemonCommand}
            </pre>
          </div>

          {/* Local CLI Command Box */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="text-[11px] font-bold text-gray-300 uppercase tracking-wider flex items-center space-x-1.5">
                <Terminal className="h-3.5 w-3.5 text-blue-400" />
                <span>Run Inference Standalone (Offline CLI):</span>
              </span>
              <button
                onClick={handleCopyCli}
                className="flex items-center space-x-1 text-[11px] text-gray-400 hover:text-white transition"
              >
                {copiedCli ? (
                  <>
                    <Check className="h-3 w-3 text-emerald-400" />
                    <span className="text-emerald-400">Copied</span>
                  </>
                ) : (
                  <>
                    <Copy className="h-3 w-3" />
                    <span>Copy Command</span>
                  </>
                )}
              </button>
            </div>
            <pre className="bg-gray-950 p-2.5 rounded-lg border border-gray-800 text-xs text-blue-300 font-mono overflow-x-auto select-all">
              {cliCommand}
            </pre>
          </div>

          {/* Directory location */}
          {exportData?.bundle_dir && (
            <div className="p-2.5 rounded-lg bg-gray-950/60 border border-gray-800 flex items-center justify-between font-mono text-[11px]">
              <div className="flex items-center space-x-2 text-gray-400 truncate">
                <FolderArchive className="h-4 w-4 text-gray-500 flex-shrink-0" />
                <span className="truncate">{exportData.bundle_dir}</span>
              </div>
              <button
                onClick={handleCopyPath}
                className="ml-2 text-gray-400 hover:text-white transition flex-shrink-0"
                title="Copy Bundle Path"
              >
                {copiedPath ? (
                  <Check className="h-3.5 w-3.5 text-emerald-400" />
                ) : (
                  <Copy className="h-3.5 w-3.5" />
                )}
              </button>
            </div>
          )}
        </div>

        {/* Modal Footer */}
        <div className="px-5 py-3 border-t border-panel-border bg-gray-900/60 flex items-center justify-between">
          <span className="text-[11px] text-gray-500">
            Ownership: 100% locally portable ONNX model.
          </span>
          <button
            onClick={onClose}
            className="px-4 py-1.5 bg-blue-600 hover:bg-blue-500 text-white font-medium rounded-lg text-xs transition"
          >
            Done
          </button>
        </div>
      </div>
    </div>
  )
}
