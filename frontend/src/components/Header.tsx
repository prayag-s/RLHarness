import React from 'react'
import {
  Cpu,
  Download,
  Settings,
  RotateCcw,
  Activity,
  Layers,
  Archive,
} from 'lucide-react'

interface HeaderProps {
  adapter: string
  onAdapterChange: (adapter: string) => void
  currentVersion: number
  availableVersions: number[]
  onRollback: (version: number) => void
  driftStatus: 'healthy' | 'drifted'
  onOpenSettings: () => void
  onOpenRegistry: () => void
  registryCount?: number
  onExport: () => void
  isExporting: boolean
}

export const Header: React.FC<HeaderProps> = ({
  adapter,
  onAdapterChange,
  currentVersion,
  availableVersions,
  onRollback,
  driftStatus,
  onOpenSettings,
  onOpenRegistry,
  registryCount = 0,
  onExport,
  isExporting,
}) => {
  return (
    <header className="h-14 border-b border-panel-border bg-panel px-4 flex items-center justify-between">
      {/* Brand */}
      <div className="flex items-center space-x-3">
        <div className="h-8 w-8 rounded-lg bg-gradient-to-tr from-blue-600 to-indigo-500 flex items-center justify-center shadow-lg shadow-blue-500/20">
          <Cpu className="h-5 w-5 text-white" />
        </div>
        <div>
          <span className="font-bold text-sm tracking-wide bg-gradient-to-r from-gray-100 to-gray-400 bg-clip-text text-transparent">
            RL-NATIVE AGENT IDE
          </span>
          <span className="ml-2 text-xs text-blue-400 font-mono px-1.5 py-0.5 rounded bg-blue-950/60 border border-blue-800/40">
            v2.0 Hardened
          </span>
        </div>
      </div>

      {/* Controls */}
      <div className="flex items-center space-x-3">
        {/* Adapter Selector */}
        <div className="flex items-center space-x-2 bg-gray-900/80 border border-gray-800 rounded-lg px-2.5 py-1 text-xs">
          <Layers className="h-3.5 w-3.5 text-gray-400" />
          <span className="text-gray-400 font-medium">Adapter:</span>
          <select
            value={adapter}
            onChange={(e) => onAdapterChange(e.target.value)}
            className="bg-transparent text-gray-200 font-semibold focus:outline-none cursor-pointer"
          >
            <option value="CSVStreamAdapter" className="bg-gray-900">
              CSV Stream (Time-Series Anomaly)
            </option>
            <option value="TextClassifyAdapter" className="bg-gray-900">
              Text Triage (Bandit / Support)
            </option>
          </select>
        </div>

        {/* Version Badge & Rollback */}
        <div className="flex items-center space-x-1.5 bg-gray-900/80 border border-gray-800 rounded-lg px-2.5 py-1 text-xs">
          <span className="text-gray-400 font-medium">Version:</span>
          <span className="text-emerald-400 font-mono font-bold">
            v{currentVersion}
          </span>
          {availableVersions.length > 1 && (
            <div className="relative group ml-1">
              <button
                title="Rollback version"
                className="hover:text-blue-400 text-gray-400 p-0.5"
              >
                <RotateCcw className="h-3 w-3" />
              </button>
              <div className="absolute right-0 top-full mt-1 hidden group-hover:flex flex-col bg-gray-900 border border-gray-800 rounded shadow-xl py-1 z-50 w-28">
                <span className="px-2 py-0.5 text-[10px] text-gray-500 uppercase font-bold">
                  Rollback to:
                </span>
                {availableVersions
                  .filter((v) => v !== currentVersion)
                  .map((v) => (
                    <button
                      key={v}
                      onClick={() => onRollback(v)}
                      className="px-2 py-1 text-left hover:bg-gray-800 text-xs text-gray-300 font-mono"
                    >
                      Restore v{v}
                    </button>
                  ))}
              </div>
            </div>
          )}
        </div>

        {/* Drift Status Indicator */}
        <div className="flex items-center space-x-1.5 bg-gray-900/80 border border-gray-800 rounded-lg px-2.5 py-1 text-xs">
          <Activity className="h-3.5 w-3.5 text-gray-400" />
          <span
            className={`h-2 w-2 rounded-full ${
              driftStatus === 'healthy' ? 'bg-emerald-500 animate-pulse' : 'bg-red-500'
            }`}
          />
          <span
            className={`font-medium ${
              driftStatus === 'healthy' ? 'text-emerald-400' : 'text-red-400'
            }`}
          >
            {driftStatus === 'healthy' ? 'Drift: Healthy' : 'Drift: Alert'}
          </span>
        </div>

        {/* Policy Registry CTA */}
        <button
          onClick={onOpenRegistry}
          className="flex items-center space-x-1.5 px-2.5 py-1 text-gray-300 hover:text-white bg-gray-900/80 hover:bg-gray-800 rounded-lg border border-gray-800 transition text-xs font-medium"
          title="Team Policy Registry & RunBundles"
        >
          <Archive className="h-3.5 w-3.5 text-blue-400" />
          <span>Registry</span>
          {registryCount > 0 && (
            <span className="px-1.5 py-0.2 rounded-full bg-blue-950 text-blue-300 text-[10px] font-mono border border-blue-800/50">
              {registryCount}
            </span>
          )}
        </button>

        {/* Settings button */}
        <button
          onClick={onOpenSettings}
          className="p-1.5 text-gray-400 hover:text-gray-200 hover:bg-gray-800 rounded-lg border border-gray-800 transition"
          title="BYOK & LLM Configuration"
        >
          <Settings className="h-4 w-4" />
        </button>

        {/* Export CTA */}
        <button
          onClick={onExport}
          disabled={isExporting}
          className="flex items-center space-x-2 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-medium text-xs px-3.5 py-1.5 rounded-lg shadow-lg shadow-blue-500/20 transition disabled:opacity-50"
        >
          <Download className="h-3.5 w-3.5" />
          <span>{isExporting ? 'Exporting...' : 'Export Owned Policy (ONNX)'}</span>
        </button>
      </div>
    </header>
  )
}
