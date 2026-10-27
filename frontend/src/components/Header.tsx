import React from 'react'
import {
  Cpu,
  Download,
  Settings,
  RotateCcw,
  Activity,
  Layers,
  Archive,
  Sparkles,
} from 'lucide-react'

interface HeaderProps {
  viewMode: 'vibe' | 'pro'
  onViewModeChange: (mode: 'vibe' | 'pro') => void
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
  viewMode,
  onViewModeChange,
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
      {/* Brand & Mode Switch */}
      <div className="flex items-center space-x-4">
        <div className="flex items-center space-x-3">
          <div className="h-8 w-8 rounded-lg bg-gradient-to-tr from-purple-600 via-indigo-600 to-blue-500 flex items-center justify-center shadow-lg shadow-purple-500/20">
            <Sparkles className="h-4 w-4 text-white" />
          </div>
          <div>
            <span className="font-bold text-sm tracking-wide bg-gradient-to-r from-gray-100 to-gray-400 bg-clip-text text-transparent">
              RL-NATIVE AGENT IDE
            </span>
            <span className="ml-2 text-xs text-purple-400 font-mono px-1.5 py-0.5 rounded bg-purple-950/60 border border-purple-800/40">
              v3.0 Vibe Studio
            </span>
          </div>
        </div>

        {/* Mode Switch: Vibe Studio (Default) | Pro RL Workbench */}
        <div className="flex items-center bg-gray-900 border border-gray-800 rounded-xl p-0.5 text-xs shadow-inner">
          <button
            onClick={() => onViewModeChange('vibe')}
            className={`px-3 py-1.5 rounded-lg font-semibold flex items-center space-x-1.5 transition ${
              viewMode === 'vibe'
                ? 'bg-gradient-to-r from-purple-600 to-indigo-600 text-white shadow-md shadow-purple-950/50'
                : 'text-gray-400 hover:text-gray-200'
            }`}
          >
            <Sparkles className="h-3.5 w-3.5" />
            <span>✨ Vibe Studio</span>
          </button>
          <button
            onClick={() => onViewModeChange('pro')}
            className={`px-3 py-1.5 rounded-lg font-semibold flex items-center space-x-1.5 transition ${
              viewMode === 'pro'
                ? 'bg-gray-800 text-white shadow-md'
                : 'text-gray-400 hover:text-gray-200'
            }`}
          >
            <Cpu className="h-3.5 w-3.5" />
            <span>🛠️ Pro RL Workbench</span>
          </button>
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
            <option value="DynamicSimAdapter" className="bg-gray-900">
              Dynamic Sim (Tic-Tac-Toe & Robotics)
            </option>
            <option value="AutoDataAdapter" className="bg-gray-900">
              Auto Data (Universal SQL / Docs)
            </option>
            <option value="CSVStreamAdapter" className="bg-gray-900">
              CSV Stream (Time-Series Anomaly)
            </option>
            <option value="TextClassifyAdapter" className="bg-gray-900">
              Text Triage (Support / Spam)
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
                      v{v}
                    </button>
                  ))}
              </div>
            </div>
          )}
        </div>

        {/* Model Drift Monitor Badge */}
        <div
          className={`flex items-center space-x-1.5 px-2.5 py-1 rounded-lg border text-xs font-semibold ${
            driftStatus === 'healthy'
              ? 'bg-emerald-950/60 border-emerald-800/50 text-emerald-400'
              : 'bg-rose-950/60 border-rose-800/50 text-rose-400'
          }`}
        >
          <Activity className="h-3.5 w-3.5" />
          <span>{driftStatus === 'healthy' ? 'Drift: None' : 'Drift Alert'}</span>
        </div>

        {/* RunBundle Policy Registry Catalog Button */}
        <button
          onClick={onOpenRegistry}
          title="Open RunBundle Catalog"
          className="flex items-center space-x-1.5 bg-gray-900/80 hover:bg-gray-800 border border-gray-800 rounded-lg px-2.5 py-1 text-xs text-gray-300 font-medium transition cursor-pointer"
        >
          <Archive className="h-3.5 w-3.5 text-blue-400" />
          <span>Bundles</span>
          {registryCount > 0 && (
            <span className="px-1.5 py-0.2 rounded-full bg-blue-950 border border-blue-800 text-blue-300 font-mono text-[10px] font-bold">
              {registryCount}
            </span>
          )}
        </button>

        {/* BYOK Settings Modal Button */}
        <button
          onClick={onOpenSettings}
          title="Configure BYOK LLM Provider"
          className="p-1.5 bg-gray-900/80 hover:bg-gray-800 border border-gray-800 rounded-lg text-gray-400 hover:text-gray-200 transition cursor-pointer"
        >
          <Settings className="h-4 w-4" />
        </button>

        {/* Export .runbundle Button */}
        <button
          onClick={onExport}
          disabled={isExporting}
          className="flex items-center space-x-1.5 bg-blue-600 hover:bg-blue-500 text-white rounded-lg px-3 py-1 text-xs font-semibold shadow-lg shadow-blue-600/20 transition cursor-pointer disabled:opacity-50"
        >
          <Download className="h-3.5 w-3.5" />
          <span>{isExporting ? 'Exporting...' : 'Export Bundle'}</span>
        </button>
      </div>
    </header>
  )
}
