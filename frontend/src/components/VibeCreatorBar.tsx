import React, { useRef, useState } from 'react'
import {
  Sparkles,
  UploadCloud,
  Database,
  Gamepad2,
  Navigation,
  FileText,
  Mail,
  TrendingUp,
  Loader2,
  CheckCircle2,
  XCircle,
} from 'lucide-react'
import { uploadIngestFile } from '../services/api'
import { IngestedDatasetSummary } from '../types'
import { SQLIngestModal } from './SQLIngestModal'

interface VibeCreatorBarProps {
  prompt: string
  onPromptChange: (val: string) => void
  onLaunch: (prompt: string, datasetId?: string) => void
  isLaunching: boolean
  activeDataset?: IngestedDatasetSummary | null
  onClearDataset?: () => void
  onDatasetAttached?: (dataset: IngestedDatasetSummary) => void
}

const DEMO_PRESETS = [
  {
    id: 'tictactoe',
    title: 'Tic-Tac-Toe Master',
    icon: Gamepad2,
    badgeColor: 'text-indigo-400 bg-indigo-950/60 border-indigo-800/40',
    prompt: 'Train an agent to play tic-tac-toe, take center when open, block opponent wins, and win.',
  },
  {
    id: 'robot_rover',
    title: '2D Robot Rover Trainer',
    icon: Navigation,
    badgeColor: 'text-cyan-400 bg-cyan-950/60 border-cyan-800/40',
    prompt: 'Train a robot rover to navigate around obstacles to a target beacon quickly while avoiding collisions and minimizing jerk.',
  },
  {
    id: 'pdf_auditor',
    title: 'PDF Policy & Invoice Auditor',
    icon: FileText,
    badgeColor: 'text-amber-400 bg-amber-950/60 border-amber-800/40',
    prompt: 'Audit financial transactions, flag any invoice over $5,000 missing a PO, and escalate high-risk compliance terms.',
  },
  {
    id: 'ticket_router',
    title: 'VIP Support Ticket Router',
    icon: Mail,
    badgeColor: 'text-emerald-400 bg-emerald-950/60 border-emerald-800/40',
    prompt: 'Classify support tickets into urgent, normal, and spam; prioritize angry VIP customers and never route spam to urgent.',
  },
  {
    id: 'stream_monitor',
    title: 'Flash-Crash Market Monitor',
    icon: TrendingUp,
    badgeColor: 'text-rose-400 bg-rose-950/60 border-rose-800/40',
    prompt: 'Detect high-volatility market price jumps and volume spikes without false alarms during normal consolidation.',
  },
]

export const VibeCreatorBar: React.FC<VibeCreatorBarProps> = ({
  prompt,
  onPromptChange,
  onLaunch,
  isLaunching,
  activeDataset,
  onClearDataset,
  onDatasetAttached,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [isUploading, setIsUploading] = useState(false)
  const [isSqlModalOpen, setIsSqlModalOpen] = useState(false)
  const [uploadError, setUploadError] = useState<string | null>(null)

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    setIsUploading(true)
    setUploadError(null)
    try {
      const summary = await uploadIngestFile(file)
      if (onDatasetAttached) onDatasetAttached(summary)
    } catch (err: any) {
      setUploadError(err.message || 'File upload failed')
    } finally {
      setIsUploading(false)
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  return (
    <div className="bg-panel border-b border-panel-border px-6 py-4 space-y-3.5 shadow-md">
      {/* Top row: Conversational Input Bar */}
      <div className="flex items-center space-x-3">
        <div className="relative flex-1">
          <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-purple-400">
            <Sparkles className="h-4 w-4" />
          </div>
          <input
            type="text"
            value={prompt}
            onChange={(e) => onPromptChange(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !isLaunching && prompt.trim()) {
                onLaunch(prompt, activeDataset?.dataset_id)
              }
            }}
            placeholder="Describe what you want your agent to learn, or pick a demo below..."
            className="w-full pl-10 pr-4 py-2.5 bg-gray-900 border border-gray-700/80 rounded-xl text-sm text-gray-100 placeholder-gray-500 focus:outline-none focus:border-purple-500/80 focus:ring-1 focus:ring-purple-500/50 shadow-inner"
          />
        </div>

        {/* Upload / SQL Ingest Buttons */}
        <div className="flex items-center space-x-2">
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv,.tsv,.xlsx,.json,.jsonl,.sqlite,.db,.pdf,.docx,.txt,.md"
            onChange={handleFileUpload}
            className="hidden"
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={isUploading || isLaunching}
            title="Upload CSV, Excel, PDF, or Word Policy Document"
            className="px-3 py-2 rounded-xl border border-gray-700 bg-gray-800/80 hover:bg-gray-700 text-gray-200 text-xs font-medium flex items-center space-x-1.5 transition disabled:opacity-50"
          >
            {isUploading ? (
              <Loader2 className="h-4 w-4 animate-spin text-purple-400" />
            ) : (
              <UploadCloud className="h-4 w-4 text-purple-400" />
            )}
            <span className="hidden sm:inline">Upload File</span>
          </button>

          <button
            onClick={() => setIsSqlModalOpen(true)}
            disabled={isLaunching}
            title="Connect Live SQL Database (PostgreSQL, MySQL, SQLite)"
            className="px-3 py-2 rounded-xl border border-gray-700 bg-gray-800/80 hover:bg-gray-700 text-gray-200 text-xs font-medium flex items-center space-x-1.5 transition"
          >
            <Database className="h-4 w-4 text-emerald-400" />
            <span className="hidden sm:inline">Connect SQL</span>
          </button>

          {/* Launch Button */}
          <button
            onClick={() => onLaunch(prompt, activeDataset?.dataset_id)}
            disabled={isLaunching || !prompt.trim()}
            className="px-5 py-2.5 rounded-xl bg-gradient-to-r from-purple-600 via-indigo-600 to-blue-600 hover:from-purple-500 hover:to-blue-500 text-white font-semibold text-xs transition shadow-lg shadow-purple-600/25 flex items-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
          >
            {isLaunching ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                <span>Training Agent...</span>
              </>
            ) : (
              <>
                <Sparkles className="h-4 w-4" />
                <span>Train Agent (1-Click)</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Active dataset badge (if attached) */}
      {activeDataset && (
        <div className="flex items-center justify-between px-3 py-1.5 rounded-lg bg-purple-950/40 border border-purple-800/50 text-xs">
          <div className="flex items-center space-x-2 text-purple-300">
            <CheckCircle2 className="h-3.5 w-3.5 text-purple-400" />
            <span>
              Attached Dataset: <strong className="text-white font-mono">{activeDataset.name}</strong> ({activeDataset.total_rows} rows, format: {activeDataset.format})
            </span>
            {activeDataset.extracted_rules && activeDataset.extracted_rules.length > 0 && (
              <span className="px-1.5 py-0.5 rounded bg-amber-950/60 border border-amber-800/40 text-amber-300 text-[10px] font-semibold">
                {activeDataset.extracted_rules.length} Document Rules Extracted
              </span>
            )}
          </div>
          {onClearDataset && (
            <button
              onClick={onClearDataset}
              className="text-gray-400 hover:text-rose-400 transition"
              title="Detach dataset"
            >
              <XCircle className="h-3.5 w-3.5" />
            </button>
          )}
        </div>
      )}

      {uploadError && (
        <div className="text-xs text-rose-400 px-1">{uploadError}</div>
      )}

      {/* Demo Preset Pills */}
      <div className="flex items-center space-x-2 overflow-x-auto pb-1 text-xs no-scrollbar">
        <span className="text-gray-400 font-semibold text-[11px] shrink-0">✨ Instant Demos:</span>
        {DEMO_PRESETS.map((demo) => {
          const Icon = demo.icon
          return (
            <button
              key={demo.id}
              onClick={() => {
                onPromptChange(demo.prompt)
                onLaunch(demo.prompt)
              }}
              disabled={isLaunching}
              className={`px-3 py-1 rounded-full border text-xs font-medium flex items-center space-x-1.5 transition shrink-0 hover:scale-105 active:scale-95 ${demo.badgeColor}`}
            >
              <Icon className="h-3.5 w-3.5" />
              <span>{demo.title}</span>
            </button>
          )
        })}
      </div>

      {/* SQL Ingest Modal */}
      <SQLIngestModal
        isOpen={isSqlModalOpen}
        onClose={() => setIsSqlModalOpen(false)}
        onDatasetSelected={(ds) => {
          if (onDatasetAttached) onDatasetAttached(ds)
        }}
      />
    </div>
  )
}
