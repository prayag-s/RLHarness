import React, { useState } from 'react'
import { Database, X, CheckCircle, AlertTriangle, ArrowRight, Loader2 } from 'lucide-react'
import { ingestSQL } from '../services/api'
import { IngestedDatasetSummary } from '../types'

interface SQLIngestModalProps {
  isOpen: boolean
  onClose: () => void
  onDatasetSelected: (dataset: IngestedDatasetSummary) => void
}

export const SQLIngestModal: React.FC<SQLIngestModalProps> = ({
  isOpen,
  onClose,
  onDatasetSelected,
}) => {
  const [uri, setUri] = useState('sqlite:///./runs/sample_data.db')
  const [query, setQuery] = useState('SELECT * FROM transactions LIMIT 500')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<IngestedDatasetSummary | null>(null)

  if (!isOpen) return null

  const handleConnect = async () => {
    setIsLoading(true)
    setError(null)
    setResult(null)
    try {
      const data = await ingestSQL(uri, query)
      setResult(data)
    } catch (err: any) {
      setError(err.message || 'SQL connection failed')
    } finally {
      setIsLoading(false)
    }
  }

  const handleApply = () => {
    if (result) {
      onDatasetSelected(result)
      onClose()
    }
  }

  return (
    <div className="fixed inset-0 bg-black/75 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-panel border border-panel-border rounded-xl shadow-2xl max-w-2xl w-full flex flex-col max-h-[90vh]">
        {/* Header */}
        <div className="px-5 py-4 border-b border-panel-border flex items-center justify-between">
          <div className="flex items-center space-x-2.5">
            <div className="p-2 rounded-lg bg-emerald-950/60 border border-emerald-800/50 text-emerald-400">
              <Database className="h-5 w-5" />
            </div>
            <div>
              <h2 className="text-base font-bold text-gray-100">Connect Live SQL Database</h2>
              <p className="text-xs text-gray-400">
                Read-only connection guardrails automatically protect production databases
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-200 p-1 rounded-lg hover:bg-gray-800 transition"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-5 overflow-y-auto space-y-4 flex-1 text-xs">
          {/* Connection URI */}
          <div>
            <label className="block text-gray-300 font-semibold mb-1">Database Connection URI</label>
            <input
              type="text"
              value={uri}
              onChange={(e) => setUri(e.target.value)}
              placeholder="sqlite:///path/to/db.sqlite or postgresql://user:pass@host:5432/dbname"
              className="w-full bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-gray-200 font-mono text-xs focus:outline-none focus:border-emerald-500"
            />
            <div className="flex items-center space-x-2 mt-1.5 text-[11px] text-gray-400">
              <span className="font-semibold text-gray-500">Presets:</span>
              <button
                onClick={() => setUri('sqlite:///./runs/sample_data.db')}
                className="hover:text-emerald-400 transition"
              >
                SQLite Demo
              </button>
              <span>•</span>
              <button
                onClick={() => setUri('postgresql://readonly:password@localhost:5432/warehouse')}
                className="hover:text-emerald-400 transition"
              >
                PostgreSQL
              </button>
              <span>•</span>
              <button
                onClick={() => setUri('mysql+pymysql://reader:password@localhost:3306/production')}
                className="hover:text-emerald-400 transition"
              >
                MySQL
              </button>
            </div>
          </div>

          {/* SQL Query */}
          <div>
            <label className="block text-gray-300 font-semibold mb-1">
              Read-Only SQL Query or Table Name
            </label>
            <textarea
              rows={3}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="SELECT id, amount, risk_score, department FROM invoices WHERE status = 'pending' LIMIT 1000"
              className="w-full bg-gray-900 border border-gray-700 rounded-lg p-3 text-gray-200 font-mono text-xs focus:outline-none focus:border-emerald-500"
            />
            <p className="text-[11px] text-gray-400 mt-1">
              🛡️ Enforces strict read-only AST safety. Any write attempts (DROP, UPDATE, INSERT, ALTER) are automatically blocked.
            </p>
          </div>

          {error && (
            <div className="p-3 rounded-lg bg-rose-950/50 border border-rose-800/60 text-rose-300 flex items-start space-x-2">
              <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5" />
              <span>{error}</span>
            </div>
          )}

          {/* Result preview */}
          {result && (
            <div className="p-3.5 rounded-lg bg-gray-900 border border-gray-800 space-y-2.5">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2 text-emerald-400 font-semibold">
                  <CheckCircle className="h-4 w-4" />
                  <span>Connected successfully ({result.total_rows} rows loaded)</span>
                </div>
                <span className="text-gray-400 font-mono text-[11px]">{result.format.toUpperCase()}</span>
              </div>

              <div>
                <span className="text-gray-400 font-medium">Detected Columns: </span>
                <span className="text-gray-200 font-mono text-[11px]">
                  {result.columns.join(', ')}
                </span>
              </div>

              {result.sample_records && result.sample_records.length > 0 && (
                <div className="overflow-x-auto max-h-36 border border-gray-800 rounded">
                  <table className="w-full text-left text-[11px]">
                    <thead className="bg-gray-800/80 text-gray-400 sticky top-0">
                      <tr>
                        {result.columns.slice(0, 6).map((col) => (
                          <th key={col} className="px-2 py-1 font-mono">{col}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-800 text-gray-300">
                      {result.sample_records.slice(0, 4).map((row, idx) => (
                        <tr key={idx} className="hover:bg-gray-800/40">
                          {result.columns.slice(0, 6).map((col) => (
                            <td key={col} className="px-2 py-1 font-mono text-gray-400">
                              {String(row[col] ?? '')}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-5 py-3 border-t border-panel-border bg-panel-subtle flex items-center justify-end space-x-2.5">
          <button
            onClick={onClose}
            className="px-3.5 py-1.5 rounded-lg border border-gray-700 text-gray-300 hover:bg-gray-800 text-xs font-medium transition"
          >
            Cancel
          </button>
          {!result ? (
            <button
              onClick={handleConnect}
              disabled={isLoading || !uri.trim()}
              className="px-4 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-semibold text-xs transition flex items-center space-x-1.5 shadow-lg shadow-emerald-600/20 disabled:opacity-50"
            >
              {isLoading ? (
                <>
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  <span>Testing Connection...</span>
                </>
              ) : (
                <>
                  <Database className="h-3.5 w-3.5" />
                  <span>Inspect Database</span>
                </>
              )}
            </button>
          ) : (
            <button
              onClick={handleApply}
              className="px-4 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white font-semibold text-xs transition flex items-center space-x-1.5 shadow-lg shadow-blue-600/20"
            >
              <span>Attach Dataset to Vibe Studio</span>
              <ArrowRight className="h-3.5 w-3.5" />
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
