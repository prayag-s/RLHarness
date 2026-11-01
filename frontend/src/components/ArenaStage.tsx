import React, { useCallback, useEffect, useRef, useState } from 'react'
import { ChevronLeft, ChevronRight, Maximize2, Monitor } from 'lucide-react'
import { SandboxedArenaIframe } from './SandboxedArenaIframe'

type ArenaProps = React.ComponentProps<typeof SandboxedArenaIframe>

interface ArenaStageProps extends ArenaProps {
  /** Highlights the stage when the thread points at an arena event. */
  pulse?: number
}

const MIN_WIDTH = 320
const MAX_WIDTH = 960

/**
 * The arena renders state; it never narrates. It sits beside the thread, resizable and
 * collapsible, so a wide camera view never pushes the conversation off screen.
 */
export const ArenaStage: React.FC<ArenaStageProps> = ({ pulse = 0, ...arenaProps }) => {
  const [width, setWidth] = useState(520)
  const [collapsed, setCollapsed] = useState(false)
  const [maximized, setMaximized] = useState(false)
  const [flashing, setFlashing] = useState(false)
  const draggingRef = useRef(false)

  // A remount would reset the iframe's mode and reload the canvas; a short flash is enough.
  useEffect(() => {
    if (!pulse) return
    setFlashing(true)
    const id = window.setTimeout(() => setFlashing(false), 700)
    return () => window.clearTimeout(id)
  }, [pulse])

  const onMouseDown = useCallback(() => {
    draggingRef.current = true
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'
  }, [])

  useEffect(() => {
    // A release outside the window would otherwise leave the whole app unselectable.
    const endDrag = () => {
      draggingRef.current = false
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
    }
    const onMove = (e: MouseEvent) => {
      if (!draggingRef.current) return
      const next = window.innerWidth - e.clientX
      setWidth(Math.max(MIN_WIDTH, Math.min(MAX_WIDTH, next)))
    }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', endDrag)
    window.addEventListener('blur', endDrag)
    return () => {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', endDrag)
      window.removeEventListener('blur', endDrag)
      endDrag()
    }
  }, [])

  if (collapsed) {
    return (
      <div className="w-10 shrink-0 border-l border-gray-800 bg-gray-950/80 flex flex-col items-center py-3 gap-3">
        <button
          onClick={() => setCollapsed(false)}
          className="rounded-md border border-gray-700 bg-gray-900 p-1.5 text-gray-300 hover:border-gray-500 transition"
          title="Show the arena"
        >
          <ChevronLeft className="h-4 w-4" />
        </button>
        <span
          className="text-[10px] uppercase tracking-widest text-gray-500"
          style={{ writingMode: 'vertical-rl' }}
        >
          Arena
        </span>
      </div>
    )
  }

  return (
    <div
      className={`relative shrink-0 border-l border-gray-800 bg-gray-950/60 flex flex-col ${
        maximized ? 'fixed inset-y-0 right-0 z-40 shadow-2xl' : ''
      }`}
      style={maximized ? { width: '80vw' } : { width }}
    >
      <div
        onMouseDown={onMouseDown}
        className="absolute -left-1 top-0 h-full w-2 cursor-col-resize hover:bg-indigo-500/30 transition"
        title="Drag to resize"
      />

      <div className="flex items-center justify-between px-3 py-2 border-b border-gray-800">
        <span className="flex items-center gap-1.5 text-[11px] font-bold uppercase tracking-wider text-gray-400">
          <Monitor className="h-3.5 w-3.5 text-cyan-400" />
          Arena
          {arenaProps.arenaPending && (
            <span className="relative flex h-1.5 w-1.5">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75" />
              <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-cyan-500" />
            </span>
          )}
        </span>
        <div className="flex items-center gap-1">
          <button
            onClick={() => setMaximized((v) => !v)}
            className="rounded p-1 text-gray-400 hover:text-gray-200 hover:bg-gray-800 transition"
            title={maximized ? 'Restore the studio layout' : 'Maximize the arena'}
          >
            <Maximize2 className="h-3.5 w-3.5" />
          </button>
          <button
            onClick={() => setCollapsed(true)}
            className="rounded p-1 text-gray-400 hover:text-gray-200 hover:bg-gray-800 transition"
            title="Collapse the arena"
          >
            <ChevronRight className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>

      <div
        className={`flex-1 min-h-0 overflow-hidden transition-shadow ${
          flashing ? 'ring-2 ring-cyan-400/60' : ''
        }`}
      >
        <SandboxedArenaIframe {...arenaProps} />
      </div>
    </div>
  )
}
