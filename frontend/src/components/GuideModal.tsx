import React, { useMemo, useRef } from 'react'
import ReactMarkdown, { type Components } from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { BookOpen, X } from 'lucide-react'
import guideMarkdown from '../../../docs/USER_GUIDE.md?raw'

interface GuideModalProps {
  isOpen: boolean
  onClose: () => void
}

interface TocEntry {
  level: 1 | 2
  text: string
  id: string
}

function slugify(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^\w\s-]/g, '')
    .trim()
    .replace(/\s+/g, '-')
}

function plainText(children: React.ReactNode): string {
  if (typeof children === 'string' || typeof children === 'number') return String(children)
  if (Array.isArray(children)) return children.map(plainText).join('')
  if (React.isValidElement<{ children?: React.ReactNode }>(children)) {
    return plainText(children.props.children)
  }
  return ''
}

function buildToc(markdown: string): TocEntry[] {
  const entries: TocEntry[] = []
  let inCode = false
  for (const line of markdown.split('\n')) {
    if (line.trimStart().startsWith('```')) {
      inCode = !inCode
      continue
    }
    if (inCode) continue
    const match = /^(#{1,2})\s+(.*)$/.exec(line)
    if (!match) continue
    const text = match[2].replace(/[*`]/g, '').trim()
    entries.push({ level: match[1].length as 1 | 2, text, id: slugify(text) })
  }
  return entries
}

const heading =
  (Tag: 'h1' | 'h2' | 'h3') =>
  ({ children }: React.HTMLAttributes<HTMLHeadingElement>) => {
    const text = plainText(children)
    return <Tag id={slugify(text)}>{children}</Tag>
  }

const components: Components = {
  h1: heading('h1'),
  h2: heading('h2'),
  h3: heading('h3'),
}

export const GuideModal: React.FC<GuideModalProps> = ({ isOpen, onClose }) => {
  const toc = useMemo(() => buildToc(guideMarkdown), [])
  const contentRef = useRef<HTMLDivElement>(null)

  if (!isOpen) return null

  const jumpTo = (id: string) => {
    const container = contentRef.current
    if (!container) return
    const target = container.querySelector<HTMLElement>(`[id="${id}"]`)
    if (!target) return
    const offset =
      target.getBoundingClientRect().top -
      container.getBoundingClientRect().top +
      container.scrollTop
    container.scrollTo({ top: Math.max(0, offset - 12) })
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-3 md:p-6">
      <div className="w-full max-w-6xl h-full max-h-[92vh] bg-gray-950 border border-gray-800 rounded-2xl shadow-2xl flex flex-col overflow-hidden text-gray-200">
        <div className="h-14 shrink-0 border-b border-gray-800 bg-gray-900/80 px-4 md:px-5 flex items-center justify-between">
          <div className="flex items-center space-x-3 min-w-0">
            <div className="h-8 w-8 shrink-0 rounded-lg bg-emerald-950/80 border border-emerald-700/50 flex items-center justify-center text-emerald-400">
              <BookOpen className="h-4 w-4" />
            </div>
            <div className="min-w-0">
              <h2 className="text-sm font-bold text-gray-100 truncate">The RL Harness Field Guide</h2>
              <p className="text-[11px] text-gray-400 truncate">
                Plain-English handbook: build, judge, and ship agents
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-gray-400 hover:text-white hover:bg-gray-800 transition"
            title="Close the guide"
            aria-label="Close the guide"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="flex-1 flex min-h-0">
          <nav className="hidden lg:block w-64 shrink-0 border-r border-gray-800 overflow-y-auto px-3 py-4">
            <div className="text-[10px] uppercase tracking-wider text-gray-500 font-bold mb-2 px-2">
              Contents
            </div>
            <ul className="space-y-0.5">
              {toc.map((entry) =>
                entry.level === 1 ? (
                  <li
                    key={entry.id}
                    className="pt-3 pb-1 px-2 text-[10px] uppercase tracking-wider text-emerald-400/80 font-bold"
                  >
                    {entry.text}
                  </li>
                ) : (
                  <li key={entry.id}>
                    <button
                      onClick={() => jumpTo(entry.id)}
                      className="w-full text-left px-2 py-1 rounded text-[11px] text-gray-400 hover:text-gray-100 hover:bg-gray-800/70 transition leading-snug"
                    >
                      {entry.text}
                    </button>
                  </li>
                ),
              )}
            </ul>
          </nav>

          <div ref={contentRef} className="flex-1 overflow-y-auto px-4 md:px-10 py-6">
            <article className="guide-prose max-w-3xl mx-auto">
              <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
                {guideMarkdown}
              </ReactMarkdown>
            </article>
          </div>
        </div>
      </div>
    </div>
  )
}
