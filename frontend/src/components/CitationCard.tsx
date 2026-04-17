import { useState } from 'react'
import type { Citation } from '../api/client'

interface Props {
  citations: Citation[]
}

export default function CitationCard({ citations }: Props) {
  const [open, setOpen] = useState(true)
  if (citations.length === 0) return null

  return (
    <div className="citations">
      <button className="citations-toggle" onClick={() => setOpen(o => !o)}>
        {open ? '▾' : '▸'} {citations.length} source{citations.length > 1 ? 's' : ''}
      </button>

      {open && (
        <ul className="citation-list">
          {citations.map(c => {
            const pages =
              c.page_start && c.page_end && c.page_start !== c.page_end
                ? `pp. ${c.page_start}–${c.page_end}`
                : c.page_start
                  ? `p. ${c.page_start}`
                  : ''
            return (
              <li key={c.chunk_id} className="citation-item">
                <div className="citation-header">
                  <span className="citation-doc">{c.doc_name}</span>
                  {pages && <span className="citation-pages">{pages}</span>}
                  {c.section_title && (
                    <span className="citation-section">{c.section_title}</span>
                  )}
                  <span className="citation-score">score {c.score.toFixed(2)}</span>
                </div>
                <p className="citation-snippet">{c.snippet}</p>
              </li>
            )
          })}
        </ul>
      )}
    </div>
  )
}
