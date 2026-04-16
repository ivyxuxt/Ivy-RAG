import { useRef, useState } from 'react'
import { ingestFiles, listDocuments, deleteDocument, type DocInfo } from '../api/client'

interface Props {
  docs: DocInfo[]
  onDocsChange: (docs: DocInfo[]) => void
}

export default function Uploader({ docs, onDocsChange }: Props) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  const [status, setStatus] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  async function handleFiles(files: FileList | null) {
    if (!files || files.length === 0) return
    const pdfs = Array.from(files).filter(f => f.name.endsWith('.pdf'))
    if (pdfs.length === 0) {
      setStatus('Only PDF files are accepted.')
      return
    }
    setLoading(true)
    setStatus(null)
    try {
      const res = await ingestFiles(pdfs)
      const names = res.ingested.map(d => d.name).join(', ')
      setStatus(
        res.ingested.length > 0
          ? `Ingested: ${names}`
          : res.errors.join('; '),
      )
      const updated = await listDocuments()
      onDocsChange(updated)
    } catch (e: unknown) {
      setStatus(e instanceof Error ? e.message : 'Upload failed')
    } finally {
      setLoading(false)
    }
  }

  async function handleDelete(docId: string) {
    try {
      await deleteDocument(docId)
      onDocsChange(docs.filter(d => d.doc_id !== docId))
    } catch (e: unknown) {
      setStatus(e instanceof Error ? e.message : 'Delete failed')
    }
  }

  return (
    <div className="uploader">
      {/* Drop zone */}
      <div
        className={`drop-zone${dragging ? ' dragging' : ''}`}
        onClick={() => inputRef.current?.click()}
        onDragOver={e => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={e => { e.preventDefault(); setDragging(false); handleFiles(e.dataTransfer.files) }}
      >
        {loading ? 'Uploading…' : 'Drop PDFs here or click to browse'}
        <input
          ref={inputRef}
          type="file"
          accept=".pdf"
          multiple
          style={{ display: 'none' }}
          onChange={e => handleFiles(e.target.files)}
        />
      </div>

      {status && <p className="upload-status">{status}</p>}

      {/* Document list */}
      {docs.length > 0 && (
        <ul className="doc-list">
          {docs.map(doc => (
            <li key={doc.doc_id} className="doc-item">
              <span className="doc-name" title={doc.doc_id}>
                {doc.name}
                <span className="doc-meta"> · {doc.n_chunks} chunks</span>
              </span>
              <button className="delete-btn" onClick={() => handleDelete(doc.doc_id)} title="Remove">
                ✕
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
