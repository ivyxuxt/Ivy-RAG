const BASE = ''  // proxied by Vite dev server to http://localhost:8000

export interface Citation {
  chunk_id: string
  doc_name: string
  page_start: number | null
  page_end: number | null
  section_title: string | null
  score: number
  snippet: string
}

export interface UnsupportedSentence {
  index: number
  text: string
  status: 'possibly_unsupported' | 'web_contradicted'
}

export interface QueryResponse {
  answer: string
  citations: Citation[]
  intent: string
  retrieved_chunks: number
  sufficient_evidence: boolean
  unsupported_sentences: UnsupportedSentence[]
}

export interface IngestedDoc {
  doc_id: string
  name: string
  n_pages: number
  n_chunks: number
}

export interface IngestResponse {
  ingested: IngestedDoc[]
  errors: string[]
}

export interface DocInfo {
  doc_id: string
  name: string
  n_chunks: number
}

export async function ingestFiles(files: File[]): Promise<IngestResponse> {
  const form = new FormData()
  for (const f of files) form.append('files', f)
  const res = await fetch(`${BASE}/ingest`, { method: 'POST', body: form })
  if (!res.ok) throw new Error(`Ingest failed: ${res.status}`)
  return res.json()
}

export async function queryRag(question: string, topK = 5): Promise<QueryResponse> {
  const res = await fetch(`${BASE}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, top_k: topK }),
  })
  if (!res.ok) throw new Error(`Query failed: ${res.status}`)
  return res.json()
}

export async function listDocuments(): Promise<DocInfo[]> {
  const res = await fetch(`${BASE}/documents`)
  if (!res.ok) throw new Error(`List failed: ${res.status}`)
  const data = await res.json()
  return data.documents
}

export async function deleteDocument(docId: string): Promise<void> {
  const res = await fetch(`${BASE}/documents/${docId}`, { method: 'DELETE' })
  if (!res.ok) throw new Error(`Delete failed: ${res.status}`)
}
