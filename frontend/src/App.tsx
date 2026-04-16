import { useEffect, useState } from 'react'
import { listDocuments, type DocInfo } from './api/client'
import ChatWindow from './components/ChatWindow'
import Uploader from './components/Uploader'
import './index.css'

export default function App() {
  const [docs, setDocs] = useState<DocInfo[]>([])

  useEffect(() => {
    listDocuments().then(setDocs).catch(() => {})
  }, [])

  return (
    <div className="app">
      <header className="app-header">
        <h1 className="app-title">Ivy RAG</h1>
        <span className="app-subtitle">PDF Knowledge Assistant</span>
      </header>

      <div className="app-body">
        <aside className="sidebar">
          <h2 className="sidebar-heading">Documents</h2>
          <Uploader docs={docs} onDocsChange={setDocs} />
        </aside>

        <main className="main-content">
          <ChatWindow />
        </main>
      </div>
    </div>
  )
}
