import { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { queryRag, type QueryResponse } from '../api/client'
import CitationCard from './CitationCard'

interface Message {
  role: 'user' | 'assistant'
  text: string
  meta?: QueryResponse
}

const INTENT_LABELS: Record<string, string> = {
  chitchat: 'Chat',
  knowledge_lookup: 'KB search',
  list_request: 'List',
  table_request: 'Table',
  summary_request: 'Summary',
  unsafe: 'Refused',
}

export default function ChatWindow() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  async function send() {
    const q = input.trim()
    if (!q || loading) return
    setInput('')
    setMessages(prev => [...prev, { role: 'user', text: q }])
    setLoading(true)

    try {
      const res = await queryRag(q)
      setMessages(prev => [...prev, { role: 'assistant', text: res.answer, meta: res }])
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Request failed'
      setMessages(prev => [...prev, { role: 'assistant', text: `Error: ${msg}` }])
    } finally {
      setLoading(false)
    }
  }

  function stripCitationBlock(text: string): string {
    // Only strip a trailing block that is PURELY citation references (no prose).
    // Pattern: optional "Citations:" header + lines containing only [Sn] refs.
    // This avoids cutting bullet points or summaries that follow the citations.
    return text
      .replace(/\n+\*{0,2}Citations?:?\*{0,2}\s*\n([\s,\[\]S\d\n]+)$/i, '')
      .trimEnd()
  }

  function renderAnswer(msg: Message) {
    // When sentences are flagged, keep plain-text rendering so per-sentence
    // highlighting spans work correctly. Otherwise render full markdown.
    const text = stripCitationBlock(msg.text)
    if (msg.meta?.unsupported_sentences?.length) {
      const unsupportedTexts = new Set(msg.meta.unsupported_sentences.map(u => u.text))
      const sentences = text.split(/(?<=[.!?])\s+(?=[A-Z"])/)
      return (
        <div className="answer-text">
          {sentences.map((s, i) => {
            const isUnsupported = unsupportedTexts.has(s)
            const webContradicted = msg.meta?.unsupported_sentences.find(
              u => u.text === s && u.status === 'web_contradicted',
            )
            return (
              <span
                key={i}
                className={isUnsupported ? (webContradicted ? 'web-contradicted' : 'unsupported') : ''}
                title={isUnsupported ? (webContradicted ? 'Contradicted by sources' : 'Low evidence support') : undefined}
              >
                {s}{' '}
              </span>
            )
          })}
        </div>
      )
    }

    return (
      <div className="answer-text answer-markdown">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{text}</ReactMarkdown>
      </div>
    )
  }

  return (
    <div className="chat-window">
      <div className="messages">
        {messages.length === 0 && (
          <p className="empty-state">Upload PDFs on the left, then ask anything about them.</p>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`message message-${msg.role}`}>
            {msg.role === 'user' && msg.meta && (
              <span className="intent-badge">
                {INTENT_LABELS[msg.meta.intent] ?? msg.meta.intent}
              </span>
            )}

            {/* For user messages just show text; for assistant render with highlights */}
            {msg.role === 'user' ? (
              <p className="answer-text">{msg.text}</p>
            ) : (
              renderAnswer(msg)
            )}

            {/* Insufficient evidence banner */}
            {msg.meta && !msg.meta.sufficient_evidence && msg.meta.intent !== 'chitchat' && (
              <div className="evidence-banner">Insufficient evidence in the uploaded documents.</div>
            )}

            {/* Citations */}
            {msg.meta?.citations && msg.meta.citations.length > 0 && (
              <CitationCard citations={msg.meta.citations} />
            )}
          </div>
        ))}

        {loading && (
          <div className="message message-assistant">
            <span className="thinking">Thinking…</span>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      <div className="input-row">
        <input
          className="chat-input"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && !e.shiftKey && send()}
          placeholder="Ask a question about your documents…"
          disabled={loading}
        />
        <button className="send-btn" onClick={send} disabled={loading || !input.trim()}>
          Send
        </button>
      </div>
    </div>
  )
}
