import { useState, KeyboardEvent } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import { Lock, Loader2 } from 'lucide-react'
import clsx from 'clsx'
import * as api from '../services/api'

export default function LoginPage() {
  const [password, setPassword] = useState('')
  const [error, setError] = useState(false)
  const [loading, setLoading] = useState(false)
  const navigate = useNavigate()
  const location = useLocation()

  const from = (location.state as { from?: { pathname: string } } | null)?.from?.pathname ?? '/'

  const handleSubmit = async () => {
    if (!password || loading) return
    setLoading(true)
    setError(false)
    try {
      await api.login(password)
      navigate(from, { replace: true })
    } catch {
      setError(true)
      setPassword('')
    } finally {
      setLoading(false)
    }
  }

  const handleKey = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') handleSubmit()
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-surface-900 via-surface-800 to-surface-900 flex items-center justify-center p-4 animate-fade-in">
      <div className="card p-8 max-w-md w-full text-center border border-primary-500/30">
        <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-primary-500 to-primary-700 flex items-center justify-center shadow-xl shadow-primary-600/40 mx-auto mb-6">
          <Lock className="w-10 h-10 text-white" />
        </div>
        <h1 className="font-display text-2xl font-bold text-surface-100 mb-2">
          AI Paperless Organizer
        </h1>
        <p className="text-surface-400 mb-6">
          Diese Anwendung ist passwortgeschützt
        </p>
        <input
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          onKeyDown={handleKey}
          placeholder="Passwort eingeben..."
          className={clsx("input mb-4 text-center", error && "border-red-500")}
          autoFocus
          disabled={loading}
        />
        {error && (
          <p className="text-red-400 text-sm mb-4">Falsches Passwort</p>
        )}
        <button
          onClick={handleSubmit}
          disabled={loading || !password}
          className="btn btn-primary w-full flex items-center justify-center gap-2"
        >
          {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Lock className="w-4 h-4" />}
          Anmelden
        </button>
        <p className="text-surface-500 text-xs mt-6">
          Passwort vergessen? Setzen Sie die Umgebungsvariable RESET_PASSWORD=true und starten Sie die Anwendung neu.
        </p>
      </div>
    </div>
  )
}
