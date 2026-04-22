import { useState, KeyboardEvent } from 'react'
import { useNavigate } from 'react-router-dom'
import { Lock, Loader2 } from 'lucide-react'
import clsx from 'clsx'
import * as api from '../services/api'

export default function SetupPage() {
  const [password, setPassword] = useState('')
  const [confirm, setConfirm] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const navigate = useNavigate()

  const handleSubmit = async () => {
    if (loading) return
    if (!password) {
      setError('Die Passwörter stimmen nicht überein.')
      return
    }
    if (password !== confirm) {
      setError('Die Passwörter stimmen nicht überein.')
      return
    }
    setError(null)
    setLoading(true)
    try {
      await api.setupPassword(password)
      navigate('/login', { replace: true })
    } catch (e: any) {
      setError(e?.message || 'Fehler bei der Einrichtung.')
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
        <h1 className="font-display text-xl font-bold text-surface-100 mb-2">
          Passwort festlegen
        </h1>
        <p className="text-surface-400 mb-6">
          Richten Sie ein Administrator-Passwort ein, um die Anwendung zu sichern.
        </p>
        <input
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          onKeyDown={handleKey}
          placeholder="Neues Passwort"
          className={clsx("input mb-3 text-center", error && "border-red-500")}
          autoFocus
          disabled={loading}
        />
        <input
          type="password"
          value={confirm}
          onChange={(e) => setConfirm(e.target.value)}
          onKeyDown={handleKey}
          placeholder="Passwort bestätigen"
          className={clsx("input mb-4 text-center", error && "border-red-500")}
          disabled={loading}
        />
        {error && (
          <p className="text-red-400 text-sm mb-4">{error}</p>
        )}
        <button
          onClick={handleSubmit}
          disabled={loading || !password || !confirm}
          className="btn btn-primary w-full flex items-center justify-center gap-2"
        >
          {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Lock className="w-4 h-4" />}
          Einrichten
        </button>
      </div>
    </div>
  )
}
