import { useEffect, useState, ReactNode } from 'react'
import { Navigate, useLocation } from 'react-router-dom'
import * as api from '../services/api'

type AuthState = 'loading' | 'authenticated' | 'unauthenticated' | 'needs_setup' | 'auth_disabled'

interface AuthStatus {
  authenticated: boolean
  requires_setup: boolean
  auth_disabled: boolean
}

// In-memory TTL cache to avoid re-fetching auth status on every navigation
// (addresses review concern about per-pathname re-fetching)
const _authCache = {
  status: null as AuthStatus | null,
  expiresAt: 0,
}
const CACHE_TTL_MS = 5000

async function fetchAuthStatus(): Promise<AuthStatus> {
  const now = Date.now()
  if (_authCache.status && now < _authCache.expiresAt) {
    return _authCache.status
  }
  const status = await api.getAuthStatus()
  _authCache.status = status
  _authCache.expiresAt = now + CACHE_TTL_MS
  return status
}

export function invalidateAuthCache() {
  _authCache.status = null
  _authCache.expiresAt = 0
}

export default function AuthGuard({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AuthState>('loading')
  const location = useLocation()

  useEffect(() => {
    let cancelled = false
    const check = async () => {
      try {
        const status = await fetchAuthStatus()
        if (cancelled) return
        if (status.auth_disabled) setState('auth_disabled')
        else if (status.requires_setup) setState('needs_setup')
        else if (status.authenticated) setState('authenticated')
        else setState('unauthenticated')
      } catch {
        if (!cancelled) setState('unauthenticated')
      }
    }
    check()

    const onFocus = () => {
      // Re-fetch when user returns to tab (e.g., logged out elsewhere)
      invalidateAuthCache()
      check()
    }
    window.addEventListener('focus', onFocus)
    return () => {
      cancelled = true
      window.removeEventListener('focus', onFocus)
    }
  }, []) // empty deps: only on mount + focus, not on every pathname change

  if (state === 'loading') {
    return (
      <div className="min-h-screen bg-gradient-to-br from-surface-900 via-surface-800 to-surface-900 flex items-center justify-center text-surface-300">
        Laden...
      </div>
    )
  }
  if (state === 'auth_disabled') return <>{children}</>
  if (state === 'needs_setup') return <Navigate to="/setup" replace />
  if (state === 'unauthenticated') {
    return <Navigate to="/login" state={{ from: location }} replace />
  }
  return <>{children}</>
}
