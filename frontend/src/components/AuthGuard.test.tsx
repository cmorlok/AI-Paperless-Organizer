import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, Routes, Route } from 'react-router-dom'
import AuthGuard, { invalidateAuthCache } from './AuthGuard'
import * as api from '../services/api'

// Auto-mock the api module so each test can control getAuthStatus
vi.mock('../services/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../services/api')>()
  return {
    ...actual,
    getAuthStatus: vi.fn(),
  }
})

function Wrapper({ initialEntries = ['/'], children }: { initialEntries?: string[]; children: React.ReactNode }) {
  return (
    <MemoryRouter initialEntries={initialEntries}>
      <Routes>
        <Route path="/login" element={<div data-testid="login-page">Login</div>} />
        <Route path="/setup" element={<div data-testid="setup-page">Setup</div>} />
        <Route path="/*" element={children} />
      </Routes>
    </MemoryRouter>
  )
}

describe('AuthGuard', () => {
  const mockedGetAuthStatus = vi.mocked(api.getAuthStatus)

  beforeEach(() => {
    invalidateAuthCache()
    mockedGetAuthStatus.mockReset()
  })

  afterEach(() => {
    vi.clearAllMocks()
  })

  it('redirects to /login when unauthenticated', async () => {
    mockedGetAuthStatus.mockResolvedValue({
      authenticated: false,
      requires_setup: false,
      auth_disabled: false,
    })

    render(
      <Wrapper>
        <AuthGuard>
          <div data-testid="protected-content">Protected</div>
        </AuthGuard>
      </Wrapper>
    )

    await waitFor(() => {
      expect(screen.getByTestId('login-page')).toBeInTheDocument()
    })
    expect(screen.queryByTestId('protected-content')).not.toBeInTheDocument()
    expect(mockedGetAuthStatus).toHaveBeenCalledTimes(1)
  })

  it('redirects to /setup when requires_setup is true', async () => {
    mockedGetAuthStatus.mockResolvedValue({
      authenticated: false,
      requires_setup: true,
      auth_disabled: false,
    })

    render(
      <Wrapper>
        <AuthGuard>
          <div data-testid="protected-content">Protected</div>
        </AuthGuard>
      </Wrapper>
    )

    await waitFor(() => {
      expect(screen.getByTestId('setup-page')).toBeInTheDocument()
    })
    expect(screen.queryByTestId('protected-content')).not.toBeInTheDocument()
  })

  it('renders children when authenticated', async () => {
    mockedGetAuthStatus.mockResolvedValue({
      authenticated: true,
      requires_setup: false,
      auth_disabled: false,
    })

    render(
      <Wrapper>
        <AuthGuard>
          <div data-testid="protected-content">Protected</div>
        </AuthGuard>
      </Wrapper>
    )

    await waitFor(() => {
      expect(screen.getByTestId('protected-content')).toBeInTheDocument()
    })
    expect(screen.queryByTestId('login-page')).not.toBeInTheDocument()
    expect(screen.queryByTestId('setup-page')).not.toBeInTheDocument()
  })

  it('renders children when auth_disabled is true', async () => {
    mockedGetAuthStatus.mockResolvedValue({
      authenticated: false,
      requires_setup: true,
      auth_disabled: true,
    })

    render(
      <Wrapper>
        <AuthGuard>
          <div data-testid="protected-content">Protected</div>
        </AuthGuard>
      </Wrapper>
    )

    await waitFor(() => {
      expect(screen.getByTestId('protected-content')).toBeInTheDocument()
    })
    expect(screen.queryByTestId('login-page')).not.toBeInTheDocument()
    expect(screen.queryByTestId('setup-page')).not.toBeInTheDocument()
  })

  it('uses TTL cache and avoids re-fetching within 5 seconds', async () => {
    mockedGetAuthStatus.mockResolvedValue({
      authenticated: true,
      requires_setup: false,
      auth_disabled: false,
    })

    const { unmount } = render(
      <Wrapper>
        <AuthGuard>
          <div data-testid="protected-content">Protected</div>
        </AuthGuard>
      </Wrapper>
    )

    await waitFor(() => {
      expect(screen.getByTestId('protected-content')).toBeInTheDocument()
    })
    expect(mockedGetAuthStatus).toHaveBeenCalledTimes(1)

    // Unmount and re-mount within TTL
    unmount()
    render(
      <Wrapper>
        <AuthGuard>
          <div data-testid="protected-content">Protected</div>
        </AuthGuard>
      </Wrapper>
    )

    await waitFor(() => {
      expect(screen.getByTestId('protected-content')).toBeInTheDocument()
    })
    // Should still be 1 because cache is fresh
    expect(mockedGetAuthStatus).toHaveBeenCalledTimes(1)
  })

  it('re-fetches auth status on window focus', async () => {
    mockedGetAuthStatus.mockResolvedValue({
      authenticated: true,
      requires_setup: false,
      auth_disabled: false,
    })

    render(
      <Wrapper>
        <AuthGuard>
          <div data-testid="protected-content">Protected</div>
        </AuthGuard>
      </Wrapper>
    )

    await waitFor(() => {
      expect(screen.getByTestId('protected-content')).toBeInTheDocument()
    })
    expect(mockedGetAuthStatus).toHaveBeenCalledTimes(1)

    // Simulate window focus
    window.dispatchEvent(new Event('focus'))

    await waitFor(() => {
      expect(mockedGetAuthStatus).toHaveBeenCalledTimes(2)
    })
  })
})
