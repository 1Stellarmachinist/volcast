import { useEffect, useState } from 'react'
import SignIn from './pages/SignIn'
import Dashboard from './pages/Dashboard'
import { post } from './auth/webauthn'

type Screen = 'signin' | 'forecast'

const TOKEN_KEY = 'volcast.access'

export default function App() {
  const [screen, setScreen] = useState<Screen>('signin')
  const [token, setToken] = useState<string | null>(null)
  const [displayName, setDisplayName] = useState<string | null>(null)

  // Silent refresh on boot: the refresh token is an HttpOnly cookie, so the only
  // thing worth persisting in JS is the short-lived access token.
  useEffect(() => {
    const saved = sessionStorage.getItem(TOKEN_KEY)
    if (saved) {
      setToken(saved)
      setScreen('forecast')
      return
    }
    void (async () => {
      try {
        // The refresh cookie is the only credential a returning visitor has.
        const res = await post<{ access_token?: string }>('/api/auth/refresh/cookie', {})
        if (res?.access_token) {
          setToken(res.access_token)
          sessionStorage.setItem(TOKEN_KEY, res.access_token)
          setScreen('forecast')
        }
      } catch {
        /* 401 here just means "not signed in yet" - expected */
      }
    })()
  }, [])

  function onAuthenticated(t: string, name: string | null) {
    setToken(t)
    setDisplayName(name)
    sessionStorage.setItem(TOKEN_KEY, t)
    setScreen('forecast')
  }

  function signOut() {
    sessionStorage.removeItem(TOKEN_KEY)
    setToken(null)
    setDisplayName(null)
    setScreen('signin')
  }

  if (screen === 'signin' || !token) {
    return <SignIn onAuthenticated={onAuthenticated} />
  }
  return <Dashboard token={token} displayName={displayName} onSignOut={signOut} />
}
