import { useEffect, useState } from 'react'
import SignIn from './pages/SignIn'
import Board from './pages/Board'
import { post } from './auth/webauthn'

type Screen = 'signin' | 'board'

const TOKEN_KEY = 'volcast.access'

export default function App() {
  const [screen, setScreen] = useState<Screen>('signin')
  const [token, setToken] = useState<string | null>(null)
  const [displayName, setDisplayName] = useState<string | null>(null)

  // The refresh token lives in an HttpOnly cookie, so a returning visitor can be
  // resumed without asking them to re-enrol a passkey.
  useEffect(() => {
    const saved = sessionStorage.getItem(TOKEN_KEY)
    if (saved) {
      setToken(saved)
      setScreen('board')
      return
    }
    void (async () => {
      try {
        const r = await post<{ access_token?: string }>('/api/auth/refresh/cookie', {})
        if (r?.access_token) {
          setToken(r.access_token)
          sessionStorage.setItem(TOKEN_KEY, r.access_token)
          setScreen('board')
        }
      } catch {
        /* not signed in yet - expected */
      }
    })()
  }, [])

  function onAuthenticated(t: string, name: string | null) {
    setToken(t)
    setDisplayName(name)
    sessionStorage.setItem(TOKEN_KEY, t)
    setScreen('board')
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
  return <Board token={token} displayName={displayName} onSignOut={signOut} />
}
