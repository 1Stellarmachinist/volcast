import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  ApiError,
  decodeCreationOptions,
  decodeRequestOptions,
  isConditionalMediationAvailable,
  isWebAuthnAvailable,
  post,
  readBackupFlags,
  serialiseCredential,
  type PublicKeyCredentialCreationOptionsJSON,
  type PublicKeyCredentialRequestOptionsJSON,
} from '../auth/webauthn'
import { isValidChallenge, readHandoff } from '../auth/pkce'

interface Props {
  onAuthenticated: (token: string, displayName: string | null) => void
}

type Mode = 'signin' | 'register'
type Status = 'idle' | 'busy' | 'error'

/**
 * Sign-in / create-account page.
 *
 * BUG #6: when opened as part of the native handoff this page MUST use the
 * `code_challenge` from the query string. It never calls `codeChallengeS256`.
 *
 * The create-account path is what makes the app usable at all: without it a new
 * user has no credential, so "sign in" had nothing to offer and the app was a
 * dead end.
 */
export default function SignIn({ onAuthenticated }: Props) {
  const [mode, setMode] = useState<Mode>('signin')
  const [username, setUsername] = useState('')
  const [status, setStatus] = useState<Status>('idle')
  const [message, setMessage] = useState<string | null>(null)
  const [conditionalReady, setConditionalReady] = useState(false)
  const [addToExisting, setAddToExisting] = useState(false)
  const abortRef = useRef<AbortController | null>(null)

  const supported = useMemo(() => isWebAuthnAvailable(), [])

  const handoff = useMemo(() => {
    try {
      return readHandoff(window.location.search)
    } catch (err) {
      return { error: err instanceof Error ? err.message : String(err) } as const
    }
  }, [])

  useEffect(() => {
    if ('error' in handoff) return
    if (handoff.codeChallenge && !isValidChallenge(handoff.codeChallenge)) {
      setStatus('error')
      setMessage('This sign-in link carries a malformed code_challenge.')
    }
  }, [handoff])

  const finishWithToken = useCallback(
    (token: string, displayName: string | null) => {
      if ('error' in handoff) {
        onAuthenticated(token, displayName)
        return
      }
      if (handoff.native && handoff.redirectUri) {
        const url = new URL(handoff.redirectUri)
        url.searchParams.set('token', token)
        if (handoff.state) url.searchParams.set('state', handoff.state)
        window.location.replace(url.toString())
        return
      }
      onAuthenticated(token, displayName)
    },
    [handoff, onAuthenticated],
  )

  // ---- Conditional UI (autofill) ----------------------------------------- //
  useEffect(() => {
    let cancelled = false
    if (!supported) return

    void (async () => {
      const available = await isConditionalMediationAvailable()
      if (!available || cancelled) return
      setConditionalReady(true)

      try {
        const begin = await post<{
          challenge_id: string
          public_key: PublicKeyCredentialRequestOptionsJSON
        }>('/api/auth/login/begin', {})
        abortRef.current = new AbortController()
        const credential = (await navigator.credentials.get({
          publicKey: decodeRequestOptions(begin.public_key),
          mediation: 'conditional',
          signal: abortRef.current.signal,
        })) as PublicKeyCredential | null

        if (!credential || cancelled) return
        const done = await post<{ access_token: string; display_name: string | null }>(
          '/api/auth/login/finish',
          { challenge_id: begin.challenge_id, credential: serialiseCredential(credential) },
        )
        finishWithToken(done.access_token, done.display_name ?? null)
      } catch (err) {
        if ((err as Error)?.name !== 'AbortError' && !cancelled) setConditionalReady(false)
      }
    })()

    return () => {
      cancelled = true
      abortRef.current?.abort()
    }
  }, [finishWithToken, supported])

  async function createPasskey() {
    setStatus('busy')
    setMessage(null)
    try {
      // A conditional-mediation get() is almost certainly still pending from the
      // autofill effect. WebAuthn allows only ONE outstanding request per page, so
      // create() would fail with "A request is already pending". Cancel it first
      // and give the abort a beat to unwind before starting a new ceremony.
      abortRef.current?.abort()
      await new Promise((r) => setTimeout(r, 60))

      const begin = await post<{
        challenge_id: string
        public_key: PublicKeyCredentialCreationOptionsJSON
        user_id: string
      }>('/api/auth/register/begin', {
        username: username.trim() || null,
        display_name: username.trim() || null,
        add_to_existing: addToExisting,
      })

      const credential = (await navigator.credentials.create({
        publicKey: decodeCreationOptions(begin.public_key),
      })) as PublicKeyCredential | null
      if (!credential) throw new Error('Your device did not create a passkey.')

      const done = await post<{
        access_token: string
        display_name: string | null
        backed_up: boolean
      }>('/api/auth/register/finish', {
        challenge_id: begin.challenge_id,
        credential: serialiseCredential(credential),
      })

      if (!done.backed_up) {
        setMessage('Passkey created, but it is device-bound — losing this device locks you out.')
      }
      finishWithToken(done.access_token, done.display_name ?? null)
    } catch (err) {
      setStatus('error')
      setMessage(describePasskeyError(err, 'create'))
    }
  }

  async function signIn() {
    setStatus('busy')
    setMessage(null)
    try {
      // Same one-pending-request rule as createPasskey().
      abortRef.current?.abort()
      await new Promise((r) => setTimeout(r, 60))

      const body = username.trim() ? { username: username.trim() } : {}
      const begin = await post<{
        challenge_id: string
        public_key: PublicKeyCredentialRequestOptionsJSON
      }>('/api/auth/login/begin', body)

      const credential = (await navigator.credentials.get({
        publicKey: decodeRequestOptions(begin.public_key),
      })) as PublicKeyCredential | null
      if (!credential) throw new Error('No credential returned.')

      const loginFlags = readBackupFlags(credential)
      const done = await post<{ access_token: string; display_name: string | null }>(
        '/api/auth/login/finish',
        { challenge_id: begin.challenge_id, credential: serialiseCredential(credential) },
      )
      if (!loginFlags.backedUp) {
        setMessage('Note: this passkey is device-bound — losing this device locks you out.')
      }
      finishWithToken(done.access_token, done.display_name ?? null)
    } catch (err) {
      setStatus('error')
      setMessage(describePasskeyError(err, 'get'))
    }
  }

  if (!supported) {
    return (
      <main className="auth">
        <h1>volcast</h1>
        <p className="error">
          This browser does not support passkeys. Open this page in Safari (iOS 16+),
          Chrome (Android), or Edge/Firefox on desktop — and note that passkeys require
          HTTPS or localhost.
        </p>
        <DemoButton onAuthenticated={onAuthenticated} />
      </main>
    )
  }

  const isRegister = mode === 'register'
  const firstName = displayNameOrDefault(username)

  return (
    <main className="auth">
      <h1>volcast</h1>
      <p className="muted">
        {isRegister
          ? 'Create a passkey to start. No password, nothing to forget.'
          : 'Sign in with a passkey. No password, no username required.'}
      </p>

      <form
        onSubmit={(e) => {
          e.preventDefault()
          void (isRegister ? createPasskey() : signIn())
        }}
      >
        <label htmlFor="username">
          {isRegister ? 'Your name (optional)' : 'Username (optional)'}
        </label>
        <input
          id="username"
          name="username"
          type="text"
          autoComplete={!isRegister && conditionalReady ? 'username webauthn' : 'username'}
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          placeholder={isRegister ? 'e.g. Daniel' : 'leave blank for usernameless'}
          autoCapitalize="none"
          spellCheck={false}
        />

        {!isRegister && conditionalReady && (
          <p className="hint">A saved passkey should appear in this field's autofill menu.</p>
        )}

        <button type="submit" disabled={status === 'busy'}>
          {status === 'busy'
            ? 'Waiting for your device…'
            : isRegister
              ? `Create passkey${firstName ? ` for ${firstName}` : ''}`
              : 'Sign in with a passkey'}
        </button>
      </form>

      {isRegister && (
        <section className="secondary">
          <label>
            <input
              type="checkbox"
              checked={addToExisting}
              onChange={(e) => setAddToExisting(e.target.checked)}
            />
            Add to my existing account
          </label>
          <p className="hint">
            Tick this if you are already signed in elsewhere and want a second device on the
            same account instead of a new one.
          </p>
        </section>
      )}

      <p className="switch">
        {isRegister ? 'Already have a passkey?' : 'First time here?'}{' '}
        <button
          type="button"
          className="link"
          onClick={() => {
            setMode(isRegister ? 'signin' : 'register')
            setMessage(null)
            setStatus('idle')
          }}
        >
          {isRegister ? 'Sign in' : 'Create account'}
        </button>
      </p>

      {'error' in handoff && <p className="error">{handoff.error}</p>}
      {message && <p className={status === 'error' ? 'error' : 'hint'}>{message}</p>}

      <DemoButton onAuthenticated={onAuthenticated} />

      {!('error' in handoff) && handoff.codeChallenge && (
        <p className="hint mono">
          app handoff · code_challenge={handoff.codeChallenge.slice(0, 12)}… (S256, from the app)
        </p>
      )}
    </main>
  )
}

function displayNameOrDefault(raw: string): string {
  return raw.trim()
}

/**
 * Dev-only escape hatch: skip the passkey ceremony and look at the product.
 * The backend refuses this when env=prod, so it cannot weaken a real deployment.
 */
function DemoButton({
  onAuthenticated,
}: {
  onAuthenticated: (token: string, displayName: string | null) => void
}) {
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)

  async function go() {
    setBusy(true)
    setErr(null)
    try {
      const r = await post<{ access_token: string; display_name: string | null }>(
        '/api/auth/demo',
        {},
      )
      onAuthenticated(r.access_token, r.display_name ?? 'Demo')
    } catch (e) {
      setErr(e instanceof ApiError ? String(e.detail) : (e as Error).message)
      setBusy(false)
    }
  }

  return (
    <section className="secondary">
      <p className="hint">
        Just want to see it? Skip the passkey for now — this shortcut is disabled when the
        server runs in production mode.
      </p>
      <button type="button" className="ghost wide" onClick={() => void go()} disabled={busy}>
        {busy ? 'Opening…' : 'Look around without an account →'}
      </button>
      {err && <p className="error">{err}</p>}
    </section>
  )
}

/**
 * Turn a WebAuthn failure into something actionable.
 *
 * The browser deliberately gives almost no detail for security reasons, so the
 * useful signal is the error NAME. Mapping the few known cases here is the
 * difference between "it just says invalid domain" and knowing what to change.
 */
function describePasskeyError(err: unknown, op: 'create' | 'get'): string {
  if (err instanceof ApiError) return String(err.detail)
  const name = (err as Error)?.name ?? ''
  const msg = (err as Error)?.message ?? String(err)
  const origin = typeof window !== 'undefined' ? window.location.origin : ''

  if (name === 'NotAllowedError') {
    return op === 'create'
      ? 'Cancelled, or this device/browser refused the passkey. Try again and approve the prompt.'
      : 'Cancelled. If you have never created a passkey here, use "Create account" below.'
  }
  if (name === 'SecurityError') {
    return `This page's address (${origin}) is not a valid passkey domain. Passkeys need a real domain name such as http://localhost:8000 � an IP like 127.0.0.1 is rejected.`
  }
  if (name === 'NotSupportedError') {
    return 'This browser or device does not support the required passkey type.'
  }
  if (name === 'InvalidStateError') {
    return 'A passkey for this account already exists on this device. Try signing in instead.'
  }
  return `${name || 'Error'}: ${msg}`
}