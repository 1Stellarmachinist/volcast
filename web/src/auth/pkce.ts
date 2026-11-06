/**
 * PKCE helpers for the browser-handoff flow.
 *
 * The VERIFIER is generated and kept by the native client. The browser only ever
 * sees the S256 hash (passed in the query string). Bug #6 was the /signin page
 * generating its own challenge, which could never match the native verifier, so
 * the handoff exchange always failed with "invalid_grant".
 */

const B64URL = /^[A-Za-z0-9\-_]{43,128}$/

/** Generate a high-entropy verifier (43-128 chars, base64url, RFC 7636 §4.1). */
export function createVerifier(byteLength = 32): string {
  const bytes = new Uint8Array(byteLength)
  crypto.getRandomValues(bytes)
  return toBase64Url(bytes)
}

/** S256 challenge for a verifier: BASE64URL(SHA256(ASCII(verifier))). */
export async function codeChallengeS256(verifier: string): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(verifier))
  return toBase64Url(new Uint8Array(digest))
}

export function toBase64Url(bytes: Uint8Array): string {
  let s = ''
  for (const b of bytes) s += String.fromCharCode(b)
  return btoa(s).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
}

export function fromBase64Url(value: string): Uint8Array {
  const pad = '='.repeat((4 - (value.length % 4)) % 4)
  const raw = atob(value.replace(/-/g, '+').replace(/_/g, '/') + pad)
  const out = new Uint8Array(raw.length)
  for (let i = 0; i < raw.length; i++) out[i] = raw.charCodeAt(i)
  return out
}

/** A well-formed S256 challenge as produced by the native client. */
export function isValidChallenge(value: string | null): value is string {
  return !!value && B64URL.test(value) && value.length === 43
}

/**
 * Read the handoff parameters the native app appended to the /signin URL.
 * `code_challenge` is consumed as-is; it is NEVER regenerated here.
 */
export interface HandoffParams {
  codeChallenge: string | null
  state: string | null
  redirectUri: string | null
  native: boolean
}

export function readHandoff(search: string): HandoffParams {
  const q = new URLSearchParams(search)
  const codeChallenge = q.get('code_challenge')
  const method = (q.get('code_challenge_method') ?? 'S256').toUpperCase()
  if (codeChallenge && method !== 'S256') {
    throw new Error(`unsupported PKCE method: ${method} (only S256 is accepted)`)
  }
  return {
    codeChallenge,
    state: q.get('state'),
    redirectUri: q.get('redirect_uri'),
    native: q.get('native') === '1' || q.get('handoff') === '1',
  }
}
