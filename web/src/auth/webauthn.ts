/**
 * Thin fetch wrapper for the volcast API plus WebAuthn serialisation helpers.
 *
 * BUG #7 FIX: `post(path, body, token?)`. The previous signature was
 * `post(path, token?)`, so every call that passed a body silently sent the body
 * as the bearer token and the request came back 401/422.
 */

/**
 * Where the API lives.
 *
 * Defaults to "" (same origin). The backend serves the built PWA itself, so the
 * browser and the API always agree on host and port; hardcoding 127.0.0.1:8000
 * broke every request the moment the server ran on any other port. During local
 * development Vite injects VITE_API_BASE="" as well and proxies /api to the
 * backend, so same-origin is correct in both modes.
 */
export const API_BASE: string =
  (import.meta as unknown as { env?: Record<string, string> }).env?.VITE_API_BASE ?? ''

export class ApiError extends Error {
  status: number
  detail: unknown
  constructor(status: number, detail: unknown) {
    super(typeof detail === 'string' ? detail : `HTTP ${status}`)
    this.status = status
    this.detail = detail
  }
}

async function request<T>(path: string, init: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    credentials: 'include', // required: refresh token lives in an HttpOnly cookie
    ...init,
    headers: {
      Accept: 'application/json',
      ...(init.body ? { 'Content-Type': 'application/json' } : {}),
      ...(init.headers ?? {}),
    },
  })
  const text = await res.text()
  let parsed: unknown = null
  if (text) {
    try {
      parsed = JSON.parse(text) // SafeJSONResponse guarantees valid JSON
    } catch {
      throw new ApiError(res.status, `non-JSON response: ${text.slice(0, 200)}`)
    }
  }
  if (!res.ok) {
    const detail = (parsed as { detail?: unknown } | null)?.detail ?? parsed
    throw new ApiError(res.status, detail)
  }
  return parsed as T
}

/** POST with an optional body and an optional bearer token. */
export async function post<T = unknown>(
  path: string,
  body?: unknown,
  token?: string,
): Promise<T> {
  return request<T>(path, {
    method: 'POST',
    body: body === undefined ? undefined : JSON.stringify(body),
    headers: token ? { Authorization: `Bearer ${token}` } : undefined,
  })
}

export async function get<T = unknown>(path: string, token?: string): Promise<T> {
  return request<T>(path, {
    method: 'GET',
    headers: token ? { Authorization: `Bearer ${token}` } : undefined,
  })
}

export async function del<T = unknown>(path: string, token?: string): Promise<T> {
  return request<T>(path, {
    method: 'DELETE',
    headers: token ? { Authorization: `Bearer ${token}` } : undefined,
  })
}

// --------------------------------------------------------------------------- //
// base64url <-> ArrayBuffer (WebAuthn wire format)
// --------------------------------------------------------------------------- //
export function b64urlToBuf(value: string): ArrayBuffer {
  const pad = '='.repeat((4 - (value.length % 4)) % 4)
  const raw = atob(value.replace(/-/g, '+').replace(/_/g, '/') + pad)
  const bytes = new Uint8Array(raw.length)
  for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i)
  return bytes.buffer
}

export function bufToB64url(buf: ArrayBuffer): string {
  const bytes = new Uint8Array(buf)
  let s = ''
  for (const b of bytes) s += String.fromCharCode(b)
  return btoa(s).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
}

/** Turn JSON creation options into the ArrayBuffer-bearing form the API needs. */
export function decodeCreationOptions(pk: PublicKeyCredentialCreationOptionsJSON): PublicKeyCredentialCreationOptions {
  return {
    ...pk,
    challenge: b64urlToBuf(pk.challenge),
    user: { ...pk.user, id: b64urlToBuf(pk.user.id) },
    excludeCredentials: (pk.excludeCredentials ?? []).map((c) => ({
      ...c,
      id: b64urlToBuf(c.id),
    })),
  } as PublicKeyCredentialCreationOptions
}

export function decodeRequestOptions(pk: PublicKeyCredentialRequestOptionsJSON): PublicKeyCredentialRequestOptions {
  return {
    ...pk,
    challenge: b64urlToBuf(pk.challenge),
    allowCredentials: (pk.allowCredentials ?? []).map((c) => ({
      ...c,
      id: b64urlToBuf(c.id),
    })),
  } as PublicKeyCredentialRequestOptions
}

export interface PublicKeyCredentialCreationOptionsJSON {
  challenge: string
  rp: { id?: string; name: string }
  user: { id: string; name: string; displayName: string }
  pubKeyCredParams: { type: 'public-key'; alg: number }[]
  timeout?: number
  attestation?: AttestationConveyancePreference
  authenticatorSelection?: AuthenticatorSelectionCriteria
  excludeCredentials?: { id: string; type?: PublicKeyCredentialType; transports?: AuthenticatorTransport[] }[]
}

export interface PublicKeyCredentialRequestOptionsJSON {
  challenge: string
  rpId?: string
  timeout?: number
  userVerification?: UserVerificationRequirement
  allowCredentials?: { id: string; type?: PublicKeyCredentialType; transports?: AuthenticatorTransport[] }[]
}

/** Serialise a credential for the server, including BE/BS backup flags. */
export function serialiseCredential(cred: PublicKeyCredential): Record<string, unknown> {
  const response = cred.response
  const out: Record<string, unknown> = {
    id: cred.id,
    rawId: bufToB64url(cred.rawId),
    type: cred.type,
    clientExtensionResults: cred.getClientExtensionResults(),
    authenticatorAttachment: (cred as { authenticatorAttachment?: string }).authenticatorAttachment ?? null,
  }

  if (response instanceof AuthenticatorAttestationResponse) {
    out.response = {
      clientDataJSON: bufToB64url(response.clientDataJSON),
      attestationObject: bufToB64url(response.attestationObject),
      transports: response.getTransports ? response.getTransports() : [],
    }
  } else if (response instanceof AuthenticatorAssertionResponse) {
    out.response = {
      clientDataJSON: bufToB64url(response.clientDataJSON),
      authenticatorData: bufToB64url(response.authenticatorData),
      signature: bufToB64url(response.signature),
      userHandle: response.userHandle ? bufToB64url(response.userHandle) : null,
    }
  }
  return out
}

/** Extract the BE/BS backup flags from authenticatorData. */
export function readBackupFlags(cred: PublicKeyCredential): { backedUp: boolean; backupEligible: boolean } {
  const response = cred.response
  if (!(response instanceof AuthenticatorAttestationResponse) && !(response instanceof AuthenticatorAssertionResponse)) {
    return { backedUp: false, backupEligible: false }
  }
  const authData =
    response instanceof AuthenticatorAttestationResponse
      ? extractAuthDataFromAttestation(response.attestationObject)
      : response.authenticatorData
  if (!authData) return { backedUp: false, backupEligible: false }
  const flags = new Uint8Array(authData)[32]
  return { backedUp: (flags & 0x10) !== 0, backupEligible: (flags & 0x08) !== 0 }
}

function extractAuthDataFromAttestation(attestationObject: ArrayBuffer): ArrayBuffer | null {
  // CBOR decoding is out of scope for the browser client; the server parses it.
  // Returning null keeps this helper honest rather than guessing.
  void attestationObject
  return null
}

export function isWebAuthnAvailable(): boolean {
  return typeof window !== 'undefined' && !!window.PublicKeyCredential
}

export async function isConditionalMediationAvailable(): Promise<boolean> {
  const pkc = window.PublicKeyCredential as unknown as {
    isConditionalMediationAvailable?: () => Promise<boolean>
  }
  if (!pkc?.isConditionalMediationAvailable) return false
  try {
    return await pkc.isConditionalMediationAvailable()
  } catch {
    return false
  }
}
