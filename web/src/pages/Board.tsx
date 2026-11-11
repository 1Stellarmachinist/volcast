import { useCallback, useEffect, useMemo, useState } from 'react'
import { ApiError, get, post } from '../auth/webauthn'

interface Card {
  ticker: string
  last: number
  change_1d: number | null
  rvol: number | null
  ma20_pos: number | null
  ma20_slope: number | null
  cross: boolean
  from_high: number | null
  signal: 'breakout' | 'trending' | 'volume_only' | 'quiet'
  score: number
  flags: string[]
  spark: number[]
}

interface BoardResponse {
  as_of: string | null
  rvol_min: number
  cards: Card[]
  note?: string
  error?: string
}

interface Props {
  token: string
  displayName: string | null
  onSignOut: () => void
}

type Filter = 'all' | 'breakout' | 'trending'
type SortKey = 'score' | 'rvol' | 'change_1d' | 'ticker'

function pct(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined) return '—'
  return `${v >= 0 ? '+' : ''}${(v * 100).toFixed(digits)}%`
}

function price(v: number): string {
  if (v >= 1) return `$${v.toFixed(2)}`
  if (v >= 0.01) return `$${v.toFixed(4)}`
  return `$${v.toFixed(5)}`
}

/**
 * Kalshi-style board: a dense grid of self-contained cards, each with one
 * dominant number, so the whole market reads at a glance instead of requiring a
 * table scan. Deliberately no charts on the cards beyond a sparkline.
 */
export default function Board({ token, displayName, onSignOut }: Props) {
  const [data, setData] = useState<BoardResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState<Filter>('all')
  const [sortKey, setSortKey] = useState<SortKey>('score')
  const [query, setQuery] = useState('')

  const withRefresh = useCallback(
    async <T,>(fn: (t: string) => Promise<T>): Promise<T> => {
      try {
        return await fn(token)
      } catch (err) {
        if (err instanceof ApiError && err.status === 401) {
          const r = await post<{ access_token: string }>('/api/auth/refresh/cookie', {})
          return fn(r.access_token)
        }
        throw err
      }
    },
    [token],
  )

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await withRefresh((t) => get<BoardResponse>('/api/market/penny', t))
      setData(res)
      if (res.error) setError(res.error)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setLoading(false)
    }
  }, [withRefresh])

  useEffect(() => {
    void load()
  }, [load])

  const cards = useMemo(() => {
    if (!data) return []
    let out = [...data.cards]
    if (filter === 'breakout') out = out.filter((c) => c.signal === 'breakout')
    if (filter === 'trending') {
      out = out.filter((c) => c.signal === 'trending' || c.signal === 'breakout')
    }
    if (query.trim()) {
      const q = query.trim().toUpperCase()
      out = out.filter((c) => c.ticker.includes(q))
    }
    out.sort((a, b) => {
      if (sortKey === 'ticker') return a.ticker.localeCompare(b.ticker)
      if (sortKey === 'rvol') return (b.rvol ?? 0) - (a.rvol ?? 0)
      if (sortKey === 'change_1d') return (b.change_1d ?? -9) - (a.change_1d ?? -9)
      return b.score - a.score
    })
    return out
  }, [data, filter, sortKey, query])

  const counts = useMemo(() => {
    const c = { breakout: 0, trending: 0, volume_only: 0, quiet: 0 }
    data?.cards.forEach((x) => {
      c[x.signal] += 1
    })
    return c
  }, [data])

  return (
    <main className="board">
      <header>
        <div>
          <h1>penny board</h1>
          <p className="muted">
            {displayName ?? 'signed in'} ·{' '}
            {data?.as_of ? `as of ${data.as_of.slice(0, 10)}` : 'loading'}
          </p>
        </div>
        <div className="head-actions">
          <button className="ghost" onClick={() => void load()}>
            Refresh
          </button>
          <button className="ghost" onClick={onSignOut}>
            Sign out
          </button>
        </div>
      </header>

      <section className="toolbar">
        <div className="segmented">
          {(['all', 'breakout', 'trending'] as Filter[]).map((f) => (
            <button
              key={f}
              className={filter === f ? 'seg active' : 'seg'}
              onClick={() => setFilter(f)}
            >
              {f === 'all' ? `All ${data?.cards.length ?? 0}` : f}
              {f === 'breakout' && counts.breakout ? ` ${counts.breakout}` : ''}
              {f === 'trending' && counts.trending + counts.breakout
                ? ` ${counts.trending + counts.breakout}`
                : ''}
            </button>
          ))}
        </div>
        <div className="toolbar-right">
          <input
            className="search"
            placeholder="filter ticker"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            spellCheck={false}
          />
          <select
            className="sort"
            value={sortKey}
            onChange={(e) => setSortKey(e.target.value as SortKey)}
          >
            <option value="score">score</option>
            <option value="rvol">volume</option>
            <option value="change_1d">change</option>
            <option value="ticker">ticker</option>
          </select>
        </div>
      </section>

      {error && <p className="error">{error}</p>}
      {loading && <p className="muted">Loading board…</p>}
      {!loading && !cards.length && !error && (
        <p className="muted">Nothing matches that filter.</p>
      )}

      <section className="grid">
        {cards.map((c) => (
          <article key={c.ticker} className={`card sig-${c.signal}`}>
            <div className="card-top">
              <span className="sym">{c.ticker}</span>
              <span className="px">{price(c.last)}</span>
            </div>

            <div className="big">{Math.round(c.score)}</div>
            <div className={`chg ${(c.change_1d ?? 0) >= 0 ? 'up' : 'down'}`}>
              {pct(c.change_1d, 2)} today
            </div>

            <Spark values={c.spark} />

            <div className="metrics">
              <div>
                <span className="k">volume</span>
                <span className="v">{c.rvol === null ? '—' : `${c.rvol.toFixed(1)}x`}</span>
              </div>
              <div>
                <span className="k">vs MA20</span>
                <span className={`v ${(c.ma20_pos ?? 0) >= 0 ? 'up' : 'down'}`}>
                  {c.ma20_pos === null ? '—' : pct(c.ma20_pos / 100, 1)}
                </span>
              </div>
              <div>
                <span className="k">from high</span>
                <span className="v">{pct(c.from_high, 0)}</span>
              </div>
            </div>

            <div className="card-foot">
              <span className={`tag t-${c.signal}`}>
                {c.signal.replace('_', ' ')}
              </span>
              {c.flags.slice(0, 2).map((f) => (
                <span key={f} className="tag t-warn">
                  {f.replace(/_/g, ' ')}
                </span>
              ))}
            </div>
          </article>
        ))}
      </section>

      {data?.note && <p className="hint footer">{data.note}</p>}
    </main>
  )
}

function Spark({ values }: { values: number[] }) {
  if (!values || values.length < 2) return <svg className="spark" viewBox="0 0 100 24" />
  const lo = Math.min(...values)
  const hi = Math.max(...values)
  const span = hi - lo || 1
  const pts = values
    .map((v, i) => {
      const x = (i / (values.length - 1)) * 100
      const y = 22 - ((v - lo) / span) * 20
      return `${x.toFixed(1)},${y.toFixed(1)}`
    })
    .join(' ')
  const rising = values[values.length - 1] >= values[0]
  return (
    <svg className="spark" viewBox="0 0 100 24" preserveAspectRatio="none">
      <polyline
        points={pts}
        fill="none"
        stroke={rising ? '#4ade80' : '#f87171'}
        strokeWidth="1.6"
        vectorEffect="non-scaling-stroke"
      />
    </svg>
  )
}
