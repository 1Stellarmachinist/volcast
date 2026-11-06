import { useCallback, useEffect, useMemo, useState } from 'react'
import { ApiError, get, post } from '../auth/webauthn'

interface ForecastRow {
  ticker: string
  sigma: number | null
  sigma_h: number | null
  ann_vol: number | null
  mu: number | null
}
interface ForecastResponse {
  as_of: string
  engine: string
  horizon: number
  rows: ForecastRow[]
}
interface BacktestResponse {
  engine: string
  days: number
  sharpe: number | null
  sharpe_se: number | null
  gross_sharpe: number | null
  ann_return: number | null
  ann_vol: number | null
  max_drawdown: number | null
  total_cost: number | null
  significant: boolean | null
  equity: number[]
  dates: string[]
}

interface Props {
  token: string
  displayName: string | null
  onSignOut: () => void
}

type SortKey = 'ticker' | 'ann_vol' | 'sigma_h' | 'mu'

/** NaN arrives as null (the API sanitises it); show a dash, never "NaN". */
function pct(v: number | null | undefined, digits = 1): string {
  return v === null || v === undefined ? '—' : `${(v * 100).toFixed(digits)}%`
}

function num(v: number | null | undefined, digits = 2): string {
  return v === null || v === undefined ? '—' : v.toFixed(digits)
}

export default function Dashboard({ token, displayName, onSignOut }: Props) {
  const [forecast, setForecast] = useState<ForecastResponse | null>(null)
  const [backtest, setBacktest] = useState<BacktestResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [btLoading, setBtLoading] = useState(false)
  const [sortKey, setSortKey] = useState<SortKey>('ann_vol')
  const [asc, setAsc] = useState(false)

  /** Refresh the access token from the HttpOnly cookie, then retry once. */
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

  useEffect(() => {
    let cancelled = false
    void (async () => {
      try {
        const res = await withRefresh((t) =>
          get<ForecastResponse>('/api/market/forecast?horizon=5', t),
        )
        if (!cancelled) setForecast(res)
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err))
      } finally {
        if (!cancelled) setLoading(false)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [withRefresh])

  async function runBacktest() {
    setBtLoading(true)
    setError(null)
    try {
      const res = await withRefresh((t) =>
        get<BacktestResponse>('/api/market/backtest?test_days=126&cost_bps=5', t),
      )
      setBacktest(res)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBtLoading(false)
    }
  }

  const rows = useMemo(() => {
    if (!forecast) return []
    const copy = [...forecast.rows]
    copy.sort((a, b) => {
      const av = a[sortKey]
      const bv = b[sortKey]
      const cmp =
        typeof av === 'string' && typeof bv === 'string'
          ? av.localeCompare(bv)
          : (Number(av ?? -Infinity) || 0) - (Number(bv ?? -Infinity) || 0)
      return asc ? cmp : -cmp
    })
    return copy
  }, [forecast, sortKey, asc])

  const maxVol = Math.max(0.0001, ...rows.map((r) => r.ann_vol ?? 0))

  function toggleSort(key: SortKey) {
    if (key === sortKey) setAsc(!asc)
    else {
      setSortKey(key)
      setAsc(key === 'ticker')
    }
  }

  return (
    <main className="app">
      <header>
        <div>
          <h1>volcast</h1>
          <p className="muted">
            {displayName ? displayName : 'signed in'} · drift engine ·{' '}
            {forecast ? forecast.horizon : 5}-day horizon
          </p>
        </div>
        <button className="ghost" onClick={onSignOut}>
          Sign out
        </button>
      </header>

      {error && <p className="error">{error}</p>}
      {loading && <p className="muted">Loading…</p>}

      {forecast && (
        <>
          <section className="cards">
            <div className="card">
              <span className="k">Most volatile</span>
              <span className="v">{rows[0]?.ticker ?? '—'}</span>
              <span className="sub">{pct(rows[0]?.ann_vol)} annualised</span>
            </div>
            <div className="card">
              <span className="k">Calmest</span>
              <span className="v">{rows[rows.length - 1]?.ticker ?? '—'}</span>
              <span className="sub">
                {pct(rows[rows.length - 1]?.ann_vol)} annualised
              </span>
            </div>
            <div className="card">
              <span className="k">Universe</span>
              <span className="v">{rows.length}</span>
              <span className="sub">as of {forecast.as_of.slice(0, 10)}</span>
            </div>
          </section>

          <section>
            <h2>Forecast volatility</h2>
            <p className="hint">
              σ<sub>h</sub> = realised σ × √h. Ranked high to low — these are the names
              most likely to move a lot over the next {forecast.horizon} days.
            </p>
            <table>
              <thead>
                <tr>
                  <th onClick={() => toggleSort('ticker')} className="sortable">
                    Ticker {sortKey === 'ticker' ? (asc ? '▲' : '▼') : ''}
                  </th>
                  <th onClick={() => toggleSort('ann_vol')} className="sortable">
                    Vol (ann.) {sortKey === 'ann_vol' ? (asc ? '▲' : '▼') : ''}
                  </th>
                  <th className="barcol">relative</th>
                  <th onClick={() => toggleSort('sigma_h')} className="sortable">
                    σ over {forecast.horizon}d {sortKey === 'sigma_h' ? (asc ? '▲' : '▼') : ''}
                  </th>
                  <th onClick={() => toggleSort('mu')} className="sortable">
                    Drift {sortKey === 'mu' ? (asc ? '▲' : '▼') : ''}
                  </th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.ticker}>
                    <td className="mono">{r.ticker}</td>
                    <td>{pct(r.ann_vol)}</td>
                    <td className="barcol">
                      <span
                        className="bar"
                        style={{ width: `${Math.round(((r.ann_vol ?? 0) / maxVol) * 100)}%` }}
                      />
                    </td>
                    <td>{pct(r.sigma_h, 2)}</td>
                    <td className={((r.mu ?? 0) < 0 ? 'neg' : 'pos')}>{pct(r.mu, 2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>

          <section className="bt">
            <div className="bt-head">
              <h2>Does the signal actually work?</h2>
              <button onClick={runBacktest} disabled={btLoading}>
                {btLoading ? 'Running…' : 'Run backtest'}
              </button>
            </div>
            <p className="hint">
              A Sharpe ratio on its own is meaningless without its standard error. This
              reports both, and says plainly when the result is noise.
            </p>

            {backtest && (
              <>
                <div className="cards">
                  <div className="card">
                    <span className="k">Sharpe</span>
                    <span className="v">{num(backtest.sharpe, 3)}</span>
                    <span className="sub">± {num(backtest.sharpe_se, 3)} (1 SE)</span>
                  </div>
                  <div className="card">
                    <span className="k">Annual return</span>
                    <span className="v">{pct(backtest.ann_return)}</span>
                    <span className="sub">vol {pct(backtest.ann_vol)}</span>
                  </div>
                  <div className="card">
                    <span className="k">Max drawdown</span>
                    <span className="v">{pct(backtest.max_drawdown)}</span>
                    <span className="sub">{backtest.days} days tested</span>
                  </div>
                  <div className="card">
                    <span className="k">Costs paid</span>
                    <span className="v">{pct(backtest.total_cost, 2)}</span>
                    <span className="sub">gross Sharpe {num(backtest.gross_sharpe, 3)}</span>
                  </div>
                </div>

                <p className={backtest.significant ? 'verdict good' : 'verdict bad'}>
                  {backtest.significant
                    ? 'Statistically significant: |Sharpe| exceeds 2 standard errors.'
                    : 'Not significant. |Sharpe| is under 2 standard errors, so this result is indistinguishable from luck.'}
                </p>

                <EquityCurve equity={backtest.equity} />
              </>
            )}
          </section>

          <p className="hint footer">
            drift baseline — no neural network, no GPU. On this machine that is the only
            engine that returns in under a second.
          </p>
        </>
      )}
    </main>
  )
}

function EquityCurve({ equity }: { equity: number[] }) {
  if (equity.length < 2) return null
  const w = 640
  const h = 120
  const lo = Math.min(...equity)
  const hi = Math.max(...equity)
  const span = hi - lo || 1
  const pts = equity
    .map((v, i) => {
      const x = (i / (equity.length - 1)) * w
      const y = h - ((v - lo) / span) * h
      return `${x.toFixed(1)},${y.toFixed(1)}`
    })
    .join(' ')

  return (
    <svg className="equity" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" role="img"
         aria-label="Equity curve">
      <polyline points={pts} fill="none" stroke="#4ade80" strokeWidth="1.5" />
    </svg>
  )
}
