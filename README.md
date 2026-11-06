# volcast

Short-horizon volatility forecasting and net-of-cost strategy research for
high-volatility equities, built to run on a low-spec machine.

> **Hardware note.** This project is developed on a 2-core Intel Celeron 4205U
> with 8 GB of RAM and no GPU. The default engine is the analytic `drift`
> baseline. Amazon Chronos-2 is wired up behind an optional extra but is *not*
> installed here: single-threaded CPU inference would take minutes-to-hours per
> forecast, which is not a useful feedback loop. See `tools/bench.py`.

## Install

```powershell
python -m venv .venv
.venv\Scripts\pip install -e ".[api,dev]"
```

`torch` and `chronos-forecasting` are deliberately **not** installed. If you ever
move to a GPU machine: `pip install -e ".[chronos]"` and pass `--engine chronos`.

## Use

```powershell
.venv\Scripts\python tools\verify.py          # self-check: 14 bug regressions + behaviour
.venv\Scripts\python -m pytest -q             # unit tests
.venv\Scripts\python -m volcast forecast      # forecast table
.venv\Scripts\python -m volcast strat -c configs/lowspec.toml
.venv\Scripts\python tools\launch.py --engine drift
.venv\Scripts\python tools\bench.py
```

## What the strategy actually does

1. **Forecast** — `drift`: `sigma_h = realised_vol(window) * sqrt(h)`. Boring on
   purpose; it is the honest baseline any fancier model has to beat.
2. **Signal** — cross-sectional: low realised vol ranks attractive, 63-day
   momentum sets direction.
3. **Size** — vol target: `w = target_vol / annualised(sigma_h)`, clipped to
   `max_weight`, then rescaled so gross exposure is 1.0.
4. **Constrain** — weights are dollar-neutral *and* capped. Those two constraints
   are satisfied jointly (alternating projections), not sequentially, because
   clipping a centred book re-introduces net exposure.
5. **Charge costs** — `cost_bps` on turnover, every rebalance.

### Reading the results

A Sharpe ratio is a point estimate. This repo always reports `sharpe_se` next to
it; if `|sharpe| < 2 * sharpe_se` the result is statistically indistinguishable
from zero regardless of how good the number looks. With 126 out-of-sample days
the standard error is roughly ±1.3, so almost nothing short of Sharpe ~2.6 is
evidence of anything.

Zero-variance returns produce a **NaN** Sharpe, not a large one: `0/0` is
undefined and pretending otherwise is how backtests lie.

## Layout

```
volcast/            library: data, indicators, panel, forecast, signals, sizing,
                    backtest, evaluate, finetune, plot
volcast/api/        FastAPI backend (optional "api" extra)
volcast/commands/   CLI subcommands: forecast, screen, strat, cs, ft
tools/              verify, launch, make_shortcut, bench
tests/              pytest suite
web/                Vite + React + PWA client (passkey auth, forecast table)
configs/lowspec.toml  tuned for 2 cores / 8 GB / no GPU
```

## Ranking small caps (`volcast pick`)

`volcast pick` ranks a universe on the things historically *associated* with
small-cap outperformance, and then tells you whether that ranking has ever worked.

```powershell
.venv\Scripts\python -m volcast pick -c configs/penny.toml --evaluate
```

### Two kinds of signal, deliberately kept apart

| Kind | Source | Backtestable? |
|---|---|---|
| Price/volume (momentum, vol regime, distance from 52w low) | computed at each date | **Yes** — point-in-time safe |
| Fundamentals (revenue growth, margins, cash/debt, short %) | yfinance `.info` = **today's snapshot** | **No** |

`volcast/ranking.py` never imports the fundamentals module. A test enforces this.
Scoring historical dates with today's revenue growth would leak the future into
the past and manufacture an edge that does not exist — the single easiest way for
a backtest to lie.

### What the evaluation actually says

Run on 9 sub-$5 names over 2 years, 60 rebalances, 5-day horizon:

```
in_sample   n=30  excess=+0.491%  SE=1.134%  t=+0.43  beat-universe=50%
out_sample  n=30  excess=+0.933%  SE=0.880%  t=+1.06  beat-universe=50%
all         n=60  excess=+0.712%  SE=0.712%  t=+1.00  beat-universe=50%
```

The score is **monotonic** across buckets — top third −0.77%, middle −1.43%,
bottom −2.23% — which is mildly encouraging and consistent with momentum working
in small caps. But **t = 1.00** and a **50% hit rate** mean the ranking is not
statistically distinguishable from noise at this sample size. It is a shortlist,
not an oracle.

Note also that all three buckets are negative: over this window the universe fell
overall, and the ranking merely lost less. Beating a falling market is not the
same as making money.

### Hard limits you should know

- **Survivorship.** Only names that still exist can be tested. `MULN`, `GOEV`,
  `FFIE` and `RIVM` were already delisted (404) when the universe was built.
  Real portfolios would have held some of those.
- **Liquidity.** The screen requires ≥ $500k average daily dollar volume and
  price ≥ $0.30. Below that you cannot get filled without moving the price, and
  the model's assumption of frictionless trading breaks.
- **Costs dominate.** See the cost-sensitivity table in the penny profile: a
  Sharpe of 3.2 at 25bps becomes 1.3 at 250bps and −0.6 at 500bps. Microcap
  spreads are realistically 100–250bps round trip.

### If you want to improve it

The most promising untested idea is **cross-sectional** scoring across a much
larger universe (hundreds of names) rather than ranking 9. Nine names is too few
for a ranking to mean much — there is no cross-section to exploit. Getting a
broader universe needs a data source beyond yfinance's free tier.

## Auth model


Passkeys (WebAuthn) only — no passwords.

* `residentKey: required`, `userVerification: required` — discoverable
  credentials, so sign-in is **usernameless**: the server sends
  `allowCredentials: []` and the authenticator offers whatever it holds for the
  relying party.
* Conditional UI on the web (`autocomplete="username webauthn"`), so the passkey
  appears in the browser's autofill dropdown instead of a modal on page load.
* Adding a second device reuses the signed-in user (`add_to_existing`) rather than
  minting a new account.
* `credential_backed_up` is surfaced in the UI: a single device-bound passkey
  means losing the phone locks you out, and the user should know that.
* Access token: 15 min JWT. Refresh token: 30 days, rotated on every use with
  reuse detection (reusing an old token revokes the whole family).
* Native app handoff uses authorization code + **PKCE S256**. The app generates
  the verifier and sends only the hash; the `/signin` page uses the incoming
  `code_challenge` verbatim and never generates its own.

Cookie policy: `Secure` is set only when `env=prod`. A `Secure` cookie over
plain `http://localhost` is silently dropped by the browser, which previously
made every local session expire after 15 minutes.

## Regression notes

Fourteen specific bugs were fixed and are now covered by `tools/verify.py`:

1. `drift_forecast` returns a finite positive `sigma_h` (NaN previously made the
   sizer take zero positions silently).
2. `_clean()` recursively replaces NaN/inf with `null` on every payload;
   `SafeJSONResponse` uses `allow_nan=False` as a second line of defence.
3. `security.decode_access_claims(cfg, token) -> dict` exists as the single
   decoding path; `read_access_token` wraps it.
4. Refresh accepts passkey subjects — `_known_subject` checks `passkey_users`
   *and* `devices`.
5. `refresh(body, request, response)` is 3-arg; `refresh_cookie` reads the
   `volcast_rt` cookie and forwards it.
6. PKCE: `/signin` uses the incoming `code_challenge`.
7. `webauthn.ts` `post(path, body, token?)`.
8. `cookie_secure` defaults to `is_prod`, not `True`.
9. `metrics()` returns NaN Sharpe on zero variance; tests assert `isnan`.
10. `pinball(actual, forecast, quantile)` is minimised at the claimed quantile:
    `pinball(0.01, 0.01, 0.9) == 0` and `pinball(0.03, 0.01, 0.9) == 0.9 * 0.02`.
11. `cross_sectional_weights` keeps the book dollar-neutral *after* capping.
12. `begin_registration` tolerates py_webauthn builds without
    `exclude_credentials` (`try/except TypeError`).
13. No `fillna(method="ffill")` anywhere — `.ffill()` only (pandas 2.2+ removed
    the `method` argument; this build runs on pandas 3.x).
14. `pyproject.toml` has a `[project.optional-dependencies] api` extra.

## License

MIT
