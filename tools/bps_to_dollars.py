"""Convert bps to dollars: what a spread actually costs on a real position."""

BPS = 0.0001  # one basis point = 0.01% = 1/100th of a percent


def bps_to_dollars(position_dollars: float, bps: float) -> float:
    """Dollar cost of crossing `bps` on a position of `position_dollars`."""
    return position_dollars * bps * BPS


def dollars_to_bps(dollars: float, position_dollars: float) -> float:
    if position_dollars <= 0:
        return float("nan")
    return dollars / position_dollars / BPS


def main() -> None:
    print("=" * 62)
    print("  1 basis point = 0.01% = $1 per $10,000")
    print("=" * 62)
    print()

    print("WHAT A ROUND TRIP COSTS, BY POSITION SIZE")
    print("-" * 62)
    print(f"{'position':>11s} {'50bp':>10s} {'150bp':>10s} {'300bp':>10s}")
    for pos in (100, 500, 1_000, 5_000, 10_000, 50_000):
        row = [bps_to_dollars(pos, b) for b in (50, 150, 300)]
        print(f"${pos:>10,} " + " ".join(f"${c:>9,.2f}" for c in row))
    print()

    print("ANNUAL DRAG: 150bp round trip, trading every 5 days (~50/yr)")
    print("-" * 62)
    print(f"{'position':>11s} {'per trip':>12s} {'per year':>12s} {'% of capital':>14s}")
    for pos in (1_000, 5_000, 10_000, 50_000):
        per = bps_to_dollars(pos, 150)
        year = per * 50
        print(f"${pos:>10,} ${per:>11,.2f} ${year:>11,.2f} {year/pos:>13.1%}")
    print()

    print("PENNY STOCK REALITY CHECK")
    print("-" * 62)
    print("  1 cent is the minimum tick. At $1.41 (SNDL):")
    print(f"    1c spread = {dollars_to_bps(0.01, 1.41):.0f} bps")
    print(f"    on a $1,000 position that's ${bps_to_dollars(1000, 71):.2f} each way")
    print()
    print("  At $0.58 (NAK):")
    print(f"    1c spread = {dollars_to_bps(0.01, 0.58):.0f} bps")
    print(f"    on a $1,000 position that's ${bps_to_dollars(1000, 172):.2f} each way")
    print()

    print("THE PENNY-STOCK TRAP")
    print("-" * 62)
    print("  bps is a RELATIVE measure, so a cheap stock looks expensive in bps")
    print("  for a reason that has nothing to do with liquidity:")
    print()
    print("    minimum tick is 1 cent, so 1c / price = bps")
    print("    $0.58 -> 172 bps      $10.00 -> 10 bps")
    print()
    print("  A $1 stock CANNOT have a tight spread in bps. The exchange rule")
    print("  makes it structurally wide. This is why penny names are a bad")
    print("  fit for any strategy that trades often.")
    print()
    print("  The fix is not better signalling. It is trading far less often,")
    print("  or not trading these names at all.")


if __name__ == "__main__":
    main()
