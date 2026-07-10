import React, { useMemo } from 'react'

const PROFIT_THRESHOLDS = [200, 150, 100, 50, 25]
const LOSS_THRESHOLDS = [25, 50, 100]

const PROFIT_COLORS = { 200: '#14532d', 150: '#166534', 100: '#15803d', 50: '#16a34a', 25: '#22c55e' }
const LOSS_COLORS = { 25: '#f87171', 50: '#dc2626', 100: '#991b1b' }

function BpsDistribution({ trades, mlTrades, mlAttnTrades, mlV3Trades }) {
  const { profitBars, lossBars, totalDays } = useMemo(() => {
    const v14 = Array.isArray(trades) ? trades : []
    const ml = Array.isArray(mlTrades) ? mlTrades : []
    const mlAttn = Array.isArray(mlAttnTrades) ? mlAttnTrades : []
    const mlV3 = Array.isArray(mlV3Trades) ? mlV3Trades : []
    const all = [...v14, ...ml, ...mlAttn, ...mlV3]

    if (all.length === 0) return { profitBars: [], lossBars: [], totalDays: 0 }

    // Group by date -> sum net_profit_bps
    const dailyPnl = {}
    for (const t of all) {
      if (!t.exit_time) continue
      const d = new Date(t.exit_time)
      const dateKey = `${d.getFullYear()}-${d.getMonth()}-${d.getDate()}`
      dailyPnl[dateKey] = (dailyPnl[dateKey] || 0) + (t.net_profit_bps || 0)
    }

    const dailyValues = Object.values(dailyPnl)
    const totalDays = dailyValues.length

    if (totalDays === 0) return { profitBars: [], lossBars: [], totalDays: 0 }

    const profitBars = PROFIT_THRESHOLDS.map(level => {
      const count = dailyValues.filter(pnl => pnl >= level).length
      return { label: `\u2265${level}`, count, pct: (count / totalDays * 100), color: PROFIT_COLORS[level] }
    })

    const lossBars = LOSS_THRESHOLDS.map(level => {
      const count = dailyValues.filter(pnl => pnl <= -level).length
      return { label: `\u2264-${level}`, count, pct: (count / totalDays * 100), color: LOSS_COLORS[level] }
    })

    return { profitBars, lossBars, totalDays }
  }, [trades, mlTrades, mlAttnTrades, mlV3Trades])

  if (totalDays === 0) {
    return (
      <div className="card bps-dist">
        <div className="card-header">Daily PnL Distribution (All Models)</div>
        <div className="trades-empty">No trades yet</div>
      </div>
    )
  }

  const allBars = [...profitBars, ...lossBars]
  const maxCount = Math.max(...allBars.map(b => b.count), 1)

  return (
    <div className="card bps-dist">
      <div className="bps-dist-header">
        <span className="card-header">Daily PnL Distribution (All Models)</span>
        <span className="bps-dist-total">{totalDays} days</span>
      </div>
      <div className="bps-dist-bars">
        {profitBars.map(({ label, count, pct, color }) => (
          <div className="bps-bar-row" key={label}>
            <span className="bps-bar-label">{label}</span>
            <div className="bps-bar-track">
              <div
                className="bps-bar-fill"
                style={{ width: `${(count / maxCount) * 100}%`, background: color }}
              />
            </div>
            <span className="bps-bar-count">{count}</span>
            <span className="bps-bar-pct">({pct.toFixed(0)}%)</span>
          </div>
        ))}

        <div className="bps-dist-divider" />

        {lossBars.map(({ label, count, pct, color }) => (
          <div className="bps-bar-row" key={label}>
            <span className="bps-bar-label">{label}</span>
            <div className="bps-bar-track">
              <div
                className="bps-bar-fill"
                style={{ width: `${(count / maxCount) * 100}%`, background: color }}
              />
            </div>
            <span className="bps-bar-count">{count}</span>
            <span className="bps-bar-pct">({pct.toFixed(0)}%)</span>
          </div>
        ))}
      </div>
    </div>
  )
}

export default BpsDistribution
