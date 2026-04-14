import React, { useMemo } from 'react'

const PROFIT_THRESHOLDS = [200, 150, 100, 50, 25]
const LOSS_THRESHOLDS = [25, 50, 100]

const PROFIT_COLORS = { 200: '#14532d', 150: '#166534', 100: '#15803d', 50: '#16a34a', 25: '#22c55e' }
const LOSS_COLORS = { 25: '#f87171', 50: '#dc2626', 100: '#991b1b' }

function BpsDistribution({ trades, mlTrades, mlAttnTrades }) {
  const { profitBars, lossBars, total } = useMemo(() => {
    const v14 = Array.isArray(trades) ? trades : []
    const ml = Array.isArray(mlTrades) ? mlTrades : []
    const mlAttn = Array.isArray(mlAttnTrades) ? mlAttnTrades : []
    const all = [...v14, ...ml, ...mlAttn]
    const total = all.length
    if (total === 0) return { profitBars: [], lossBars: [], total: 0 }

    const bpsValues = all.map(t => t.net_profit_bps || 0)

    const profitBars = PROFIT_THRESHOLDS.map(level => {
      const count = bpsValues.filter(b => b >= level).length
      return { label: `\u2265${level}`, count, pct: (count / total * 100), color: PROFIT_COLORS[level] }
    })

    const lossBars = LOSS_THRESHOLDS.map(level => {
      const count = bpsValues.filter(b => b <= -level).length
      return { label: `\u2264-${level}`, count, pct: (count / total * 100), color: LOSS_COLORS[level] }
    })

    return { profitBars, lossBars, total }
  }, [trades, mlTrades, mlAttnTrades])

  if (total === 0) {
    return (
      <div className="card bps-dist">
        <div className="card-header">BPS Distribution (All Models)</div>
        <div className="trades-empty">No trades yet</div>
      </div>
    )
  }

  const allBars = [...profitBars, ...lossBars]
  const maxCount = Math.max(...allBars.map(b => b.count), 1)

  return (
    <div className="card bps-dist">
      <div className="bps-dist-header">
        <span className="card-header">BPS Distribution (All Models)</span>
        <span className="bps-dist-total">{total} trades</span>
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
