import React, { useMemo } from 'react'

function PerformancePanel({ trades }) {
  const analytics = useMemo(() => {
    if (!trades || trades.length === 0) return null

    const sorted = [...trades].reverse() // oldest first

    // --- By direction breakdown ---
    const byDir = {}
    for (const t of sorted) {
      const dir = t.direction || 'UNKNOWN'
      if (!byDir[dir]) byDir[dir] = { wins: 0, losses: 0, totalBps: 0, trades: 0 }
      byDir[dir].trades++
      byDir[dir].totalBps += t.net_profit_bps || 0
      if ((t.net_profit_bps || 0) > 0) byDir[dir].wins++
      else byDir[dir].losses++
    }

    // --- Best / Worst trade ---
    let best = sorted[0], worst = sorted[0]
    for (const t of sorted) {
      if ((t.net_profit_bps || 0) > (best.net_profit_bps || 0)) best = t
      if ((t.net_profit_bps || 0) < (worst.net_profit_bps || 0)) worst = t
    }

    // --- Avg win / Avg loss ---
    const winners = sorted.filter(t => (t.net_profit_bps || 0) > 0)
    const losers = sorted.filter(t => (t.net_profit_bps || 0) <= 0)
    const avgWin = winners.length > 0 ? winners.reduce((s, t) => s + t.net_profit_bps, 0) / winners.length : 0
    const avgLoss = losers.length > 0 ? losers.reduce((s, t) => s + t.net_profit_bps, 0) / losers.length : 0

    // --- Consecutive W/L streaks ---
    let maxWinStreak = 0, maxLossStreak = 0, curWin = 0, curLoss = 0
    for (const t of sorted) {
      if ((t.net_profit_bps || 0) > 0) {
        curWin++
        curLoss = 0
        if (curWin > maxWinStreak) maxWinStreak = curWin
      } else {
        curLoss++
        curWin = 0
        if (curLoss > maxLossStreak) maxLossStreak = curLoss
      }
    }

    // --- Rolling Profit Factor (last N trades) ---
    const rollingPF = (n) => {
      const recent = sorted.slice(-n)
      const grossWin = recent.filter(t => (t.net_profit_bps || 0) > 0).reduce((s, t) => s + t.net_profit_bps, 0)
      const grossLoss = Math.abs(recent.filter(t => (t.net_profit_bps || 0) <= 0).reduce((s, t) => s + t.net_profit_bps, 0))
      return grossLoss > 0 ? grossWin / grossLoss : grossWin > 0 ? Infinity : 0
    }

    // --- Drawdown from equity curve ---
    let cumBps = 0, peak = 0, maxDD = 0
    const equityPoints = []
    for (const t of sorted) {
      cumBps += t.net_profit_bps || 0
      if (cumBps > peak) peak = cumBps
      const dd = peak - cumBps
      if (dd > maxDD) maxDD = dd
      equityPoints.push({ cum: cumBps, dd: peak - cumBps })
    }

    // --- By exit reason ---
    const byExit = {}
    for (const t of sorted) {
      const reason = t.exit_reason || 'UNKNOWN'
      if (!byExit[reason]) byExit[reason] = { count: 0, totalBps: 0, wins: 0 }
      byExit[reason].count++
      byExit[reason].totalBps += t.net_profit_bps || 0
      if ((t.net_profit_bps || 0) > 0) byExit[reason].wins++
    }

    return {
      byDir,
      best,
      worst,
      avgWin,
      avgLoss,
      maxWinStreak,
      maxLossStreak,
      rollingPF10: sorted.length >= 10 ? rollingPF(10) : null,
      rollingPF20: sorted.length >= 20 ? rollingPF(20) : null,
      maxDD,
      totalTrades: sorted.length,
      byExit,
    }
  }, [trades])

  if (!analytics) {
    return (
      <div className="card perf-section">
        <div className="card-header">Performance Analytics</div>
        <div className="perf-empty">No trades yet</div>
      </div>
    )
  }

  const fmtBps = (v) => {
    if (v == null) return '---'
    const sign = v >= 0 ? '+' : ''
    return `${sign}${v.toFixed(1)}`
  }

  const fmtPF = (v) => {
    if (v == null) return '---'
    if (v === Infinity) return '---'
    return v.toFixed(2)
  }

  const dirColors = {
    LONG: 'var(--accent-green)',
    SHORT: 'var(--accent-red)',
  }

  return (
    <div className="perf-section">
      {/* Key Stats */}
      <div className="card perf-card">
        <div className="card-header">Key Metrics</div>
        <div className="perf-metrics-grid">
          <div className="perf-metric">
            <span className="perf-metric-label">Best Trade</span>
            <span className="perf-metric-value positive">{fmtBps(analytics.best.net_profit_bps)} bps</span>
            <span className="perf-metric-sub">{analytics.best.direction}</span>
          </div>
          <div className="perf-metric">
            <span className="perf-metric-label">Worst Trade</span>
            <span className="perf-metric-value negative">{fmtBps(analytics.worst.net_profit_bps)} bps</span>
            <span className="perf-metric-sub">{analytics.worst.direction}</span>
          </div>
          <div className="perf-metric">
            <span className="perf-metric-label">Avg Win</span>
            <span className="perf-metric-value positive">{fmtBps(analytics.avgWin)} bps</span>
          </div>
          <div className="perf-metric">
            <span className="perf-metric-label">Avg Loss</span>
            <span className="perf-metric-value negative">{fmtBps(analytics.avgLoss)} bps</span>
          </div>
          <div className="perf-metric">
            <span className="perf-metric-label">Max Win Streak</span>
            <span className="perf-metric-value">{analytics.maxWinStreak}</span>
          </div>
          <div className="perf-metric">
            <span className="perf-metric-label">Max Loss Streak</span>
            <span className="perf-metric-value">{analytics.maxLossStreak}</span>
          </div>
          <div className="perf-metric">
            <span className="perf-metric-label">Max Drawdown</span>
            <span className="perf-metric-value negative">-{analytics.maxDD.toFixed(1)} bps</span>
          </div>
          {analytics.rollingPF10 != null && (
            <div className="perf-metric">
              <span className="perf-metric-label">Rolling PF (10)</span>
              <span className="perf-metric-value">{fmtPF(analytics.rollingPF10)}</span>
            </div>
          )}
        </div>
      </div>

      {/* Direction Breakdown */}
      <div className="card perf-card">
        <div className="card-header">By Direction</div>
        <div className="perf-dir-table">
          <div className="perf-dir-header">
            <span>Direction</span>
            <span>Trades</span>
            <span>Win%</span>
            <span>Total</span>
          </div>
          {Object.entries(analytics.byDir).map(([dir, d]) => (
            <div key={dir} className="perf-dir-row">
              <span style={{ color: dirColors[dir] || 'var(--text-primary)', fontWeight: 700 }}>{dir}</span>
              <span>{d.trades} ({d.wins}W/{d.losses}L)</span>
              <span>{d.trades > 0 ? ((d.wins / d.trades) * 100).toFixed(1) : 0}%</span>
              <span className={(d.totalBps || 0) >= 0 ? 'positive' : 'negative'}>
                {fmtBps(d.totalBps)} bps
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Exit Reason Breakdown */}
      <div className="card perf-card">
        <div className="card-header">By Exit Reason</div>
        <div className="perf-dir-table">
          <div className="perf-dir-header">
            <span>Reason</span>
            <span>Count</span>
            <span>Win%</span>
            <span>Total</span>
          </div>
          {Object.entries(analytics.byExit).map(([reason, d]) => (
            <div key={reason} className="perf-dir-row">
              <span className="perf-reason-name">{reason}</span>
              <span>{d.count}</span>
              <span>{d.count > 0 ? ((d.wins / d.count) * 100).toFixed(1) : 0}%</span>
              <span className={(d.totalBps || 0) >= 0 ? 'positive' : 'negative'}>
                {fmtBps(d.totalBps)} bps
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

export default PerformancePanel
