import React, { useMemo } from 'react'

const SIGNAL_COLORS = {
  V12_LONG: '#00d97e',
  V12_SHORT: '#e63757',
  BEAR_LONG: '#3b82f6',
  BULL_SHORT: '#f59e0b',
  ML_LONG: '#8b5cf6',
  ML_SHORT: '#ec4899',
}

function PerformancePanel({ trades, mlTrades, mlAttnTrades, mlV3Trades }) {
  const analytics = useMemo(() => {
    const allTrades = [...(trades || []), ...(mlTrades || []), ...(mlAttnTrades || []), ...(mlV3Trades || [])]
    if (allTrades.length === 0) return null

    const sorted = [...allTrades].reverse() // oldest first

    // --- By direction ---
    const byDir = {}
    for (const t of sorted) {
      const dir = t.direction || 'UNKNOWN'
      if (!byDir[dir]) byDir[dir] = { wins: 0, losses: 0, totalBps: 0, trades: 0 }
      byDir[dir].trades++
      byDir[dir].totalBps += t.net_profit_bps || 0
      if ((t.net_profit_bps || 0) > 0) byDir[dir].wins++
      else byDir[dir].losses++
    }

    // --- By signal type ---
    const bySignal = {}
    for (const t of sorted) {
      const sig = t.signal_type || 'UNKNOWN'
      if (!bySignal[sig]) bySignal[sig] = { wins: 0, losses: 0, totalBps: 0, trades: 0, avgBps: 0 }
      bySignal[sig].trades++
      bySignal[sig].totalBps += t.net_profit_bps || 0
      if ((t.net_profit_bps || 0) > 0) bySignal[sig].wins++
      else bySignal[sig].losses++
    }
    for (const sig of Object.keys(bySignal)) {
      bySignal[sig].avgBps = bySignal[sig].trades > 0 ? bySignal[sig].totalBps / bySignal[sig].trades : 0
    }

    // --- Best / Worst ---
    let best = sorted[0], worst = sorted[0]
    for (const t of sorted) {
      if ((t.net_profit_bps || 0) > (best.net_profit_bps || 0)) best = t
      if ((t.net_profit_bps || 0) < (worst.net_profit_bps || 0)) worst = t
    }

    // --- Avg win / loss ---
    const winners = sorted.filter(t => (t.net_profit_bps || 0) > 0)
    const losers = sorted.filter(t => (t.net_profit_bps || 0) <= 0)
    const avgWin = winners.length > 0 ? winners.reduce((s, t) => s + (t.net_profit_bps || 0), 0) / winners.length : 0
    const avgLoss = losers.length > 0 ? losers.reduce((s, t) => s + (t.net_profit_bps || 0), 0) / losers.length : 0

    // --- Streaks ---
    let maxWinStreak = 0, maxLossStreak = 0, curWin = 0, curLoss = 0
    for (const t of sorted) {
      if ((t.net_profit_bps || 0) > 0) {
        curWin++; curLoss = 0
        if (curWin > maxWinStreak) maxWinStreak = curWin
      } else {
        curLoss++; curWin = 0
        if (curLoss > maxLossStreak) maxLossStreak = curLoss
      }
    }

    // --- Drawdown ---
    let cumBps = 0, peak = 0, maxDD = 0
    for (const t of sorted) {
      cumBps += t.net_profit_bps || 0
      if (cumBps > peak) peak = cumBps
      const dd = peak - cumBps
      if (dd > maxDD) maxDD = dd
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

    return { byDir, bySignal, best, worst, avgWin, avgLoss, maxWinStreak, maxLossStreak, maxDD, totalTrades: sorted.length, byExit }
  }, [trades, mlTrades, mlAttnTrades, mlV3Trades])

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

  const dirColors = { LONG: 'var(--accent-green)', SHORT: 'var(--accent-red)' }

  return (
    <div className="perf-section">
      {/* Key Stats — 4 column */}
      <div className="card perf-card">
        <div className="card-header">Key Metrics</div>
        <div className="perf-metrics-grid">
          <div className="perf-metric">
            <span className="perf-metric-label">Best Trade</span>
            <span className={`perf-metric-value ${(analytics.best.net_profit_bps || 0) >= 0 ? 'positive' : 'negative'}`}>{fmtBps(analytics.best.net_profit_bps)}</span>
          </div>
          <div className="perf-metric">
            <span className="perf-metric-label">Worst Trade</span>
            <span className={`perf-metric-value ${(analytics.worst.net_profit_bps || 0) >= 0 ? 'positive' : 'negative'}`}>{fmtBps(analytics.worst.net_profit_bps)}</span>
          </div>
          <div className="perf-metric">
            <span className="perf-metric-label">Avg Win</span>
            <span className="perf-metric-value positive">{analytics.avgWin > 0 ? fmtBps(analytics.avgWin) : '---'}</span>
          </div>
          <div className="perf-metric">
            <span className="perf-metric-label">Avg Loss</span>
            <span className="perf-metric-value negative">{analytics.avgLoss < 0 ? fmtBps(analytics.avgLoss) : '---'}</span>
          </div>
          <div className="perf-metric">
            <span className="perf-metric-label">Win Streak</span>
            <span className="perf-metric-value">{analytics.maxWinStreak}</span>
          </div>
          <div className="perf-metric">
            <span className="perf-metric-label">Loss Streak</span>
            <span className="perf-metric-value">{analytics.maxLossStreak}</span>
          </div>
          <div className="perf-metric">
            <span className="perf-metric-label">Max Drawdown</span>
            <span className="perf-metric-value negative">-{analytics.maxDD.toFixed(1)}</span>
          </div>
          <div className="perf-metric">
            <span className="perf-metric-label">Payoff Ratio</span>
            <span className="perf-metric-value">
              {analytics.avgLoss < 0 ? (analytics.avgWin / Math.abs(analytics.avgLoss)).toFixed(2) : '---'}
            </span>
          </div>
        </div>
      </div>

      {/* Two tables side by side */}
      <div className="perf-tables-row">
        <div className="card perf-card perf-table-half">
          <div className="card-header">By Direction</div>
          <div className="perf-dir-table">
            <div className="perf-dir-header">
              <span>Direction</span><span>Trades</span><span>Win%</span><span>Total</span>
            </div>
            {Object.entries(analytics.byDir).map(([dir, d]) => (
              <div key={dir} className="perf-dir-row">
                <span style={{ color: dirColors[dir] || 'var(--text-primary)', fontWeight: 700 }}>{dir}</span>
                <span>{d.trades} <span className="perf-wl">({d.wins}W/{d.losses}L)</span></span>
                <span>{d.trades > 0 ? ((d.wins / d.trades) * 100).toFixed(1) : 0}%</span>
                <span className={(d.totalBps || 0) >= 0 ? 'positive' : 'negative'}>{fmtBps(d.totalBps)}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="card perf-card perf-table-half">
          <div className="card-header">By Exit Reason</div>
          <div className="perf-dir-table">
            <div className="perf-dir-header">
              <span>Reason</span><span>Count</span><span>Win%</span><span>Total</span>
            </div>
            {Object.entries(analytics.byExit).map(([reason, d]) => (
              <div key={reason} className="perf-dir-row">
                <span className="perf-reason-name">{reason}</span>
                <span>{d.count}</span>
                <span>{d.count > 0 ? ((d.wins / d.count) * 100).toFixed(1) : 0}%</span>
                <span className={(d.totalBps || 0) >= 0 ? 'positive' : 'negative'}>{fmtBps(d.totalBps)}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Signal Type Breakdown */}
      <div className="card perf-card">
        <div className="card-header">By Signal Type</div>
        <div className="perf-dir-table">
          <div className="perf-dir-header perf-sig-header">
            <span>Signal</span><span>Trades</span><span>Win%</span><span>Avg</span><span>Total</span>
          </div>
          {Object.entries(analytics.bySignal)
            .sort((a, b) => b[1].totalBps - a[1].totalBps)
            .map(([sig, d]) => (
            <div key={sig} className="perf-dir-row perf-sig-row">
              <span style={{ color: SIGNAL_COLORS[sig] || 'var(--text-secondary)', fontWeight: 700, fontSize: '0.7rem' }}>{sig}</span>
              <span>{d.trades} <span className="perf-wl">({d.wins}W/{d.losses}L)</span></span>
              <span>{d.trades > 0 ? ((d.wins / d.trades) * 100).toFixed(1) : 0}%</span>
              <span className={d.avgBps >= 0 ? 'positive' : 'negative'}>{fmtBps(d.avgBps)}</span>
              <span className={(d.totalBps || 0) >= 0 ? 'positive' : 'negative'}>{fmtBps(d.totalBps)}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

export default PerformancePanel
