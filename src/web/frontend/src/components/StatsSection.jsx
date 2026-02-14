import React from 'react'

function StatsSection({ stats, risk }) {
  if (!stats) return null

  const totalBps = stats.total_bps || 0
  const bpsClass = totalBps >= 0 ? 'positive' : 'negative'
  const bpsSign = totalBps >= 0 ? '+' : ''
  const winRate = ((stats.win_rate || 0) * 100).toFixed(1)
  const avgBps = stats.avg_bps || 0
  const wallet = risk?.wallet_usd ?? 0

  return (
    <div className="stats-hero">
      {/* Hero: Total P&L */}
      <div className="stats-hero-main">
        <div className="stats-hero-pnl-label">Total P&L</div>
        <div className={`stats-hero-pnl ${bpsClass}`}>
          {bpsSign}{totalBps.toFixed(1)} <span className="stats-hero-unit">bps</span>
        </div>
        {wallet > 0 && (
          <div className="stats-hero-wallet">Wallet: ${wallet.toFixed(2)}</div>
        )}
      </div>

      {/* Stat Cards */}
      <div className="stats-hero-grid">
        <div className="stats-hero-card">
          <div className="stats-hero-card-value">{stats.total_trades || 0}</div>
          <div className="stats-hero-card-label">Trades</div>
          <div className="stats-hero-card-sub">
            <span className="positive">{stats.wins || 0}W</span>
            {' / '}
            <span className="negative">{stats.losses || 0}L</span>
          </div>
        </div>

        <div className="stats-hero-card">
          <div className="stats-hero-card-value">{winRate}%</div>
          <div className="stats-hero-card-label">Win Rate</div>
        </div>

        <div className="stats-hero-card">
          <div className="stats-hero-card-value">{stats.profit_factor?.toFixed(2) || '---'}</div>
          <div className="stats-hero-card-label">Profit Factor</div>
        </div>

        <div className="stats-hero-card">
          <div className={`stats-hero-card-value ${avgBps >= 0 ? 'positive' : 'negative'}`}>
            {avgBps >= 0 ? '+' : ''}{avgBps.toFixed(1)}
          </div>
          <div className="stats-hero-card-label">Avg Trade (bps)</div>
        </div>
      </div>
    </div>
  )
}

export default StatsSection
