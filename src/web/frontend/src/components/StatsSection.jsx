import React from 'react'

const FILTER_TO_VIEW = { 'Combined': 'combined', 'ML': 'ml', 'ML V2': 'mlv2', 'ML V3': 'mlv3' }

function aggregate(trades) {
  const t = trades || []
  const wins = t.filter(x => (x.net_profit_bps || 0) > 0).length
  const losses = t.length - wins
  const totalBps = t.reduce((s, x) => s + (x.net_profit_bps || 0), 0)
  const grossWin = t.filter(x => (x.net_profit_bps || 0) > 0).reduce((s, x) => s + (x.net_profit_bps || 0), 0)
  const grossLoss = Math.abs(t.filter(x => (x.net_profit_bps || 0) <= 0).reduce((s, x) => s + (x.net_profit_bps || 0), 0))
  return {
    total_trades: t.length,
    wins,
    losses,
    win_rate: t.length > 0 ? wins / t.length : 0,
    total_bps: totalBps,
    avg_bps: t.length > 0 ? totalBps / t.length : 0,
    profit_factor: grossLoss > 0 ? Math.round((grossWin / grossLoss) * 100) / 100 : 0,
  }
}

function StatsSection({ ml, mlAttn, mlV3, mlTrades, mlAttnTrades, mlV3Trades, activeFilter, onFilterChange }) {
  const view = FILTER_TO_VIEW[activeFilter] || 'combined'

  let displayStats
  let walletUsd
  if (view === 'ml') {
    displayStats = aggregate(mlTrades)
    walletUsd = ml?.wallet_usd ?? 0
  } else if (view === 'mlv2') {
    displayStats = aggregate(mlAttnTrades)
    walletUsd = mlAttn?.wallet_usd ?? 0
  } else if (view === 'mlv3') {
    displayStats = aggregate(mlV3Trades)
    walletUsd = mlV3?.wallet_usd ?? 0
  } else {
    displayStats = aggregate([...(mlTrades || []), ...(mlAttnTrades || []), ...(mlV3Trades || [])])
    walletUsd = (ml?.wallet_usd ?? 0) + (mlAttn?.wallet_usd ?? 0) + (mlV3?.wallet_usd ?? 0)
  }

  const totalBps = displayStats.total_bps || 0
  const bpsClass = totalBps >= 0 ? 'positive' : 'negative'
  const bpsSign = totalBps >= 0 ? '+' : ''
  const winRate = ((displayStats.win_rate || 0) * 100).toFixed(1)
  const avgBps = displayStats.avg_bps || 0

  return (
    <div>
      <div className="stats-filter-bar">
        {['Combined', 'ML', 'ML V2', 'ML V3'].map(f => {
          const viewKey = FILTER_TO_VIEW[f]
          return (
            <button key={f} className={`stats-filter-btn ${view === viewKey ? 'active' : ''}`} onClick={() => onFilterChange?.(f)}>{f}</button>
          )
        })}
      </div>
      <div className="stats-hero">
        <div className="stats-hero-card">
          <div className={`stats-hero-card-value ${bpsClass}`}>
            {bpsSign}{totalBps.toFixed(1)}
          </div>
          <div className="stats-hero-card-label">Total P&L (bps)</div>
          {walletUsd > 0 && (
            <div className="stats-hero-card-sub">${walletUsd.toFixed(2)}</div>
          )}
        </div>

        <div className="stats-hero-card">
          <div className="stats-hero-card-value">{displayStats.total_trades || 0}</div>
          <div className="stats-hero-card-label">Trades</div>
          <div className="stats-hero-card-sub">
            <span className="positive">{displayStats.wins || 0}W</span>
            {' / '}
            <span className="negative">{displayStats.losses || 0}L</span>
          </div>
        </div>

        <div className="stats-hero-card">
          <div className="stats-hero-card-value">{winRate}%</div>
          <div className="stats-hero-card-label">Win Rate</div>
        </div>

        <div className="stats-hero-card">
          <div className="stats-hero-card-value">{displayStats.profit_factor?.toFixed(2) || '---'}</div>
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
