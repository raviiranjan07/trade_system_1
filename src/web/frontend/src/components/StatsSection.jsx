import React from 'react'

const FILTER_TO_VIEW = { 'Combined': 'combined', 'V1.4': 'v14', 'ML': 'ml', 'ML V2': 'mlv2', 'ML V3': 'mlv3' }

function StatsSection({ stats, risk, ml, trades, mlTrades, mlAttnTrades, mlV3Trades, activeFilter, onFilterChange }) {
  const view = FILTER_TO_VIEW[activeFilter] || 'combined'

  // Compute stats based on view
  let displayStats = stats || {}
  let walletUsd = risk?.wallet_usd ?? 0

  if (view === 'ml' && mlTrades && mlTrades.length > 0) {
    const wins = mlTrades.filter(t => (t.net_profit_bps || 0) > 0).length
    const losses = mlTrades.length - wins
    const totalBps = mlTrades.reduce((s, t) => s + (t.net_profit_bps || 0), 0)
    const grossWin = mlTrades.filter(t => (t.net_profit_bps || 0) > 0).reduce((s, t) => s + (t.net_profit_bps || 0), 0)
    const grossLoss = Math.abs(mlTrades.filter(t => (t.net_profit_bps || 0) <= 0).reduce((s, t) => s + (t.net_profit_bps || 0), 0))
    displayStats = {
      total_trades: mlTrades.length,
      wins,
      losses,
      win_rate: mlTrades.length > 0 ? wins / mlTrades.length : 0,
      total_bps: totalBps,
      avg_bps: mlTrades.length > 0 ? totalBps / mlTrades.length : 0,
      profit_factor: grossLoss > 0 ? grossWin / grossLoss : 0,
    }
    walletUsd = ml?.wallet_usd ?? 0
  } else if (view === 'mlv2' && mlAttnTrades && mlAttnTrades.length > 0) {
    const wins = mlAttnTrades.filter(t => (t.net_profit_bps || 0) > 0).length
    const losses = mlAttnTrades.length - wins
    const totalBps = mlAttnTrades.reduce((s, t) => s + (t.net_profit_bps || 0), 0)
    const grossWin = mlAttnTrades.filter(t => (t.net_profit_bps || 0) > 0).reduce((s, t) => s + (t.net_profit_bps || 0), 0)
    const grossLoss = Math.abs(mlAttnTrades.filter(t => (t.net_profit_bps || 0) <= 0).reduce((s, t) => s + (t.net_profit_bps || 0), 0))
    displayStats = {
      total_trades: mlAttnTrades.length,
      wins,
      losses,
      win_rate: mlAttnTrades.length > 0 ? wins / mlAttnTrades.length : 0,
      total_bps: totalBps,
      avg_bps: mlAttnTrades.length > 0 ? totalBps / mlAttnTrades.length : 0,
      profit_factor: grossLoss > 0 ? grossWin / grossLoss : 0,
    }
    walletUsd = 0  // will be updated when ml_attn wallet is available
  } else if (view === 'mlv3' && mlV3Trades && mlV3Trades.length > 0) {
    const wins = mlV3Trades.filter(t => (t.net_profit_bps || 0) > 0).length
    const losses = mlV3Trades.length - wins
    const totalBps = mlV3Trades.reduce((s, t) => s + (t.net_profit_bps || 0), 0)
    const grossWin = mlV3Trades.filter(t => (t.net_profit_bps || 0) > 0).reduce((s, t) => s + (t.net_profit_bps || 0), 0)
    const grossLoss = Math.abs(mlV3Trades.filter(t => (t.net_profit_bps || 0) <= 0).reduce((s, t) => s + (t.net_profit_bps || 0), 0))
    displayStats = {
      total_trades: mlV3Trades.length, wins, losses,
      win_rate: mlV3Trades.length > 0 ? wins / mlV3Trades.length : 0,
      total_bps: totalBps,
      avg_bps: mlV3Trades.length > 0 ? totalBps / mlV3Trades.length : 0,
      profit_factor: grossLoss > 0 ? grossWin / grossLoss : 0,
    }
    walletUsd = 0
  } else if (view === 'v14') {
    walletUsd = risk?.wallet_usd ?? 0
  } else {
    // combined — merge V1.4 + ML + ML V2 + ML V3
    const allTrades = [...(trades || []), ...(mlTrades || []), ...(mlAttnTrades || []), ...(mlV3Trades || [])]
    if (allTrades.length > 0 && stats) {
      const totalWins = allTrades.filter(t => (t.net_profit_bps || 0) > 0).length
      const totalLosses = allTrades.length - totalWins
      const totalBps = allTrades.reduce((s, t) => s + (t.net_profit_bps || 0), 0)
      const grossWin = allTrades.filter(t => (t.net_profit_bps || 0) > 0).reduce((s, t) => s + (t.net_profit_bps || 0), 0)
      const grossLoss = Math.abs(allTrades.filter(t => (t.net_profit_bps || 0) <= 0).reduce((s, t) => s + (t.net_profit_bps || 0), 0))

      displayStats = {
        total_trades: allTrades.length,
        wins: totalWins,
        losses: totalLosses,
        win_rate: allTrades.length > 0 ? totalWins / allTrades.length : 0,
        total_bps: totalBps,
        avg_bps: allTrades.length > 0 ? totalBps / allTrades.length : 0,
        profit_factor: grossLoss > 0 ? Math.round((grossWin / grossLoss) * 100) / 100 : 0,
      }
      walletUsd = (risk?.wallet_usd ?? 0) + (ml?.wallet_usd ?? 0)
    }
  }

  const totalBps = displayStats.total_bps || 0
  const bpsClass = totalBps >= 0 ? 'positive' : 'negative'
  const bpsSign = totalBps >= 0 ? '+' : ''
  const winRate = ((displayStats.win_rate || 0) * 100).toFixed(1)
  const avgBps = displayStats.avg_bps || 0

  return (
    <div>
      <div className="stats-filter-bar">
        {['Combined', 'V1.4', 'ML', 'ML V2', 'ML V3'].map(f => {
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
