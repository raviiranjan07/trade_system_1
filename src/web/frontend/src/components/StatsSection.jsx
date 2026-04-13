import React, { useState } from 'react'

function StatsSection({ stats, risk, ml, trades, mlTrades, mlAttnTrades }) {
  const [view, setView] = useState('combined')

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
    walletUsd = ml?.ml_wallet_usd ?? 0
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
  } else if (view === 'v14') {
    walletUsd = risk?.wallet_usd ?? 0
  } else {
    // combined — merge V1.4 + ML + ML V2
    const allTrades = [...(trades || []), ...(mlTrades || []), ...(mlAttnTrades || [])]
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
      walletUsd = (risk?.wallet_usd ?? 0) + (ml?.ml_wallet_usd ?? 0)
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
        <button className={`stats-filter-btn ${view === 'combined' ? 'active' : ''}`} onClick={() => setView('combined')}>Combined</button>
        <button className={`stats-filter-btn ${view === 'v14' ? 'active' : ''}`} onClick={() => setView('v14')}>V1.4</button>
        <button className={`stats-filter-btn ${view === 'ml' ? 'active' : ''}`} onClick={() => setView('ml')}>ML</button>
        <button className={`stats-filter-btn ${view === 'mlv2' ? 'active' : ''}`} onClick={() => setView('mlv2')}>ML V2</button>
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
