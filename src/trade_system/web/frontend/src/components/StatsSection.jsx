import React from 'react'

function StatsSection({ stats }) {
  if (!stats) return null

  const formatPnl = (pnl) => {
    const sign = pnl >= 0 ? '+' : ''
    return `${sign}$${pnl.toFixed(2)}`
  }

  const pnlClass = stats.total_pnl >= 0 ? 'positive' : 'negative'

  return (
    <div className="stats-section">
      <div className="card">
        <div className="card-header">Session Statistics</div>
        <div className="stats-grid">
          <div className="stat-item">
            <div className="label">Capital</div>
            <div className="value">${stats.capital?.toFixed(2)}</div>
          </div>
          <div className="stat-item">
            <div className="label">Total P&L</div>
            <div className={`value ${pnlClass}`}>
              {formatPnl(stats.total_pnl || 0)}
              <span style={{ fontSize: '0.8rem', marginLeft: '4px' }}>
                ({(stats.total_pnl_pct * 100 || 0).toFixed(2)}%)
              </span>
            </div>
          </div>
          <div className="stat-item">
            <div className="label">Trades</div>
            <div className="value">{stats.total_trades || 0}</div>
          </div>
          <div className="stat-item">
            <div className="label">Win Rate</div>
            <div className="value">{((stats.win_rate || 0) * 100).toFixed(0)}%</div>
          </div>
          <div className="stat-item">
            <div className="label">Commission</div>
            <div className="value">${(stats.total_commission || 0).toFixed(2)}</div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default StatsSection
