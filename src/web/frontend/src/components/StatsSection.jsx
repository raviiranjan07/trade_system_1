import React from 'react'

function StatsSection({ stats }) {
  if (!stats) return null

  const bpsClass = stats.total_bps >= 0 ? 'positive' : 'negative'
  const bpsSign = stats.total_bps >= 0 ? '+' : ''

  return (
    <div className="stats-section">
      <div className="card">
        <div className="card-header">Session Statistics</div>
        <div className="stats-grid">
          <div className="stat-item">
            <div className="label">Total P&L</div>
            <div className={`value ${bpsClass}`}>
              {bpsSign}{stats.total_bps?.toFixed(1)} bps
            </div>
          </div>
          <div className="stat-item">
            <div className="label">Trades</div>
            <div className="value">
              {stats.total_trades || 0}
              <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginLeft: '4px' }}>
                ({stats.wins || 0}W / {stats.losses || 0}L)
              </span>
            </div>
          </div>
          <div className="stat-item">
            <div className="label">Win Rate</div>
            <div className="value">{((stats.win_rate || 0) * 100).toFixed(1)}%</div>
          </div>
          <div className="stat-item">
            <div className="label">Profit Factor</div>
            <div className="value">{stats.profit_factor?.toFixed(2) || '---'}</div>
          </div>
          <div className="stat-item">
            <div className="label">Avg Trade</div>
            <div className={`value ${(stats.avg_bps || 0) >= 0 ? 'positive' : 'negative'}`}>
              {(stats.avg_bps || 0) >= 0 ? '+' : ''}{stats.avg_bps?.toFixed(1) || '0'} bps
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default StatsSection
