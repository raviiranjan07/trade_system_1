import React from 'react'

function RiskPanel({ risk }) {
  if (!risk) return null

  const wallet = risk.wallet_usd ?? 0
  const peak = risk.peak_usd ?? 0
  const dd = risk.drawdown_pct ?? 0
  const health = risk.health_multiplier ?? 1
  const streak = risk.consecutive_losses ?? 0
  const winrate = risk.recent_winrate ?? 1
  const skips = risk.total_skips ?? 0
  const lastPnl = risk.last_pnl_usd ?? 0

  const ddClass = dd > 0.15 ? 'negative' : dd > 0.05 ? 'warning' : ''
  const healthClass = health < 0.5 ? 'negative' : health < 0.8 ? 'warning' : 'positive'
  const pnlClass = lastPnl >= 0 ? 'positive' : 'negative'

  return (
    <div className="card risk-panel">
      <div className="card-header">Risk Manager</div>

      {/* Wallet — big number */}
      <div className="risk-wallet">
        <span className="risk-wallet-label">Wallet</span>
        <span className="risk-wallet-value">${wallet.toFixed(2)}</span>
        {lastPnl !== 0 && (
          <span className={`risk-last-pnl ${pnlClass}`}>
            {lastPnl >= 0 ? '+' : ''}{lastPnl.toFixed(4)}
          </span>
        )}
      </div>

      {/* Drawdown bar */}
      <div className="risk-row">
        <span className="label">Drawdown</span>
        <div className="risk-dd-bar-container">
          <div
            className={`risk-dd-bar-fill ${dd > 0.15 ? 'danger' : dd > 0.05 ? 'warn' : ''}`}
            style={{ width: `${Math.min(dd * 100, 100)}%` }}
          />
        </div>
        <span className={`risk-dd-value ${ddClass}`}>{(dd * 100).toFixed(1)}%</span>
      </div>

      {/* Grid: health, streak, winrate, skips */}
      <div className="risk-grid">
        <div className="risk-item">
          <div className="label">Health</div>
          <div className={`value ${healthClass}`}>{health.toFixed(2)}</div>
        </div>
        <div className="risk-item">
          <div className="label">Streak</div>
          <div className={`value ${streak >= 5 ? 'negative' : ''}`}>{streak}L</div>
        </div>
        <div className="risk-item">
          <div className="label">Win Rate</div>
          <div className="value">{(winrate * 100).toFixed(0)}%</div>
        </div>
        <div className="risk-item">
          <div className="label">Skips</div>
          <div className="value">{skips}</div>
        </div>
      </div>

      {/* Peak */}
      <div className="risk-footer">
        <span className="label">Peak: ${peak.toFixed(2)}</span>
      </div>
    </div>
  )
}

export default RiskPanel
