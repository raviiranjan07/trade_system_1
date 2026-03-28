import React from 'react'

function RiskPanel({ risk, ml }) {
  if (!risk) return null

  const v14 = {
    wallet: risk.wallet_usd ?? 0,
    peak: risk.peak_usd ?? 0,
    dd: risk.drawdown_pct ?? 0,
    health: risk.health_multiplier ?? 1,
    streak: risk.consecutive_losses ?? 0,
    winrate: risk.recent_winrate ?? 1,
    skips: risk.total_skips ?? 0,
    lastPnl: risk.last_pnl_usd ?? 0,
    lastQty: risk.last_qty ?? 0,
  }

  const mlData = {
    wallet: ml?.ml_wallet_usd ?? 0,
    peak: ml?.ml_peak_usd ?? 0,
    dd: ml?.ml_drawdown_pct ?? 0,
    health: ml?.ml_health_multiplier ?? 1,
    streak: ml?.ml_consecutive_losses ?? 0,
    winrate: ml?.ml_recent_winrate ?? 1,
    skips: ml?.ml_total_skips ?? 0,
    lastPnl: ml?.ml_last_pnl_usd ?? 0,
    lastQty: ml?.ml_last_qty ?? 0,
    loaded: ml?.ml_model_loaded ?? false,
  }

  return (
    <div className="card risk-panel">
      <div className="card-header">Risk Manager</div>

      <div className="risk-two-col">
        <RiskColumn label="V1.4" data={v14} />
        <RiskColumn label="ML" data={mlData} dimmed={!mlData.loaded} />
      </div>
    </div>
  )
}

function RiskColumn({ label, data, dimmed = false }) {
  const ddClass = data.dd > 0.15 ? 'negative' : data.dd > 0.05 ? 'warning' : ''
  const healthClass = data.health < 0.5 ? 'negative' : data.health < 0.8 ? 'warning' : 'positive'
  const pnlClass = data.lastPnl >= 0 ? 'positive' : 'negative'

  return (
    <div className={`risk-column ${dimmed ? 'dimmed' : ''}`}>
      <div className="risk-col-header">{label}</div>

      <div className="risk-wallet">
        <span className="risk-wallet-value">${data.wallet.toFixed(2)}</span>
        {data.lastPnl !== 0 && (
          <span className={`risk-last-pnl ${pnlClass}`}>
            {data.lastPnl >= 0 ? '+' : ''}{data.lastPnl.toFixed(4)}
          </span>
        )}
      </div>

      <div className="risk-row">
        <span className="label">DD</span>
        <div className="risk-dd-bar-container">
          <div
            className={`risk-dd-bar-fill ${data.dd > 0.15 ? 'danger' : data.dd > 0.05 ? 'warn' : ''}`}
            style={{ width: `${Math.min(data.dd * 100, 100)}%` }}
          />
        </div>
        <span className={`risk-dd-value ${ddClass}`}>{(data.dd * 100).toFixed(1)}%</span>
      </div>

      <div className="risk-grid">
        <div className="risk-item">
          <div className="label">Health</div>
          <div className={`value ${healthClass}`}>{data.health.toFixed(2)}</div>
        </div>
        <div className="risk-item">
          <div className="label">Streak</div>
          <div className={`value ${data.streak >= 5 ? 'negative' : ''}`}>{data.streak}L</div>
        </div>
        <div className="risk-item">
          <div className="label">WR</div>
          <div className="value">{(data.winrate * 100).toFixed(0)}%</div>
        </div>
        <div className="risk-item">
          <div className="label">Skips</div>
          <div className="value">{data.skips}</div>
        </div>
      </div>

      <div className="risk-footer">
        <span className="label">Peak: ${data.peak.toFixed(2)}</span>
      </div>
    </div>
  )
}

export default RiskPanel
