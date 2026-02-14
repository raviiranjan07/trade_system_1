import React from 'react'

function PositionCard({ position }) {
  if (!position || !position.has_position) {
    return (
      <div className="card position-card no-position">
        <div className="card-header">Current Position</div>
        <div className="position-empty">
          <div className="position-empty-icon">--</div>
          No open position
        </div>
      </div>
    )
  }

  const formatPrice = (price) => {
    if (!price) return '---'
    return `$${Number(price).toLocaleString('en-US', {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    })}`
  }

  const pnlBps = position.current_pnl_bps ?? 0
  const mfeBps = position.mfe_bps ?? 0
  const maeBps = position.mae_bps ?? 0
  const peakBps = position.highest_profit_bps ?? 0
  const pnlClass = pnlBps >= 0 ? 'positive' : 'negative'
  const pnlSign = pnlBps >= 0 ? '+' : ''

  // Trailing stop progress: how close are we to being stopped out?
  const tsProgress = position.trailing_stop_bps > 0
    ? Math.min(100, (peakBps / position.trailing_stop_bps) * 100)
    : 0

  // Time progress: bars_held / max_bars
  const timeProgress = position.max_bars > 0
    ? Math.min(100, (position.bars_held / position.max_bars) * 100)
    : 0

  return (
    <div className="card position-card">
      <div className="position-header">
        <div>
          <div className="card-header">Current Position</div>
          <span className={`side-badge ${position.side?.toLowerCase()}`}>
            {position.side}
          </span>
          {position.is_reentry && (
            <span className="reentry-badge">RE-ENTRY</span>
          )}
        </div>
        <div className="pnl-display">
          <div className={`amount ${pnlClass}`}>
            {pnlSign}{pnlBps.toFixed(1)} bps
          </div>
          <div className={`percent ${pnlClass}`}>
            MFE: {mfeBps.toFixed(1)} | MAE: {maeBps.toFixed(1)}
          </div>
        </div>
      </div>

      <div className="position-grid">
        <div className="position-item">
          <div className="label">Entry Price</div>
          <div className="value">{formatPrice(position.entry_price)}</div>
        </div>
        <div className="position-item">
          <div className="label">Bars Held</div>
          <div className="value">
            {position.bars_held} / {position.max_bars}
            <div className="progress-bar">
              <div className="progress-fill time-fill" style={{ width: `${timeProgress}%` }}></div>
            </div>
          </div>
        </div>
        <div className="position-item">
          <div className="label">Trailing Stop</div>
          <div className="value">
            {position.trailing_stop_bps} bps
            <div className="progress-bar">
              <div className="progress-fill ts-fill" style={{ width: `${tsProgress}%` }}></div>
            </div>
          </div>
        </div>
        <div className="position-item">
          <div className="label">Peak Profit</div>
          <div className="value positive">+{peakBps.toFixed(1)} bps</div>
        </div>
      </div>
    </div>
  )
}

export default PositionCard
