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

  const formatPnl = (pnl) => {
    const sign = pnl >= 0 ? '+' : ''
    return `${sign}$${pnl.toFixed(2)}`
  }

  const formatPnlPct = (pct) => {
    const sign = pct >= 0 ? '+' : ''
    return `${sign}${(pct * 100).toFixed(2)}%`
  }

  const pnlClass = position.unrealized_pnl >= 0 ? 'positive' : 'negative'

  return (
    <div className="card position-card">
      <div className="position-header">
        <div>
          <div className="card-header">Current Position</div>
          <span className={`side-badge ${position.side?.toLowerCase()}`}>
            {position.side}
          </span>
        </div>
        <div className="pnl-display">
          <div className={`amount ${pnlClass}`}>
            {formatPnl(position.unrealized_pnl)}
          </div>
          <div className={`percent ${pnlClass}`}>
            {formatPnlPct(position.unrealized_pnl_pct)}
          </div>
        </div>
      </div>

      <div className="position-grid">
        <div className="position-item">
          <div className="label">Size</div>
          <div className="value">{position.size_btc?.toFixed(6)} BTC</div>
        </div>
        <div className="position-item">
          <div className="label">Entry Price</div>
          <div className="value">{formatPrice(position.entry_price)}</div>
        </div>
        <div className="position-item">
          <div className="label">Take Profit</div>
          <div className="value positive">{formatPrice(position.tp_price)}</div>
        </div>
        <div className="position-item">
          <div className="label">Stop Loss</div>
          <div className="value negative">{formatPrice(position.sl_price)}</div>
        </div>
      </div>
    </div>
  )
}

export default PositionCard
