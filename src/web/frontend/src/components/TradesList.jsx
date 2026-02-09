import React from 'react'

function TradesList({ trades }) {
  const formatPrice = (price) => {
    if (!price) return '---'
    return `$${Number(price).toLocaleString('en-US', {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    })}`
  }

  const formatBps = (bps) => {
    const sign = bps >= 0 ? '+' : ''
    return `${sign}${bps.toFixed(1)}`
  }

  const formatTime = (isoString) => {
    if (!isoString) return '---'
    const date = new Date(isoString)
    return date.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      hour12: false
    })
  }

  return (
    <div className="trades-section">
      <div className="card">
        <div className="card-header">Recent Trades</div>
        {!trades || trades.length === 0 ? (
          <div className="trades-empty">No trades yet</div>
        ) : (
          <table className="trades-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Side</th>
                <th>Entry</th>
                <th>Exit</th>
                <th>P&L (bps)</th>
                <th>Bar</th>
                <th>Reason</th>
                <th>Time</th>
              </tr>
            </thead>
            <tbody>
              {trades.map((trade, index) => (
                <tr key={trade.trade_id || index}>
                  <td>{trades.length - index}</td>
                  <td>
                    <span className={trade.direction?.toLowerCase() === 'long' ? 'positive' : 'negative'}>
                      {trade.direction}
                    </span>
                    {trade.is_reentry && <span className="re-tag">RE</span>}
                  </td>
                  <td>{formatPrice(trade.entry_price)}</td>
                  <td>{formatPrice(trade.exit_price)}</td>
                  <td className={trade.net_profit_bps >= 0 ? 'positive' : 'negative'}>
                    {formatBps(trade.net_profit_bps || 0)}
                  </td>
                  <td>{trade.exit_bar || '---'}</td>
                  <td>{trade.exit_reason || '---'}</td>
                  <td>{formatTime(trade.exit_time)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}

export default TradesList
