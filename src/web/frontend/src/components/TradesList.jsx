import React from 'react'

const SIGNAL_COLORS = {
  V12_LONG: '#00d97e',
  V12_SHORT: '#e63757',
  BEAR_LONG: '#3b82f6',
  BULL_SHORT: '#f59e0b',
}

const REASON_STYLES = {
  TRAILING_STOP: { bg: 'rgba(0, 217, 126, 0.12)', color: '#00d97e', label: 'TS' },
  TIME_EXIT: { bg: 'rgba(245, 158, 11, 0.12)', color: '#f59e0b', label: 'TIME' },
}

function TradesList({ trades }) {
  const formatPrice = (price) => {
    if (!price) return '---'
    return `$${Number(price).toLocaleString('en-US', {
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    })}`
  }

  const formatBps = (bps) => {
    const sign = bps >= 0 ? '+' : ''
    return `${sign}${bps.toFixed(1)}`
  }

  const formatDateTime = (isoString) => {
    if (!isoString) return { date: '---', time: '---' }
    const d = new Date(isoString)
    const date = d.toLocaleDateString('en-GB', { day: '2-digit', month: 'short' })
    const time = d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false })
    return { date, time }
  }

  const getReasonStyle = (reason) => {
    return REASON_STYLES[reason] || { bg: 'rgba(136, 146, 160, 0.12)', color: '#8892a0', label: reason || '---' }
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
                <th>Signal</th>
                <th>Side</th>
                <th>Entry</th>
                <th>Exit</th>
                <th>P&L</th>
                <th>Bar</th>
                <th>Exit</th>
                <th>Date</th>
                <th>Time</th>
              </tr>
            </thead>
            <tbody>
              {trades.map((trade, index) => {
                const { date, time } = formatDateTime(trade.exit_time)
                const reasonStyle = getReasonStyle(trade.exit_reason)
                const sigColor = SIGNAL_COLORS[trade.signal_type] || 'var(--text-secondary)'

                return (
                  <tr key={trade.trade_id || index}>
                    <td className="trade-num">{trades.length - index}</td>
                    <td>
                      {trade.signal_type ? (
                        <span className="signal-tag" style={{ color: sigColor, borderColor: sigColor }}>
                          {trade.signal_type}
                        </span>
                      ) : '---'}
                    </td>
                    <td>
                      <span className={`side-tag ${trade.direction?.toLowerCase()}`}>
                        {trade.direction}
                      </span>
                      {trade.is_reentry && <span className="re-tag">RE</span>}
                    </td>
                    <td className="trade-price">{formatPrice(trade.entry_price)}</td>
                    <td className="trade-price">{formatPrice(trade.exit_price)}</td>
                    <td className={`trade-pnl ${(trade.net_profit_bps || 0) >= 0 ? 'positive' : 'negative'}`}>
                      {formatBps(trade.net_profit_bps || 0)}
                    </td>
                    <td className="trade-bar">{trade.exit_bar || '---'}</td>
                    <td>
                      <span className="reason-tag" style={{ background: reasonStyle.bg, color: reasonStyle.color }}>
                        {reasonStyle.label}
                      </span>
                    </td>
                    <td className="trade-date">{date}</td>
                    <td className="trade-time">{time}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}

export default TradesList
