import React, { useState, useMemo } from 'react'

const SIGNAL_COLORS = {
  V12_LONG: '#00d97e',
  V12_SHORT: '#e63757',
  BEAR_LONG: '#3b82f6',
  BULL_SHORT: '#f59e0b',
  ML_LONG: '#8b5cf6',
  ML_SHORT: '#ec4899',
}

const REASON_STYLES = {
  TRAILING_STOP: { bg: 'rgba(0, 217, 126, 0.12)', color: '#00d97e', label: 'TS' },
  TIGHT_TS: { bg: 'rgba(59, 130, 246, 0.12)', color: '#3b82f6', label: 'TIGHT' },
  TIME_EXIT: { bg: 'rgba(245, 158, 11, 0.12)', color: '#f59e0b', label: 'TIME' },
  BE_LOCK: { bg: 'rgba(139, 92, 246, 0.12)', color: '#8b5cf6', label: 'BE' },
  EARLY_CUT: { bg: 'rgba(230, 55, 87, 0.12)', color: '#e63757', label: 'EARLY' },
}

function TradesList({ trades, mlTrades }) {
  const [filter, setFilter] = useState('Combined')
  const [sortCol, setSortCol] = useState(null)
  const [sortDir, setSortDir] = useState('desc')

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

  const isV14 = (sig) => sig && (sig.startsWith('V12_') || sig.startsWith('BEAR_') || sig.startsWith('BULL_'))
  const isML = (sig) => sig && sig.startsWith('ML_')

  // Merge and sort by exit_time descending, then apply filter and sort
  const displayTrades = useMemo(() => {
    const v14 = Array.isArray(trades) ? trades : []
    const ml = Array.isArray(mlTrades) ? mlTrades : []
    let merged = [...v14, ...ml]

    // Sort by exit_time descending first (default order)
    merged.sort((a, b) => {
      const ta = a.exit_time ? new Date(a.exit_time).getTime() : 0
      const tb = b.exit_time ? new Date(b.exit_time).getTime() : 0
      return tb - ta
    })

    // Apply filter
    if (filter === 'V1.4') {
      merged = merged.filter(t => isV14(t.signal_type))
    } else if (filter === 'ML') {
      merged = merged.filter(t => isML(t.signal_type))
    }
    // 'Combined' shows all

    // Apply sort
    if (sortCol) {
      const mult = sortDir === 'asc' ? 1 : -1
      merged.sort((a, b) => {
        let va, vb
        if (sortCol === 'pnl') {
          va = a.net_profit_bps || 0
          vb = b.net_profit_bps || 0
        } else if (sortCol === 'datetime') {
          va = a.exit_time ? new Date(a.exit_time).getTime() : 0
          vb = b.exit_time ? new Date(b.exit_time).getTime() : 0
        } else if (sortCol === 'bar') {
          va = a.exit_bar || 0
          vb = b.exit_bar || 0
        }
        return (va - vb) * mult
      })
    }

    return merged
  }, [trades, mlTrades, filter, sortCol, sortDir])

  const handleSort = (col) => {
    if (sortCol === col) {
      setSortDir(prev => prev === 'asc' ? 'desc' : 'asc')
    } else {
      setSortCol(col)
      setSortDir('desc')
    }
  }

  const sortIndicator = (col) => {
    if (sortCol !== col) return ''
    return sortDir === 'asc' ? ' \u25B2' : ' \u25BC'
  }

  return (
    <div className="trades-section">
      <div className="card">
        <div className="card-header">Recent Trades</div>
        <div className="trades-filter-bar">
          {['Combined', 'V1.4', 'ML'].map(f => (
            <button
              key={f}
              className={`trades-filter-btn${filter === f ? ' active' : ''}`}
              onClick={() => setFilter(f)}
            >
              {f}
            </button>
          ))}
        </div>
        {displayTrades.length === 0 ? (
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
                <th className="sortable-header" onClick={() => handleSort('pnl')}>
                  P&L{sortIndicator('pnl')}
                </th>
                <th>Qty</th>
                <th>Liq Price</th>
                <th className="sortable-header" onClick={() => handleSort('bar')}>
                  Bar{sortIndicator('bar')}
                </th>
                <th>Reason</th>
                <th className="sortable-header" onClick={() => handleSort('datetime')}>
                  Date{sortIndicator('datetime')}
                </th>
                <th>Time</th>
              </tr>
            </thead>
            <tbody>
              {displayTrades.map((trade, index) => {
                const { date, time } = formatDateTime(trade.exit_time)
                const reasonStyle = getReasonStyle(trade.exit_reason)
                const sigColor = SIGNAL_COLORS[trade.signal_type] || 'var(--text-secondary)'

                return (
                  <tr key={trade.trade_id || index}>
                    <td className="trade-num">{displayTrades.length - index}</td>
                    <td>
                      {trade.signal_type ? (
                        <span className="signal-tag" style={{ color: sigColor, borderColor: sigColor, background: `${sigColor}18` }}>
                          {trade.signal_type}
                        </span>
                      ) : '---'}
                    </td>
                    <td>
                      <span className={`side-tag ${(trade.direction || '').toLowerCase()}`}>
                        {trade.direction}
                      </span>
                      {trade.is_reentry && <span className="re-tag">RE</span>}
                    </td>
                    <td className="trade-price">{formatPrice(trade.entry_price)}</td>
                    <td className="trade-price">{formatPrice(trade.exit_price)}</td>
                    <td className={`trade-pnl ${(trade.net_profit_bps || 0) >= 0 ? 'positive' : 'negative'}`}>
                      {formatBps(trade.net_profit_bps || 0)}
                    </td>
                    <td className="trade-qty">{trade.qty ? Number(trade.qty).toFixed(4) : '---'}</td>
                    <td className="trade-price">{trade.liq_price ? formatPrice(trade.liq_price) : '---'}</td>
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
