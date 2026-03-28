import React from 'react'

const DEFAULT_CAPITAL = 5

const SIGNAL_COLORS = {
  V12_LONG: '#00d97e',
  V12_SHORT: '#e63757',
  BEAR_LONG: '#3b82f6',
  BULL_SHORT: '#f59e0b',
  ML_LONG: '#8b5cf6',
  ML_SHORT: '#ec4899',
}

function timeAgo(isoString) {
  if (!isoString) return '---'
  const now = new Date()
  const then = new Date(isoString)
  const diffMs = now - then
  if (diffMs < 0) return 'just now'
  const mins = Math.floor(diffMs / 60000)
  if (mins < 1) return 'just now'
  if (mins < 60) return `${mins}m ago`
  const hours = Math.floor(mins / 60)
  if (hours < 24) return `${hours}h ago`
  const days = Math.floor(hours / 24)
  return `${days}d ago`
}

function isToday(isoString) {
  if (!isoString) return false
  const d = new Date(isoString)
  const now = new Date()
  return d.getFullYear() === now.getFullYear() &&
    d.getMonth() === now.getMonth() &&
    d.getDate() === now.getDate()
}

function formatPrice(price) {
  if (!price) return '---'
  return `$${Number(price).toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 })}`
}

function formatBps(bps) {
  if (bps == null) return '---'
  const sign = bps >= 0 ? '+' : ''
  return `${sign}${Number(bps).toFixed(1)}`
}

function formatDateTime(isoString) {
  if (!isoString) return '---'
  const d = new Date(isoString)
  const date = d.toLocaleDateString('en-GB', { day: '2-digit', month: 'short' })
  const time = d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false })
  return `${date} ${time}`
}

function OverviewPage({ stats, risk, ml, trades, mlTrades, status, position }) {
  const v14Wallet = risk?.wallet_usd ?? 0
  const mlWallet = ml?.ml_wallet_usd ?? 0
  const totalBalance = v14Wallet + mlWallet

  // Today's P&L from V1.4 trades
  const todayV14Bps = (trades || [])
    .filter(t => isToday(t.exit_time))
    .reduce((sum, t) => sum + (t.net_profit_bps || 0), 0)

  // Today's P&L from ML trades
  const todayMlBps = (mlTrades || [])
    .filter(t => isToday(t.exit_time))
    .reduce((sum, t) => sum + (t.net_profit_bps || 0), 0)

  const todayTotalBps = todayV14Bps + todayMlBps

  // Active positions count
  const activePositions = position?.has_position ? 1 : 0

  // V1.4 stats
  const v14TotalBps = stats?.total_bps ?? 0
  const v14Trades = stats?.total_trades ?? 0
  const v14WinRate = stats?.win_rate ? (stats.win_rate * 100).toFixed(0) : '0'
  const v14Drawdown = risk?.drawdown_pct ?? 0
  const v14Growth = v14Wallet > 0 ? ((v14Wallet - DEFAULT_CAPITAL) / DEFAULT_CAPITAL * 100).toFixed(0) : '0'
  const v14LastTrade = trades && trades.length > 0 ? trades[0] : null

  // ML stats
  const mlTotalBps = ml?.ml_total_bps ?? 0
  const mlTotalTrades = ml?.ml_total_trades ?? 0
  const mlWinRate = ml?.ml_win_rate ? (ml.ml_win_rate * 100).toFixed(0) : '0'
  const mlDrawdown = ml?.ml_drawdown_pct ?? 0
  const mlGrowth = mlWallet > 0 ? ((mlWallet - DEFAULT_CAPITAL) / DEFAULT_CAPITAL * 100).toFixed(0) : '0'
  const mlLastTrade = mlTrades && mlTrades.length > 0 ? mlTrades[0] : null
  const mlActive = ml?.ml_model_loaded ?? false

  // Combined recent trades (last 10, sorted by exit_time desc)
  const v14Tagged = (trades || []).map(t => ({ ...t, source: 'V1.4' }))
  const mlTagged = (mlTrades || []).map(t => ({ ...t, source: 'ML' }))
  const combined = [...v14Tagged, ...mlTagged]
    .sort((a, b) => {
      const ta = a.exit_time ? new Date(a.exit_time).getTime() : 0
      const tb = b.exit_time ? new Date(b.exit_time).getTime() : 0
      return tb - ta
    })
    .slice(0, 10)

  return (
    <div className="ov-page">
      {/* Hero Bar */}
      <div className="ov-hero">
        <div className="ov-hero-item">
          <div className="ov-hero-value">${totalBalance.toFixed(2)}</div>
          <div className="ov-hero-label">Total Balance</div>
        </div>
        <div className="ov-hero-item">
          <div className={`ov-hero-value ${todayTotalBps >= 0 ? 'positive' : 'negative'}`}>
            {formatBps(todayTotalBps)} bps
          </div>
          <div className="ov-hero-label">Today's P&L</div>
        </div>
        <div className="ov-hero-item">
          <div className="ov-hero-value">{activePositions}</div>
          <div className="ov-hero-label">Active Position{activePositions !== 1 ? 's' : ''}</div>
        </div>
      </div>

      {/* Bot Cards */}
      <div className="ov-cards">
        {/* V1.4 Card */}
        <div className="ov-card">
          <div className="ov-card-header">
            <span className="ov-card-title">V1.4 Strategy</span>
            <span className="ov-status-dot ov-status-running" />
            <span className="ov-status-text">Running</span>
          </div>
          <div className="ov-card-body">
            <div className="ov-card-row">
              <span className="ov-card-label">Wallet</span>
              <span className="ov-card-value">
                ${v14Wallet.toFixed(2)}
                <span className={`ov-growth ${Number(v14Growth) >= 0 ? 'positive' : 'negative'}`}>
                  {' '}({Number(v14Growth) >= 0 ? '+' : ''}{v14Growth}%)
                </span>
              </span>
            </div>
            <div className="ov-card-row">
              <span className="ov-card-label">P&L</span>
              <span className={`ov-card-value ${v14TotalBps >= 0 ? 'positive' : 'negative'}`}>
                {formatBps(v14TotalBps)} bps
              </span>
            </div>
            <div className="ov-card-row">
              <span className="ov-card-label">Trades</span>
              <span className="ov-card-value">{v14Trades} | Win: {v14WinRate}%</span>
            </div>
            <div className="ov-card-row">
              <span className="ov-card-label">Drawdown</span>
              <span className="ov-card-value">{v14Drawdown.toFixed(1)}%</span>
            </div>
            <div className="ov-card-row">
              <span className="ov-card-label">Last</span>
              <span className="ov-card-value">
                {v14LastTrade ? (
                  <>
                    <span className={v14LastTrade.net_profit_bps >= 0 ? 'positive' : 'negative'}>
                      {formatBps(v14LastTrade.net_profit_bps)} bps
                    </span>
                    {' '}
                    <span className="ov-time-ago">({timeAgo(v14LastTrade.exit_time)})</span>
                  </>
                ) : '---'}
              </span>
            </div>
          </div>
        </div>

        {/* ML Card */}
        <div className="ov-card">
          <div className="ov-card-header">
            <span className="ov-card-title">ML Strategy</span>
            <span className={`ov-status-dot ${mlActive ? 'ov-status-running' : 'ov-status-inactive'}`} />
            <span className="ov-status-text">{mlActive ? 'Active' : 'Inactive'}</span>
          </div>
          <div className="ov-card-body">
            <div className="ov-card-row">
              <span className="ov-card-label">Wallet</span>
              <span className="ov-card-value">
                ${mlWallet.toFixed(2)}
                <span className={`ov-growth ${Number(mlGrowth) >= 0 ? 'positive' : 'negative'}`}>
                  {' '}({Number(mlGrowth) >= 0 ? '+' : ''}{mlGrowth}%)
                </span>
              </span>
            </div>
            <div className="ov-card-row">
              <span className="ov-card-label">P&L</span>
              <span className={`ov-card-value ${mlTotalBps >= 0 ? 'positive' : 'negative'}`}>
                {formatBps(mlTotalBps)} bps
              </span>
            </div>
            <div className="ov-card-row">
              <span className="ov-card-label">Trades</span>
              <span className="ov-card-value">{mlTotalTrades} | Win: {mlWinRate}%</span>
            </div>
            <div className="ov-card-row">
              <span className="ov-card-label">Drawdown</span>
              <span className="ov-card-value">{mlDrawdown.toFixed(1)}%</span>
            </div>
            <div className="ov-card-row">
              <span className="ov-card-label">Last</span>
              <span className="ov-card-value">
                {mlLastTrade ? (
                  <>
                    <span className={mlLastTrade.net_profit_bps >= 0 ? 'positive' : 'negative'}>
                      {formatBps(mlLastTrade.net_profit_bps)} bps
                    </span>
                    {' '}
                    <span className="ov-time-ago">({timeAgo(mlLastTrade.exit_time)})</span>
                  </>
                ) : '---'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Recent Trades Table */}
      <div className="ov-trades">
        <div className="ov-trades-header">Recent Trades</div>
        {combined.length === 0 ? (
          <div className="ov-trades-empty">No trades yet</div>
        ) : (
          <table className="ov-trades-table">
            <thead>
              <tr>
                <th>Time</th>
                <th>Signal</th>
                <th>Dir</th>
                <th>Entry</th>
                <th>Exit</th>
                <th>P&L bps</th>
                <th>Exit Reason</th>
              </tr>
            </thead>
            <tbody>
              {combined.map((t, i) => {
                const sigColor = SIGNAL_COLORS[t.signal_type] || '#8892a0'
                const dirLabel = t.direction === 'LONG' ? 'LONG' : 'SHORT'
                const dirClass = t.direction === 'LONG' ? 'positive' : 'negative'
                const pnlClass = (t.net_profit_bps || 0) >= 0 ? 'positive' : 'negative'
                return (
                  <tr key={`${t.source}-${t.trade_id || i}`}>
                    <td className="ov-td-time">{formatDateTime(t.exit_time)}</td>
                    <td>
                      <span className="ov-signal-badge" style={{ color: sigColor, borderColor: sigColor }}>
                        {t.signal_type || '---'}
                      </span>
                    </td>
                    <td className={dirClass}>{dirLabel}</td>
                    <td>{formatPrice(t.entry_price)}</td>
                    <td>{formatPrice(t.exit_price)}</td>
                    <td className={pnlClass}>{formatBps(t.net_profit_bps)}</td>
                    <td className="ov-td-reason">{t.exit_reason || '---'}</td>
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

export default OverviewPage
