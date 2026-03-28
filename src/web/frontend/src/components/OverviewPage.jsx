import React from 'react'

const DEFAULT_CAPITAL = 5

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

function ProgressBar({ value, max, color, bg }) {
  const pct = max > 0 ? Math.min((value / max) * 100, 100) : 0
  return (
    <div style={{ height: 6, borderRadius: 3, background: bg || 'rgba(255,255,255,0.06)', overflow: 'hidden' }}>
      <div style={{ width: `${pct}%`, height: '100%', borderRadius: 3, background: color, transition: 'width 0.3s' }} />
    </div>
  )
}

function buildWalletHistory(trades, startCapital) {
  const sorted = [...(trades || [])].reverse()
  let wallet = startCapital
  const points = [{ value: wallet }]
  for (const t of sorted) {
    wallet += wallet * ((t.net_profit_bps || 0) / 10000)
    points.push({ value: wallet })
  }
  return points
}

function Sparkline({ data, width = 80, height = 30, color }) {
  if (!data || data.length < 2) return null
  const values = data.map(d => d.value)
  const min = Math.min(...values)
  const max = Math.max(...values)
  const range = max - min || 1
  const points = values.map((v, i) => {
    const x = (i / (values.length - 1)) * width
    const y = height - ((v - min) / range) * (height - 4) - 2
    return `${x},${y}`
  }).join(' ')
  return (
    <svg width={width} height={height} style={{ display: 'block' }}>
      <polyline points={points} fill="none" stroke={color} strokeWidth="1.5" />
    </svg>
  )
}

function WinRateGauge({ rate, size = 44 }) {
  const r = rate || 0
  const radius = (size - 6) / 2
  const circumference = 2 * Math.PI * radius
  const filled = (r / 100) * circumference
  const color = r >= 60 ? '#00d97e' : r >= 45 ? '#f59e0b' : '#e63757'
  return (
    <svg width={size} height={size} style={{ display: 'block', margin: '0 auto' }}>
      <circle cx={size / 2} cy={size / 2} r={radius} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="3" />
      <circle cx={size / 2} cy={size / 2} r={radius} fill="none" stroke={color} strokeWidth="3"
        strokeDasharray={`${filled} ${circumference}`}
        strokeLinecap="round"
        transform={`rotate(-90 ${size / 2} ${size / 2})`}
      />
      <text x={size / 2} y={size / 2 + 1} textAnchor="middle" dominantBaseline="middle"
        fill={color} fontSize="11" fontWeight="700">{r}%</text>
    </svg>
  )
}

function PnlBar({ bps, maxBps = 500 }) {
  if (bps == null) return null
  const absBps = Math.abs(bps)
  const pct = Math.min((absBps / maxBps) * 100, 100)
  const color = bps >= 0 ? '#00d97e' : '#e63757'
  return (
    <div style={{
      width: 60,
      height: 6,
      borderRadius: 3,
      background: 'rgba(255,255,255,0.06)',
      overflow: 'hidden',
      display: 'inline-block',
      verticalAlign: 'middle',
      marginLeft: 6,
    }}>
      <div style={{
        width: `${pct}%`,
        height: '100%',
        borderRadius: 3,
        background: color,
        transition: 'width 0.3s',
      }} />
    </div>
  )
}

function BotCard({ title, active, wallet, growth, totalBps, trades, winRate, drawdown, lastTrade, accentColor, tradeHistory }) {
  const growthNum = Number(growth)
  const ddPct = (drawdown * 100)
  const ddColor = ddPct > 15 ? '#e63757' : ddPct > 5 ? '#f59e0b' : '#00d97e'
  const wrNum = Number(winRate)

  const walletHistory = buildWalletHistory(tradeHistory, DEFAULT_CAPITAL)
  const currentAboveStart = wallet >= DEFAULT_CAPITAL
  const sparklineColor = currentAboveStart ? '#00d97e' : '#e63757'

  return (
    <div className="db-bot-card" style={{ borderTop: `3px solid ${accentColor}` }}>
      {/* Header */}
      <div className="db-bot-header">
        <div className="db-bot-title">{title}</div>
        <div className="db-bot-status">
          <span className={`db-dot ${active ? 'db-dot-green' : 'db-dot-red'}`} />
          <span className="db-status-label">{active ? 'Running' : 'Inactive'}</span>
        </div>
      </div>

      {/* Wallet with Sparkline */}
      <div className="db-wallet-section">
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div>
            <div className="db-wallet-amount">${wallet.toFixed(2)}</div>
            <div className={`db-wallet-growth ${growthNum >= 0 ? 'positive' : 'negative'}`}>
              {growthNum >= 0 ? '+' : ''}{growth}% from $5
            </div>
          </div>
          <Sparkline data={walletHistory} width={80} height={30} color={sparklineColor} />
        </div>
      </div>

      {/* Stats Grid */}
      <div className="db-stats-grid">
        <div className="db-stat-item">
          <div className={`db-stat-value ${totalBps >= 0 ? 'positive' : 'negative'}`} style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            {formatBps(totalBps)}
            <PnlBar bps={totalBps} />
          </div>
          <div className="db-stat-label">P&L (bps)</div>
        </div>
        <div className="db-stat-item">
          <div className="db-stat-value">{trades}</div>
          <div className="db-stat-label">Trades</div>
        </div>
        <div className="db-stat-item">
          <WinRateGauge rate={wrNum} />
          <div className="db-stat-label" style={{ marginTop: 4 }}>Win Rate</div>
        </div>
        <div className="db-stat-item">
          <div className="db-stat-value" style={{ color: ddColor }}>{ddPct.toFixed(1)}%</div>
          <div className="db-stat-label">Drawdown</div>
          <ProgressBar value={ddPct} max={20} color={ddColor} />
        </div>
      </div>

      {/* Last Trade */}
      <div className="db-last-trade">
        {lastTrade ? (
          <>
            <span className="db-last-label">Last trade:</span>
            <span className={`db-last-value ${lastTrade.net_profit_bps >= 0 ? 'positive' : 'negative'}`}>
              {formatBps(lastTrade.net_profit_bps)} bps
            </span>
            <span className="db-last-time">{timeAgo(lastTrade.exit_time)}</span>
          </>
        ) : (
          <span className="db-last-label">No trades yet</span>
        )}
      </div>
    </div>
  )
}

function OverviewPage({ stats, risk, ml, trades, mlTrades, status, position }) {
  const v14Wallet = risk?.wallet_usd ?? 0
  const mlWallet = ml?.ml_wallet_usd ?? 0
  const totalBalance = v14Wallet + mlWallet

  const todayV14Bps = (trades || []).filter(t => isToday(t.exit_time)).reduce((s, t) => s + (t.net_profit_bps || 0), 0)
  const todayMlBps = (mlTrades || []).filter(t => isToday(t.exit_time)).reduce((s, t) => s + (t.net_profit_bps || 0), 0)
  const todayTotalBps = todayV14Bps + todayMlBps
  const todayTrades = (trades || []).filter(t => isToday(t.exit_time)).length + (mlTrades || []).filter(t => isToday(t.exit_time)).length

  const activePositions = position?.has_position ? 1 : 0

  const v14TotalBps = stats?.total_bps ?? 0
  const v14Trades = stats?.total_trades ?? 0
  const v14WinRate = stats?.win_rate ? (stats.win_rate * 100).toFixed(0) : '0'
  const v14Drawdown = risk?.drawdown_pct ?? 0
  const v14Growth = v14Wallet > 0 ? ((v14Wallet - DEFAULT_CAPITAL) / DEFAULT_CAPITAL * 100).toFixed(0) : '0'
  const v14LastTrade = trades && trades.length > 0 ? trades[0] : null

  const mlTotalBps = ml?.ml_total_bps ?? 0
  const mlTotalTrades = ml?.ml_total_trades ?? 0
  const mlWinRate = ml?.ml_win_rate ? (ml.ml_win_rate * 100).toFixed(0) : '0'
  const mlDrawdown = ml?.ml_drawdown_pct ?? 0
  const mlGrowth = mlWallet > 0 ? ((mlWallet - DEFAULT_CAPITAL) / DEFAULT_CAPITAL * 100).toFixed(0) : '0'
  const mlLastTrade = mlTrades && mlTrades.length > 0 ? mlTrades[0] : null
  const mlActive = ml?.ml_model_loaded ?? false

  const pos = position || {}
  const pnlBps = pos.current_pnl_bps || 0
  const pnlClass = pnlBps >= 0 ? 'positive' : 'negative'
  const barsHeld = pos.bars_held || 0
  const maxBars = pos.max_bars || 10
  const barsPct = Math.min((barsHeld / maxBars) * 100, 100)

  return (
    <div className="db-page">
      {/* Hero Cards */}
      <div className="db-hero">
        <div className="db-hero-card db-hero-balance">
          <div className="db-hero-icon">$</div>
          <div>
            <div className="db-hero-value">${totalBalance.toFixed(2)}</div>
            <div className="db-hero-label">Total Balance</div>
          </div>
        </div>
        <div className={`db-hero-card ${todayTotalBps >= 0 ? 'db-hero-profit' : 'db-hero-loss'}`}>
          <div className="db-hero-icon">{todayTotalBps >= 0 ? '\u2191' : '\u2193'}</div>
          <div>
            <div className={`db-hero-value ${todayTotalBps >= 0 ? 'positive' : 'negative'}`}>
              {formatBps(todayTotalBps)} bps
            </div>
            <div className="db-hero-label">Today ({todayTrades} trade{todayTrades !== 1 ? 's' : ''})</div>
          </div>
        </div>
        <div className="db-hero-card db-hero-positions">
          <div className="db-hero-icon">{activePositions > 0 ? '\u25CF' : '\u25CB'}</div>
          <div>
            <div className="db-hero-value">{activePositions}</div>
            <div className="db-hero-label">Active Position{activePositions !== 1 ? 's' : ''}</div>
          </div>
        </div>
        <div className="db-hero-card db-hero-connection">
          <div className="db-hero-icon">{status?.status === 'RUNNING' ? '\u26A1' : '\u23F8'}</div>
          <div>
            <div className="db-hero-value">{status?.status || '---'}</div>
            <div className="db-hero-label">Bot Status</div>
          </div>
        </div>
      </div>

      {/* Bot Cards */}
      <div className="db-bots">
        <BotCard
          title="V1.4 Strategy"
          active={true}
          wallet={v14Wallet}
          growth={v14Growth}
          totalBps={v14TotalBps}
          trades={v14Trades}
          winRate={v14WinRate}
          drawdown={v14Drawdown}
          lastTrade={v14LastTrade}
          accentColor="#3b82f6"
          tradeHistory={trades}
        />
        <BotCard
          title="ML Strategy"
          active={mlActive}
          wallet={mlWallet}
          growth={mlGrowth}
          totalBps={mlTotalBps}
          trades={mlTotalTrades}
          winRate={mlWinRate}
          drawdown={mlDrawdown}
          lastTrade={mlLastTrade}
          accentColor="#8b5cf6"
          tradeHistory={mlTrades}
        />
      </div>

      {/* Active Position */}
      <div className="db-trades-card">
        <div className="db-trades-header">Active Position</div>
        {!pos.has_position ? (
          <div className="db-trades-empty">No open position</div>
        ) : (
          <div className="db-position">
            <div className="db-pos-header">
              <span className={`db-pos-side ${pos.side === 'LONG' ? 'positive' : 'negative'}`}>
                {pos.side}
              </span>
              {pos.is_reentry && <span className="db-pos-re">RE-ENTRY</span>}
              <span className="db-pos-entry">@ {formatPrice(pos.entry_price)}</span>
              <span className="db-pos-current">Now: {formatPrice(pos.current_price)}</span>
            </div>

            <div className="db-pos-stats">
              <div className="db-pos-stat">
                <div className={`db-pos-stat-value ${pnlClass}`} style={{ fontSize: 24 }}>
                  {formatBps(pnlBps)} bps
                </div>
                <div className="db-pos-stat-label">Current P&L</div>
              </div>
              <div className="db-pos-stat">
                <div className="db-pos-stat-value positive">{formatBps(pos.mfe_bps)}</div>
                <div className="db-pos-stat-label">MFE</div>
              </div>
              <div className="db-pos-stat">
                <div className="db-pos-stat-value negative">{formatBps(pos.mae_bps)}</div>
                <div className="db-pos-stat-label">MAE</div>
              </div>
              <div className="db-pos-stat">
                <div className="db-pos-stat-value">{pos.trailing_stop_bps || '---'} bps</div>
                <div className="db-pos-stat-label">Trailing Stop</div>
              </div>
              <div className="db-pos-stat">
                <div className="db-pos-stat-value">{formatBps(pos.highest_profit_bps)}</div>
                <div className="db-pos-stat-label">Peak Profit</div>
              </div>
            </div>

            <div className="db-pos-bar-section">
              <div className="db-pos-bar-label">
                <span>Bar {barsHeld} / {maxBars}</span>
                <span>{barsHeld >= maxBars ? 'TIME EXIT' : `${maxBars - barsHeld} bars left`}</span>
              </div>
              <ProgressBar value={barsHeld} max={maxBars} color={barsHeld >= maxBars - 1 ? '#e63757' : barsHeld >= maxBars - 3 ? '#f59e0b' : '#3b82f6'} />
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default OverviewPage
