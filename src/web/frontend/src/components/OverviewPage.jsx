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

function BotCard({ title, active, wallet, growth, totalBps, trades, winRate, drawdown, lastTrade, accentColor, tradeHistory, prediction }) {
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

      {/* ML Prediction (only for ML card) */}
      {prediction && (
        <div className="db-prediction">
          <div className="db-prediction-header">ML Prediction</div>
          <div className="db-prediction-row">
            <span className="db-prediction-label" style={{ color: '#00d97e' }}>LONG</span>
            <div className="db-prediction-bar-track">
              <div className="db-prediction-bar-fill" style={{
                width: `${prediction.longPct}%`,
                background: prediction.longPct > 60 ? '#00d97e' : 'rgba(0, 217, 126, 0.3)',
              }} />
            </div>
            <span className="db-prediction-pct">{prediction.longPct.toFixed(1)}%</span>
          </div>
          <div className="db-prediction-row">
            <span className="db-prediction-label" style={{ color: '#e63757' }}>SHORT</span>
            <div className="db-prediction-bar-track">
              <div className="db-prediction-bar-fill" style={{
                width: `${prediction.shortPct}%`,
                background: prediction.shortPct > 65 ? '#e63757' : 'rgba(230, 55, 87, 0.3)',
              }} />
            </div>
            <span className="db-prediction-pct">{prediction.shortPct.toFixed(1)}%</span>
          </div>
          {prediction.status && (
            <div className={`db-prediction-status ${prediction.status.includes('SIGNAL') ? 'db-prediction-signal' : ''}`}>
              {prediction.status}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function PositionBlock({ label, accentColor, side, entryPrice, currentPrice, pnlBps, mfeBps, maeBps, trailingStopBps, highestProfitBps, barsHeld, maxBars, isReentry, liqPrice }) {
  const pnl = pnlBps || 0
  const pc = pnl >= 0 ? 'positive' : 'negative'
  const bh = barsHeld || 0
  const mb = maxBars || 10

  return (
    <div style={{ borderLeft: `3px solid ${accentColor}`, paddingLeft: 12 }}>
      <div className="db-pos-header">
        <span style={{ color: accentColor, fontWeight: 700, fontSize: 12, marginRight: 8 }}>{label}</span>
        <span className={`db-pos-side ${side === 'LONG' ? 'positive' : 'negative'}`}>
          {side}
        </span>
        {isReentry && <span className="db-pos-re">RE-ENTRY</span>}
        <span className="db-pos-entry">@ {formatPrice(entryPrice)}</span>
        {currentPrice ? <span className="db-pos-current">Now: {formatPrice(currentPrice)}</span> : null}
      </div>

      <div className="db-pos-stats">
        <div className="db-pos-stat">
          <div className={`db-pos-stat-value ${pc}`} style={{ fontSize: 24 }}>
            {formatBps(pnl)} bps
          </div>
          <div className="db-pos-stat-label">Current P&L</div>
        </div>
        <div className="db-pos-stat">
          <div className="db-pos-stat-value positive">{formatBps(mfeBps)}</div>
          <div className="db-pos-stat-label">MFE</div>
        </div>
        <div className="db-pos-stat">
          <div className="db-pos-stat-value negative">{formatBps(maeBps)}</div>
          <div className="db-pos-stat-label">MAE</div>
        </div>
        <div className="db-pos-stat">
          <div className="db-pos-stat-value">{trailingStopBps || '---'} bps</div>
          <div className="db-pos-stat-label">Trailing Stop</div>
        </div>
        <div className="db-pos-stat">
          <div className="db-pos-stat-value positive">{formatBps(highestProfitBps)}</div>
          <div className="db-pos-stat-label">Peak Profit</div>
        </div>
        <div className="db-pos-stat">
          <div className="db-pos-stat-value negative">{liqPrice ? formatPrice(liqPrice) : '---'}</div>
          <div className="db-pos-stat-label">Liq Price</div>
        </div>
      </div>

      <div className="db-pos-bar-section">
        <div className="db-pos-bar-label">
          <span>Bar {bh} / {mb}</span>
          <span>{bh >= mb ? 'TIME EXIT' : `${mb - bh} bars left`}</span>
        </div>
        <ProgressBar value={bh} max={mb} color={bh >= mb - 1 ? '#e63757' : bh >= mb - 3 ? '#f59e0b' : accentColor} />
      </div>
    </div>
  )
}

function OverviewPage({ stats, risk, ml, mlAttn, trades, mlTrades, mlAttnTrades, status, position }) {
  const v14Wallet = risk?.wallet_usd ?? 0
  const mlWallet = ml?.ml_wallet_usd ?? 0
  const mlAttnWallet = mlAttn?.ml_attn_wallet_usd ?? 0
  const totalBalance = v14Wallet + mlWallet + mlAttnWallet

  const todayV14Bps = (trades || []).filter(t => isToday(t.exit_time)).reduce((s, t) => s + (t.net_profit_bps || 0), 0)
  const todayMlBps = (mlTrades || []).filter(t => isToday(t.exit_time)).reduce((s, t) => s + (t.net_profit_bps || 0), 0)
  const todayMlAttnBps = (mlAttnTrades || []).filter(t => isToday(t.exit_time)).reduce((s, t) => s + (t.net_profit_bps || 0), 0)
  const todayTotalBps = todayV14Bps + todayMlBps + todayMlAttnBps
  const todayTrades = (trades || []).filter(t => isToday(t.exit_time)).length + (mlTrades || []).filter(t => isToday(t.exit_time)).length + (mlAttnTrades || []).filter(t => isToday(t.exit_time)).length

  const v14HasPos = position?.has_position ?? false
  const mlHasPos = ml?.ml_has_position ?? false
  const mlAttnHasPos_ = mlAttn?.ml_attn_has_position ?? false
  const activePositions = (v14HasPos ? 1 : 0) + (mlHasPos ? 1 : 0) + (mlAttnHasPos_ ? 1 : 0)

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

  const mlAttnTotalBps = mlAttn?.ml_attn_total_bps ?? 0
  const mlAttnTotalTrades = mlAttn?.ml_attn_total_trades ?? 0
  const mlAttnWinRate = mlAttn?.ml_attn_win_rate ? (mlAttn.ml_attn_win_rate * 100).toFixed(0) : '0'
  const mlAttnGrowth = mlAttnWallet > 0 ? ((mlAttnWallet - DEFAULT_CAPITAL) / DEFAULT_CAPITAL * 100).toFixed(0) : '0'
  const mlAttnActive = mlAttn?.ml_attn_model_loaded ?? false
  const mlAttnHasPos = mlAttn?.ml_attn_has_position ?? false

  // V1.4 position
  const pos = position || {}
  const pnlBps = pos.current_pnl_bps || 0
  const pnlClass = pnlBps >= 0 ? 'positive' : 'negative'
  const barsHeld = pos.bars_held || 0
  const maxBars = pos.max_bars || 10

  // ML position
  const mlPos = {
    has_position: ml?.ml_has_position ?? false,
    side: ml?.ml_position_side,
    entry_price: ml?.ml_position_entry_price ?? 0,
    current_pnl_bps: ml?.ml_position_pnl_bps ?? 0,
    trailing_stop_bps: ml?.ml_position_trailing_stop_bps ?? 0,
    highest_profit_bps: ml?.ml_position_highest_profit_bps ?? 0,
    bars_held: ml?.ml_position_bars_held ?? 0,
    max_bars: ml?.ml_position_max_bars ?? 10,
    mfe_bps: ml?.ml_position_mfe_bps ?? 0,
    mae_bps: ml?.ml_position_mae_bps ?? 0,
    liq_price: ml?.ml_position_liq_price ?? 0,
  }

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
          prediction={{
            longPct: ml?.ml_long_pct ?? 50,
            shortPct: ml?.ml_short_pct ?? 50,
            status: ml?.ml_signal_status ?? '',
          }}
        />
        <BotCard
          title="ML V2"
          active={mlAttnActive}
          wallet={mlAttnWallet}
          growth={mlAttnGrowth}
          totalBps={mlAttnTotalBps}
          trades={mlAttnTotalTrades}
          winRate={mlAttnWinRate}
          drawdown={mlAttn?.ml_attn_drawdown_pct ?? 0}
          lastTrade={mlAttnTrades && mlAttnTrades.length > 0 ? mlAttnTrades[0] : null}
          accentColor="#f59e0b"
          tradeHistory={mlAttnTrades}
          prediction={{
            longPct: mlAttn?.ml_attn_long_pct ?? 50,
            shortPct: mlAttn?.ml_attn_short_pct ?? 50,
            status: mlAttn?.ml_attn_signal_status ?? '',
          }}
        />
      </div>

      {/* Active Positions */}
      <div className="db-trades-card">
        <div className="db-trades-header">Active Positions</div>
        {!v14HasPos && !mlHasPos && !mlAttnHasPos ? (
          <div className="db-trades-empty">No open positions</div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            {v14HasPos && (
              <PositionBlock
                label="V1.4"
                accentColor="#3b82f6"
                side={pos.side}
                entryPrice={pos.entry_price}
                currentPrice={pos.current_price}
                pnlBps={pnlBps}
                mfeBps={pos.mfe_bps}
                maeBps={pos.mae_bps}
                trailingStopBps={pos.trailing_stop_bps}
                highestProfitBps={pos.highest_profit_bps}
                barsHeld={barsHeld}
                maxBars={maxBars}
                isReentry={pos.is_reentry}
                liqPrice={pos.liq_price}
              />
            )}
            {mlHasPos && (
              <PositionBlock
                label="ML"
                accentColor="#8b5cf6"
                side={mlPos.side}
                entryPrice={mlPos.entry_price}
                pnlBps={mlPos.current_pnl_bps}
                mfeBps={mlPos.mfe_bps}
                maeBps={mlPos.mae_bps}
                trailingStopBps={mlPos.trailing_stop_bps}
                highestProfitBps={mlPos.highest_profit_bps}
                barsHeld={mlPos.bars_held}
                maxBars={mlPos.max_bars}
                liqPrice={mlPos.liq_price}
              />
            )}
            {mlAttnHasPos && (
              <PositionBlock
                label="ML V2"
                accentColor="#f59e0b"
                side={mlAttn?.ml_attn_position_side}
                entryPrice={mlAttn?.ml_attn_position_entry_price ?? 0}
                pnlBps={mlAttn?.ml_attn_position_pnl_bps ?? 0}
                mfeBps={mlAttn?.ml_attn_position_mfe_bps ?? 0}
                maeBps={mlAttn?.ml_attn_position_mae_bps ?? 0}
                trailingStopBps={mlAttn?.ml_attn_position_trailing_stop_bps ?? 0}
                highestProfitBps={mlAttn?.ml_attn_position_highest_profit_bps ?? 0}
                barsHeld={mlAttn?.ml_attn_position_bars_held ?? 0}
                maxBars={mlAttn?.ml_attn_position_max_bars ?? 10}
                liqPrice={mlAttn?.ml_attn_position_liq_price ?? 0}
              />
            )}
          </div>
        )}
      </div>

    </div>
  )
}

export default OverviewPage
