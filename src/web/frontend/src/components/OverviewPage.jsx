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

function formatElapsed(days) {
  if (days == null || days < 0) return '0m'
  if (days < 1 / 24) return `${Math.round(days * 24 * 60)}m`
  if (days < 1) return `${(days * 24).toFixed(1)}h`
  return `${days.toFixed(1)}d`
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

const STATUS_META = {
  on_track: { color: '#00d97e', label: 'On Track' },
  drift:    { color: '#f59e0b', label: 'Drift' },
  diverged: { color: '#e63757', label: 'Diverged' },
  pending:  { color: '#8892a0', label: 'Pending' },
}

function StatusDot({ status }) {
  const meta = STATUS_META[status] || STATUS_META.pending
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
      <span style={{
        width: 8, height: 8, borderRadius: '50%', background: meta.color,
        boxShadow: `0 0 6px ${meta.color}80`,
      }} />
      <span style={{ fontSize: 11, color: meta.color, fontWeight: 600 }}>{meta.label}</span>
    </span>
  )
}

function MonitoringRow({ label, actual, expected, delta, status }) {
  const meta = STATUS_META[status] || STATUS_META.pending
  const isPending = status === 'pending'
  return (
    <div className="db-monitoring-row">
      <div className="db-monitoring-row-label">{label}</div>
      <div className="db-monitoring-row-body">
        <span className={`db-monitoring-num ${actual >= 0 ? 'positive' : 'negative'}`}>
          {formatBps(actual)}
        </span>
        <span className="db-monitoring-slash">/</span>
        <span className="db-monitoring-num-exp">{formatBps(expected)} <span className="db-monitoring-unit">bps exp</span></span>
        <span className="db-monitoring-spacer" />
        {isPending ? (
          <span className="db-monitoring-delta db-monitoring-pending">---</span>
        ) : (
          <span className={`db-monitoring-delta ${delta >= 0 ? 'positive' : 'negative'}`}>
            {delta >= 0 ? '+' : ''}{delta.toFixed(1)}%
          </span>
        )}
        <span style={{ marginLeft: 8 }}>
          <span style={{ width: 7, height: 7, borderRadius: '50%', background: meta.color,
            boxShadow: `0 0 5px ${meta.color}80`, display: 'inline-block' }} />
        </span>
      </div>
    </div>
  )
}

function BotCard({ title, active, wallet, growth, totalBps, trades, winRate, drawdown, lastTrade, accentColor, tradeHistory, prediction, exitVersion, monitoring, onTradesClick }) {
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
        <div className="db-bot-title">
          {title}
          {exitVersion ? (
            <span className={`db-exit-badge db-exit-${exitVersion.toLowerCase()}`}>
              Exit {exitVersion}
            </span>
          ) : null}
        </div>
        <div className="db-bot-status">
          {monitoring ? <StatusDot status={monitoring.cum.status} /> : null}
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
        <div className="db-stat-item" onClick={onTradesClick} style={onTradesClick ? { cursor: 'pointer' } : undefined}>
          <div className="db-stat-value" style={onTradesClick ? { textDecoration: 'underline', textDecorationStyle: 'dotted', textUnderlineOffset: 3 } : undefined}>{trades}</div>
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

      {/* Expected vs Actual Monitoring */}
      {monitoring && (
        <div className="db-monitoring">
          <div className="db-monitoring-title">
            Expected vs Actual <span className="db-monitoring-sub">({formatElapsed(monitoring.daysElapsed)} elapsed · {formatBps(monitoring.dailyExpected)} bps/day expected)</span>
          </div>
          <MonitoringRow
            label="Last 7d"
            actual={monitoring.window7d.actual}
            expected={monitoring.window7d.expected}
            delta={monitoring.window7d.delta}
            status={monitoring.window7d.status}
          />
          <MonitoringRow
            label="Cumulative"
            actual={monitoring.cum.actual}
            expected={monitoring.cum.expected}
            delta={monitoring.cum.delta}
            status={monitoring.cum.status}
          />
        </div>
      )}

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

// Legacy filter labels for the Trades page (TradesList still uses these)
const TRADES_FILTER_LABEL = {
  ML_V1: 'ML',
  ML_ATTN: 'ML V2',
  ML_V3: 'ML V3',
}

function OverviewPage({ models, modelTrades, status, onTradesNavigate }) {
  // One list drives everything: [name, panel, trades] per registered model.
  const entries = Object.entries(models || {}).map(([name, panel]) => ({
    name,
    panel: panel || {},
    trades: (modelTrades || {})[name] || [],
  }))

  const totalBalance = entries.reduce((s, e) => s + (e.panel.wallet_usd || 0), 0)
  const todayTotalBps = entries.reduce((s, e) =>
    s + e.trades.filter(t => isToday(t.exit_time)).reduce((a, t) => a + (t.net_profit_bps || 0), 0), 0)
  const todayTrades = entries.reduce((s, e) =>
    s + e.trades.filter(t => isToday(t.exit_time)).length, 0)
  const activePositions = entries.filter(e => e.panel.has_position).length
  const inPosition = entries.filter(e => e.panel.has_position)

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

      {/* Bot Cards — one per registered model, in sort order */}
      <div className="db-bots">
        {entries.map(({ name, panel, trades }) => {
          const wallet = panel.wallet_usd || 0
          const growth = wallet > 0 ? ((wallet - DEFAULT_CAPITAL) / DEFAULT_CAPITAL * 100).toFixed(0) : '0'
          const hasBaseline = (panel.daily_expected_bps || 0) > 0
          return (
            <BotCard
              key={name}
              title={panel.title || name}
              active={panel.model_loaded || false}
              wallet={wallet}
              growth={growth}
              totalBps={panel.total_bps || 0}
              trades={panel.total_trades || 0}
              winRate={panel.win_rate ? (panel.win_rate * 100).toFixed(0) : '0'}
              drawdown={panel.drawdown_pct || 0}
              lastTrade={trades.length > 0 ? trades[0] : null}
              accentColor={panel.accent_color}
              tradeHistory={trades}
              exitVersion={panel.exit_version}
              prediction={hasBaseline ? {
                longPct: panel.long_pct ?? 50,
                shortPct: panel.short_pct ?? 50,
                status: panel.signal_status ?? '',
              } : undefined}
              monitoring={hasBaseline ? {
                dailyExpected: panel.daily_expected_bps ?? 0,
                daysElapsed: panel.days_elapsed ?? 0,
                window7d: {
                  actual: panel.d7_actual_bps ?? 0,
                  expected: panel.d7_expected_bps ?? 0,
                  delta: panel.d7_delta_pct ?? 0,
                  status: panel.d7_status ?? 'pending',
                },
                cum: {
                  actual: panel.cum_actual_bps ?? 0,
                  expected: panel.cum_expected_bps ?? 0,
                  delta: panel.cum_delta_pct ?? 0,
                  status: panel.cum_status ?? 'pending',
                },
              } : undefined}
              onTradesClick={() => onTradesNavigate?.(TRADES_FILTER_LABEL[name] || name)}
            />
          )
        })}
      </div>

      {/* Active Positions — one block per model in a position */}
      <div className="db-trades-card">
        <div className="db-trades-header">Active Positions</div>
        {inPosition.length === 0 ? (
          <div className="db-trades-empty">No open positions</div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            {inPosition.map(({ name, panel }) => (
              <PositionBlock
                key={name}
                label={panel.title || name}
                accentColor={panel.accent_color}
                side={panel.position_side}
                entryPrice={panel.position_entry_price}
                currentPrice={status?.price}
                pnlBps={panel.position_pnl_bps || 0}
                mfeBps={panel.position_mfe_bps || 0}
                maeBps={panel.position_mae_bps || 0}
                trailingStopBps={panel.position_trailing_stop_bps || 0}
                highestProfitBps={panel.position_highest_profit_bps || 0}
                barsHeld={panel.position_bars_held || 0}
                maxBars={panel.position_max_bars || 10}
                isReentry={panel.is_reentry || false}
                liqPrice={panel.position_liq_price || 0}
              />
            ))}
          </div>
        )}
      </div>

    </div>
  )
}

export default OverviewPage
