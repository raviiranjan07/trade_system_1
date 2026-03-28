import React from 'react'

const DEFAULT_CAPITAL = 5

function formatTimestamp(ts) {
  if (!ts) return '---'
  const d = new Date(ts)
  const date = d.toLocaleDateString('en-GB', { day: '2-digit', month: 'short' })
  const time = d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false })
  return `${date} ${time}`
}

function DrawdownBar({ pct }) {
  const val = Math.min(Math.max(pct || 0, 0), 100)
  let barColor = '#00d97e'
  if (val > 15) barColor = '#e63757'
  else if (val > 5) barColor = '#f59e0b'

  return (
    <div className="rp-dd-bar-container">
      <div className="rp-dd-bar-track">
        <div
          className="rp-dd-bar-fill"
          style={{ width: `${val}%`, backgroundColor: barColor }}
        />
      </div>
      <span className="rp-dd-bar-label">{val.toFixed(1)}%</span>
    </div>
  )
}

function RiskPage({ risk, ml, decisions }) {
  const v14Wallet = risk?.wallet_usd ?? 0
  const mlWallet = ml?.ml_wallet_usd ?? 0
  const totalCapital = v14Wallet + mlWallet

  const v14Peak = risk?.peak_usd ?? v14Wallet
  const mlPeak = ml?.ml_peak_usd ?? mlWallet

  const v14Growth = ((v14Wallet - DEFAULT_CAPITAL) / DEFAULT_CAPITAL * 100).toFixed(1)
  const mlGrowth = ((mlWallet - DEFAULT_CAPITAL) / DEFAULT_CAPITAL * 100).toFixed(1)

  const v14DD = risk?.drawdown_pct ?? 0
  const mlDD = ml?.ml_drawdown_pct ?? 0
  const combinedDD = totalCapital > 0
    ? ((1 - totalCapital / (v14Peak + mlPeak)) * 100)
    : 0
  const combinedDDClamped = Math.max(0, combinedDD)

  const v14Health = risk?.health_multiplier ?? 1
  const mlHealth = ml?.ml_health_multiplier ?? 1

  const v14Streak = risk?.consecutive_losses ?? 0
  const mlStreak = ml?.ml_consecutive_losses ?? 0

  const v14WR = risk?.recent_winrate != null ? (risk.recent_winrate * 100).toFixed(0) : '---'
  const mlWR = ml?.ml_recent_winrate != null ? (ml.ml_recent_winrate * 100).toFixed(0) : '---'

  const v14Skips = risk?.total_skips ?? 0
  const mlSkips = ml?.ml_total_skips ?? 0

  const recentDecisions = (decisions || []).slice(0, 20)

  return (
    <div className="rp-page">
      {/* Hero */}
      <div className="rp-hero">
        <div className="rp-hero-item">
          <div className="rp-hero-value">${totalCapital.toFixed(2)}</div>
          <div className="rp-hero-label">Total Capital</div>
        </div>
        <div className="rp-hero-item">
          <div className="rp-hero-value">{combinedDDClamped.toFixed(1)}%</div>
          <div className="rp-hero-label">Combined Drawdown</div>
        </div>
      </div>

      {/* Two columns */}
      <div className="rp-cards">
        {/* V1.4 Card */}
        <div className="rp-card">
          <div className="rp-card-title">V1.4 Strategy</div>
          <div className="rp-card-body">
            <div className="rp-row">
              <span className="rp-label">Wallet</span>
              <span className="rp-value">${v14Wallet.toFixed(2)}</span>
            </div>
            <div className="rp-row">
              <span className="rp-label">Peak</span>
              <span className="rp-value">${v14Peak.toFixed(2)}</span>
            </div>
            <div className="rp-row">
              <span className="rp-label">Growth</span>
              <span className={`rp-value ${Number(v14Growth) >= 0 ? 'positive' : 'negative'}`}>
                {Number(v14Growth) >= 0 ? '+' : ''}{v14Growth}% from ${DEFAULT_CAPITAL}
              </span>
            </div>
            <div className="rp-row">
              <span className="rp-label">Drawdown</span>
              <DrawdownBar pct={v14DD} />
            </div>
            <div className="rp-row">
              <span className="rp-label">Health</span>
              <span className="rp-value">{v14Health.toFixed(2)}</span>
            </div>
            <div className="rp-row">
              <span className="rp-label">Streak</span>
              <span className="rp-value">{v14Streak} loss{v14Streak !== 1 ? 'es' : ''}</span>
            </div>
            <div className="rp-row">
              <span className="rp-label">WR (20)</span>
              <span className="rp-value">{v14WR}%</span>
            </div>
            <div className="rp-row">
              <span className="rp-label">Skips</span>
              <span className="rp-value">{v14Skips}</span>
            </div>
          </div>
        </div>

        {/* ML Card */}
        <div className="rp-card">
          <div className="rp-card-title">ML Strategy</div>
          <div className="rp-card-body">
            <div className="rp-row">
              <span className="rp-label">Wallet</span>
              <span className="rp-value">${mlWallet.toFixed(2)}</span>
            </div>
            <div className="rp-row">
              <span className="rp-label">Peak</span>
              <span className="rp-value">${mlPeak.toFixed(2)}</span>
            </div>
            <div className="rp-row">
              <span className="rp-label">Growth</span>
              <span className={`rp-value ${Number(mlGrowth) >= 0 ? 'positive' : 'negative'}`}>
                {Number(mlGrowth) >= 0 ? '+' : ''}{mlGrowth}% from ${DEFAULT_CAPITAL}
              </span>
            </div>
            <div className="rp-row">
              <span className="rp-label">Drawdown</span>
              <DrawdownBar pct={mlDD} />
            </div>
            <div className="rp-row">
              <span className="rp-label">Health</span>
              <span className="rp-value">{mlHealth.toFixed(2)}</span>
            </div>
            <div className="rp-row">
              <span className="rp-label">Streak</span>
              <span className="rp-value">{mlStreak} loss{mlStreak !== 1 ? 'es' : ''}</span>
            </div>
            <div className="rp-row">
              <span className="rp-label">WR (20)</span>
              <span className="rp-value">{mlWR}%</span>
            </div>
            <div className="rp-row">
              <span className="rp-label">Skips</span>
              <span className="rp-value">{mlSkips}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Decision Log */}
      <div className="rp-decisions">
        <div className="rp-decisions-header">Decision Log</div>
        {recentDecisions.length === 0 ? (
          <div className="rp-decisions-empty">No decisions recorded</div>
        ) : (
          <div className="rp-decisions-scroll">
            <table className="rp-decisions-table">
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Signal</th>
                  <th>Action</th>
                  <th>Wallet</th>
                  <th>BTC Price</th>
                  <th>Risk%</th>
                  <th>P&L bps</th>
                  <th>P&L $</th>
                </tr>
              </thead>
              <tbody>
                {recentDecisions.map((d, i) => {
                  const pnlBpsClass = d.pnl_bps != null && d.pnl_bps !== ''
                    ? (Number(d.pnl_bps) >= 0 ? 'positive' : 'negative')
                    : ''
                  const pnlUsdClass = d.pnl_usd != null && d.pnl_usd !== ''
                    ? (Number(d.pnl_usd) >= 0 ? 'positive' : 'negative')
                    : ''
                  const actionClass = d.action === 'SKIP' ? 'rp-action-skip' : 'rp-action-trade'
                  return (
                    <tr key={i}>
                      <td className="rp-td-time">{formatTimestamp(d.timestamp)}</td>
                      <td>{d.signal_type || '---'}</td>
                      <td><span className={actionClass}>{d.action || '---'}</span></td>
                      <td>${Number(d.wallet_usd || 0).toFixed(2)}</td>
                      <td>${Number(d.btc_price || 0).toLocaleString('en-US', { maximumFractionDigits: 0 })}</td>
                      <td>{d.risk_pct != null ? (Number(d.risk_pct) * 100).toFixed(1) + '%' : '---'}</td>
                      <td className={pnlBpsClass}>
                        {d.pnl_bps != null && d.pnl_bps !== ''
                          ? `${Number(d.pnl_bps) >= 0 ? '+' : ''}${Number(d.pnl_bps).toFixed(1)}`
                          : '---'}
                      </td>
                      <td className={pnlUsdClass}>
                        {d.pnl_usd != null && d.pnl_usd !== ''
                          ? `${Number(d.pnl_usd) >= 0 ? '+' : ''}$${Number(d.pnl_usd).toFixed(4)}`
                          : '---'}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}

export default RiskPage
