import React, { useEffect, useRef } from 'react'
import { createChart, ColorType, LineSeries } from 'lightweight-charts'

const DEFAULT_CAPITAL = 5

function formatTimestamp(ts) {
  if (!ts) return '---'
  const d = new Date(ts)
  const date = d.toLocaleDateString('en-GB', { day: '2-digit', month: 'short' })
  const time = d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false })
  return `${date} ${time}`
}

function buildWalletHistory(tradesArr, startCapital) {
  const sorted = [...(tradesArr || [])].reverse()
  let wallet = startCapital
  const points = [{ time: null, value: wallet }]
  for (const t of sorted) {
    wallet += wallet * ((t.net_profit_bps || 0) / 10000)
    points.push({ time: t.exit_time, value: wallet })
  }
  return points
}

function WalletChart({ trades, mlTrades, mlAttnTrades }) {
  const containerRef = useRef(null)
  const chartRef = useRef(null)

  const v14Points = buildWalletHistory(trades, DEFAULT_CAPITAL)
  const mlPoints = buildWalletHistory(mlTrades, DEFAULT_CAPITAL)
  const mlAttnPoints = buildWalletHistory(mlAttnTrades, DEFAULT_CAPITAL)

  useEffect(() => {
    if (!containerRef.current) return

    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height: 280,
      layout: {
        background: { type: ColorType.Solid, color: '#0a0e14' },
        textColor: '#8892a0',
        fontSize: 11,
      },
      grid: {
        vertLines: { color: '#1a2028' },
        horzLines: { color: '#1a2028' },
      },
      rightPriceScale: {
        borderColor: '#2a3544',
        scaleMargins: { top: 0.08, bottom: 0.08 },
      },
      timeScale: {
        borderColor: '#2a3544',
        timeVisible: true,
        secondsVisible: false,
      },
      crosshair: {
        horzLine: { color: '#3a4555', labelBackgroundColor: '#2a3544' },
        vertLine: { color: '#3a4555', labelBackgroundColor: '#2a3544' },
      },
      handleScroll: { mouseWheel: true, pressedMouseMove: true },
      handleScale: { mouseWheel: true, pinch: true },
    })
    chart.applyOptions({ branding: { visible: false } })

    // V1.4 line (blue)
    const v14Series = chart.addSeries(LineSeries, {
      color: '#3b82f6',
      lineWidth: 2,
      lastValueVisible: true,
      priceLineVisible: false,
      crosshairMarkerVisible: true,
      title: 'V1.4',
    })

    // ML line (purple)
    const mlSeries = chart.addSeries(LineSeries, {
      color: '#8b5cf6',
      lineWidth: 2,
      lastValueVisible: true,
      priceLineVisible: false,
      crosshairMarkerVisible: true,
      title: 'ML',
    })

    // $5 reference line
    const refLine = chart.addSeries(LineSeries, {
      color: 'rgba(136, 146, 160, 0.3)',
      lineWidth: 1,
      lineStyle: 2,
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
    })

    function toChartData(points) {
      const data = []
      for (const p of points) {
        if (!p.time) continue
        const d = new Date(p.time)
        if (isNaN(d.getTime())) continue
        const ts = Math.floor(d.getTime() / 1000)
        data.push({ time: ts, value: Math.round(p.value * 100) / 100 })
      }
      const deduped = []
      for (let i = 0; i < data.length; i++) {
        if (i === data.length - 1 || data[i].time !== data[i + 1].time) {
          deduped.push(data[i])
        }
      }
      return deduped
    }

    const v14Data = toChartData(v14Points)
    const mlData = toChartData(mlPoints)

    if (v14Data.length > 0) v14Series.setData(v14Data)
    if (mlData.length > 0) mlSeries.setData(mlData)

    const allData = [...v14Data, ...mlData]
    if (allData.length >= 2) {
      const minTime = Math.min(...allData.map(d => d.time))
      const maxTime = Math.max(...allData.map(d => d.time))
      refLine.setData([
        { time: minTime, value: DEFAULT_CAPITAL },
        { time: maxTime, value: DEFAULT_CAPITAL },
      ])
    }

    chart.timeScale().fitContent()
    chartRef.current = chart

    const handleResize = () => {
      if (containerRef.current) {
        chart.applyOptions({ width: containerRef.current.clientWidth })
      }
    }
    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      chart.remove()
    }
  }, [v14Points.length, mlPoints.length])

  return (
    <div className="rp-wallet-chart">
      <div ref={containerRef} style={{ width: '100%' }} />
    </div>
  )
}

function WinRateDonut({ pct }) {
  // pct is 0-100 or '---'
  const val = typeof pct === 'number' ? pct : (pct !== '---' ? Number(pct) : null)
  if (val == null || isNaN(val)) {
    return <span className="rp-value">---%</span>
  }

  const size = 36
  const strokeW = 5
  const r = (size - strokeW) / 2
  const cx = size / 2
  const cy = size / 2
  const circ = 2 * Math.PI * r
  const filled = (val / 100) * circ
  const color = val >= 60 ? '#00d97e' : val >= 50 ? '#f59e0b' : '#e63757'

  return (
    <span className="rp-wr-donut-wrap">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <circle cx={cx} cy={cy} r={r} fill="none" stroke="#2a2a3e" strokeWidth={strokeW} />
        <circle cx={cx} cy={cy} r={r} fill="none" stroke={color} strokeWidth={strokeW}
          strokeDasharray={`${filled} ${circ - filled}`}
          strokeDashoffset={circ * 0.25}
          strokeLinecap="round"
          transform={`rotate(-90 ${cx} ${cy})`} />
        <text x={cx} y={cy + 3} fill="#eee" fontSize="9" fontWeight="600" textAnchor="middle">
          {val}%
        </text>
      </svg>
    </span>
  )
}

function HealthGauge({ value }) {
  const val = typeof value === 'number' ? value : Number(value)
  if (isNaN(val)) return <span className="rp-value">---</span>

  const size = 40
  const strokeW = 5
  const r = (size - strokeW) / 2
  const cx = size / 2
  const cy = size / 2

  // Arc from -135 deg to +135 deg (270 degrees total)
  const totalAngle = 270
  const startAngle = 135 // degrees from top, going clockwise starting from bottom-left
  const filledAngle = Math.min(val, 1.0) * totalAngle

  const color = val >= 0.8 ? '#00d97e' : val >= 0.5 ? '#f59e0b' : '#e63757'

  function polarToXY(angleDeg) {
    const rad = ((angleDeg - 90) * Math.PI) / 180
    return {
      x: cx + r * Math.cos(rad),
      y: cy + r * Math.sin(rad)
    }
  }

  // Background arc (full 270 degrees)
  const bgStart = polarToXY(startAngle)
  const bgEnd = polarToXY(startAngle + totalAngle)
  const bgPath = `M ${bgStart.x} ${bgStart.y} A ${r} ${r} 0 1 1 ${bgEnd.x} ${bgEnd.y}`

  // Filled arc
  const fStart = polarToXY(startAngle)
  const fEnd = polarToXY(startAngle + filledAngle)
  const largeArc = filledAngle > 180 ? 1 : 0
  const fPath = filledAngle > 0
    ? `M ${fStart.x} ${fStart.y} A ${r} ${r} 0 ${largeArc} 1 ${fEnd.x} ${fEnd.y}`
    : ''

  return (
    <span className="rp-health-gauge-wrap">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <path d={bgPath} fill="none" stroke="#2a2a3e" strokeWidth={strokeW} strokeLinecap="round" />
        {fPath && (
          <path d={fPath} fill="none" stroke={color} strokeWidth={strokeW} strokeLinecap="round" />
        )}
        <text x={cx} y={cy + 4} fill="#eee" fontSize="9" fontWeight="600" textAnchor="middle">
          {val.toFixed(2)}
        </text>
      </svg>
    </span>
  )
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

function RiskPage({ risk, ml, mlAttn, decisions, trades, mlTrades, mlAttnTrades }) {
  const v14Wallet = risk?.wallet_usd ?? 0
  const mlWallet = ml?.ml_wallet_usd ?? 0
  const mlAttnWallet = mlAttn?.ml_attn_wallet_usd ?? 0
  const totalCapital = v14Wallet + mlWallet + mlAttnWallet

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

  const v14WR = risk?.recent_winrate != null ? Number((risk.recent_winrate * 100).toFixed(0)) : null
  const mlWR = ml?.ml_recent_winrate != null ? Number((ml.ml_recent_winrate * 100).toFixed(0)) : null

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

      {/* Wallet Growth Chart */}
      <WalletChart trades={trades} mlTrades={mlTrades} mlAttnTrades={mlAttnTrades} />

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
              <DrawdownBar pct={v14DD * 100} />
            </div>
            <div className="rp-row">
              <span className="rp-label">Health</span>
              <HealthGauge value={v14Health} />
            </div>
            <div className="rp-row">
              <span className="rp-label">Streak</span>
              <span className="rp-value">{v14Streak} loss{v14Streak !== 1 ? 'es' : ''}</span>
            </div>
            <div className="rp-row">
              <span className="rp-label">WR (20)</span>
              <WinRateDonut pct={v14WR} />
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
              <DrawdownBar pct={mlDD * 100} />
            </div>
            <div className="rp-row">
              <span className="rp-label">Health</span>
              <HealthGauge value={mlHealth} />
            </div>
            <div className="rp-row">
              <span className="rp-label">Streak</span>
              <span className="rp-value">{mlStreak} loss{mlStreak !== 1 ? 'es' : ''}</span>
            </div>
            <div className="rp-row">
              <span className="rp-label">WR (20)</span>
              <WinRateDonut pct={mlWR} />
            </div>
            <div className="rp-row">
              <span className="rp-label">Skips</span>
              <span className="rp-value">{mlSkips}</span>
            </div>
          </div>
        </div>

        {/* ML V3 Card */}
        <div className="rp-card">
          <div className="rp-card-title" style={{borderColor: '#f59e0b'}}>ML V3</div>
          <div className="rp-card-body">
            <div className="rp-row">
              <span className="rp-label">Wallet</span>
              <span className="rp-value">${mlAttnWallet.toFixed(2)}</span>
            </div>
            <div className="rp-row">
              <span className="rp-label">Growth</span>
              <span className={`rp-value ${mlAttnWallet >= DEFAULT_CAPITAL ? 'positive' : 'negative'}`}>
                {mlAttnWallet >= DEFAULT_CAPITAL ? '+' : ''}{((mlAttnWallet - DEFAULT_CAPITAL) / DEFAULT_CAPITAL * 100).toFixed(1)}% from ${DEFAULT_CAPITAL}
              </span>
            </div>
            <div className="rp-row">
              <span className="rp-label">Drawdown</span>
              <DrawdownBar pct={(mlAttn?.ml_attn_drawdown_pct ?? 0) * 100} />
            </div>
            <div className="rp-row">
              <span className="rp-label">Trades</span>
              <span className="rp-value">{mlAttn?.ml_attn_total_trades ?? 0}</span>
            </div>
            <div className="rp-row">
              <span className="rp-label">Win Rate</span>
              <span className="rp-value">{((mlAttn?.ml_attn_win_rate ?? 0) * 100).toFixed(0)}%</span>
            </div>
            <div className="rp-row">
              <span className="rp-label">Total bps</span>
              <span className={`rp-value ${(mlAttn?.ml_attn_total_bps ?? 0) >= 0 ? 'positive' : 'negative'}`}>
                {(mlAttn?.ml_attn_total_bps ?? 0) >= 0 ? '+' : ''}{(mlAttn?.ml_attn_total_bps ?? 0).toFixed(1)}
              </span>
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
