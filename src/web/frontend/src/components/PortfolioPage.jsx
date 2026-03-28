import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react'

const POLL_INTERVAL = 30000

const STATUS_COLORS = {
  FILLED: '#00d97e',
  NEW: '#f59e0b',
  PARTIALLY_FILLED: '#f59e0b',
  CANCELED: '#5c6675',
  EXPIRED: '#5c6675',
  REJECTED: '#e63757',
}

const INCOME_LABELS = {
  REALIZED_PNL: 'Realized PNL',
  FUNDING_FEE: 'Funding Fee',
  COMMISSION: 'Commission',
  TRANSFER: 'Transfer',
  INSURANCE_CLEAR: 'Insurance',
}

function PortfolioPage() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [configured, setConfigured] = useState(null)
  const [error, setError] = useState(null)
  const [lastUpdate, setLastUpdate] = useState(null)
  const [refreshing, setRefreshing] = useState(false)
  const [activeTab, setActiveTab] = useState('orders')
  const [pnlPeriod, setPnlPeriod] = useState('week')
  const [pnlData, setPnlData] = useState(null)
  const [pnlLoading, setPnlLoading] = useState(false)
  const intervalRef = useRef(null)

  const fetchPortfolio = useCallback((manual = false) => {
    if (manual) setRefreshing(true)
    fetch('/api/portfolio')
      .then(r => r.json())
      .then(d => {
        if (d.error && d.error !== 'not_configured') setError(d.error)
        else if (!d.error) { setData(d); setError(null); setLastUpdate(new Date()) }
        setLoading(false); setRefreshing(false)
      })
      .catch(e => { setError(e.message); setLoading(false); setRefreshing(false) })
  }, [])

  useEffect(() => {
    fetch('/api/portfolio/status')
      .then(r => r.json())
      .then(d => { setConfigured(d.configured); if (d.configured) fetchPortfolio(); else setLoading(false) })
      .catch(() => { setConfigured(false); setLoading(false) })
    return () => { if (intervalRef.current) clearInterval(intervalRef.current) }
  }, [])

  useEffect(() => {
    if (!configured) return
    intervalRef.current = setInterval(() => fetchPortfolio(false), POLL_INTERVAL)
    return () => clearInterval(intervalRef.current)
  }, [configured, fetchPortfolio])

  // Fetch PNL whenever period changes or data refreshes
  useEffect(() => {
    if (!configured) return
    setPnlLoading(true)
    fetch(`/api/portfolio/pnl?period=${pnlPeriod}`)
      .then(r => r.json())
      .then(d => { if (!d.error) setPnlData(d); setPnlLoading(false) })
      .catch(() => setPnlLoading(false))
  }, [pnlPeriod, configured, lastUpdate])

  const incomeStats = useMemo(() => {
    if (!data?.income) return null
    let realized = 0, funding = 0, commission = 0
    for (const i of data.income) {
      if (i.type === 'REALIZED_PNL') realized += i.amount
      else if (i.type === 'FUNDING_FEE') funding += i.amount
      else if (i.type === 'COMMISSION') commission += i.amount
    }
    return { realized, funding, commission }
  }, [data])

  const fmt = (v) => v == null ? '$0.00' : `${v < 0 ? '-' : ''}$${Math.abs(v).toFixed(2)}`
  const fmtSign = (v, d = 2) => v == null ? '$0.00' : `${v >= 0 ? '+' : '-'}$${Math.abs(v).toFixed(d)}`
  const fmtTime = (ms) => { if (!ms) return '---'; const d = new Date(ms); return d.toLocaleString('en-GB', { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit', hour12: false }) }

  // --- States ---
  if (configured === false) return (
    <div className="page-content pf-page">
      <div className="pf-card pf-center-msg">
        <div className="pf-center-icon">&#x1F511;</div>
        <h3>Connect Binance Account</h3>
        <p>Set <code>BINANCE_API_KEY</code> and <code>BINANCE_API_SECRET</code> in <code>.env</code> and restart.</p>
      </div>
    </div>
  )

  if (loading || configured === null) return (
    <div className="page-content pf-page">
      <div className="pf-card pf-center-msg">
        <div className="loading-spinner" />
        <p style={{ marginTop: 12, color: 'var(--text-muted)' }}>Loading portfolio...</p>
      </div>
    </div>
  )

  if (error && !data) return (
    <div className="page-content pf-page">
      <div className="pf-card pf-center-msg">
        <h3 style={{ color: 'var(--accent-red)' }}>Connection Error</h3>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>{error}</p>
        <button className="pf-refresh-btn" style={{ marginTop: 12 }} onClick={() => fetchPortfolio(true)}>Retry</button>
      </div>
    </div>
  )

  if (!data) return null
  const { summary, balances, positions, orders, income } = data
  const equity = summary.total_wallet_balance + summary.total_unrealized_pnl

  return (
    <div className="page-content pf-page">

      {/* ===== HEADER ROW ===== */}
      <div className="pf-header-row">
        <div className="pf-header-left">
          <h2 className="pf-page-title">Portfolio</h2>
          <div className="pf-live-indicator">
            <span className="pf-dot" />
            <span className="pf-timestamp">{lastUpdate ? lastUpdate.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false }) : '---'}</span>
          </div>
          {error && <span className="pf-sync-error">Sync failed</span>}
        </div>
        <button className={`pf-refresh-btn ${refreshing ? 'pf-spinning' : ''}`} onClick={() => fetchPortfolio(true)} disabled={refreshing}>
          &#x21BB; {refreshing ? 'Refreshing...' : 'Refresh'}
        </button>
      </div>

      {/* ===== EQUITY HERO ===== */}
      <div className="pf-card pf-equity-hero">
        <div className="pf-equity-main">
          <div className="pf-equity-label">Total Equity</div>
          <div className="pf-equity-value">{fmt(equity)}</div>
          <div className={`pf-equity-change ${summary.total_unrealized_pnl >= 0 ? 'pf-green' : 'pf-red'}`}>
            {fmtSign(summary.total_unrealized_pnl)} unrealized
          </div>
        </div>
        <div className="pf-equity-stats">
          <div className="pf-mini-stat">
            <div className="pf-mini-label">Wallet</div>
            <div className="pf-mini-value">{fmt(summary.total_wallet_balance)}</div>
          </div>
          <div className="pf-mini-stat">
            <div className="pf-mini-label">Margin Used</div>
            <div className="pf-mini-value">{fmt(summary.total_position_margin)}</div>
          </div>
          <div className="pf-mini-stat">
            <div className="pf-mini-label">Available</div>
            <div className="pf-mini-value">{fmt(summary.available_balance)}</div>
          </div>
          {incomeStats && (
            <div className="pf-mini-stat">
              <div className="pf-mini-label">Realized (recent)</div>
              <div className={`pf-mini-value ${incomeStats.realized >= 0 ? 'pf-green' : 'pf-red'}`}>{fmtSign(incomeStats.realized)}</div>
            </div>
          )}
        </div>
      </div>

      {/* ===== PNL BREAKDOWN ===== */}
      <div className="pf-card">
        <div className="pf-card-title">
          PNL Breakdown
          <div className="pf-period-btns">
            {['week', 'month', 'all'].map(p => (
              <button key={p} className={`pf-period-btn ${pnlPeriod === p ? 'pf-period-active' : ''}`} onClick={() => setPnlPeriod(p)}>
                {p === 'all' ? '90 Days' : p === 'week' ? '7 Days' : '30 Days'}
              </button>
            ))}
          </div>
        </div>
        {pnlLoading && !pnlData ? (
          <div className="pf-empty">Loading...</div>
        ) : pnlData ? (
          <div className="pf-pnl-grid">
            <div className="pf-pnl-item pf-pnl-net">
              <div className="pf-pnl-label">Net PNL</div>
              <div className={`pf-pnl-value ${pnlData.net >= 0 ? 'pf-green' : 'pf-red'}`}>{fmtSign(pnlData.net)}</div>
            </div>
            <div className="pf-pnl-item">
              <div className="pf-pnl-label">Realized</div>
              <div className={`pf-pnl-value ${pnlData.realized >= 0 ? 'pf-green' : 'pf-red'}`}>{fmtSign(pnlData.realized)}</div>
            </div>
            <div className="pf-pnl-item">
              <div className="pf-pnl-label">Funding</div>
              <div className={`pf-pnl-value ${pnlData.funding >= 0 ? 'pf-green' : 'pf-red'}`}>{fmtSign(pnlData.funding)}</div>
            </div>
            <div className="pf-pnl-item">
              <div className="pf-pnl-label">Commission</div>
              <div className="pf-pnl-value pf-red">{fmt(Math.abs(pnlData.commission))}</div>
            </div>
            <div className="pf-pnl-item">
              <div className="pf-pnl-label">Trades</div>
              <div className="pf-pnl-value">{pnlData.trade_count}</div>
            </div>
          </div>
        ) : (
          <div className="pf-empty">No data</div>
        )}
      </div>

      {/* ===== OPEN POSITIONS ===== */}
      <div className="pf-card">
        <div className="pf-card-title">Open Positions <span className="pf-count">{positions.length}</span></div>
        {positions.length === 0 ? (
          <div className="pf-empty">No open positions</div>
        ) : (
          <div className="pf-pos-list">
            {positions.map((p, i) => {
              const roe = p.entry_price > 0 ? ((p.mark_price - p.entry_price) / p.entry_price * 100 * (p.side === 'LONG' ? 1 : -1)) : 0
              const isUp = p.unrealized_pnl >= 0
              return (
                <div key={i} className={`pf-pos-row ${isUp ? 'pf-pos-win' : 'pf-pos-loss'}`}>
                  <div className="pf-pos-left">
                    <span className={`pf-pos-side-tag ${p.side.toLowerCase()}`}>{p.side}</span>
                    <span className="pf-pos-symbol">{p.symbol}</span>
                    <span className="pf-pos-lev">{p.leverage}x</span>
                  </div>
                  <div className="pf-pos-mid">
                    <span className="pf-pos-info">Size: <b>{p.size}</b></span>
                    <span className="pf-pos-info">Entry: <b>${p.entry_price.toFixed(2)}</b></span>
                    <span className="pf-pos-info">Mark: <b>${p.mark_price.toFixed(2)}</b></span>
                  </div>
                  <div className="pf-pos-right">
                    <span className={`pf-pos-pnl ${isUp ? 'pf-green' : 'pf-red'}`}>{fmtSign(p.unrealized_pnl)}</span>
                    <span className={`pf-pos-roe ${isUp ? 'pf-green' : 'pf-red'}`}>{roe >= 0 ? '+' : ''}{roe.toFixed(2)}%</span>
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </div>

      {/* ===== BALANCES ===== */}
      {balances.length > 0 && (
        <div className="pf-card">
          <div className="pf-card-title">Balances</div>
          <div className="pf-balance-list">
            {balances.map((b, i) => (
              <div key={i} className="pf-bal-item">
                <span className="pf-bal-asset">{b.asset}</span>
                <span className="pf-bal-amount">{b.balance.toFixed(4)}</span>
                <span className="pf-bal-avail">Avail: {b.available.toFixed(4)}</span>
                {b.cross_pnl !== 0 && <span className={`pf-bal-pnl ${b.cross_pnl >= 0 ? 'pf-green' : 'pf-red'}`}>{fmtSign(b.cross_pnl)}</span>}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ===== ORDERS & INCOME TABS ===== */}
      <div className="pf-card pf-card-tabs">
        <div className="pf-tabs">
          <button className={`pf-tab-btn ${activeTab === 'orders' ? 'pf-tab-active' : ''}`} onClick={() => setActiveTab('orders')}>
            Orders ({orders.length})
          </button>
          <button className={`pf-tab-btn ${activeTab === 'income' ? 'pf-tab-active' : ''}`} onClick={() => setActiveTab('income')}>
            Income ({income.length})
          </button>
        </div>

        <div className="pf-tab-content">
          {activeTab === 'orders' && (
            orders.length === 0 ? <div className="pf-empty">No orders</div> : (
              <div className="pf-table-scroll">
                <table className="pf-table">
                  <thead><tr>
                    <th>Time</th><th>Pair</th><th>Side</th><th>Type</th><th>Price</th><th>Qty</th><th>Filled</th><th>Status</th>
                  </tr></thead>
                  <tbody>
                    {orders.map((o, i) => (
                      <tr key={o.order_id || i}>
                        <td className="pf-cell-dim">{fmtTime(o.time)}</td>
                        <td className="pf-cell-bold">{o.symbol}</td>
                        <td><span className={`pf-pill ${o.side === 'BUY' ? 'pf-pill-green' : 'pf-pill-red'}`}>{o.side}</span></td>
                        <td className="pf-cell-dim">{o.type}</td>
                        <td className="pf-cell-mono">{o.price > 0 ? `$${o.price.toFixed(2)}` : 'MKT'}</td>
                        <td className="pf-cell-mono">{o.qty}</td>
                        <td className="pf-cell-mono">{o.filled_qty}</td>
                        <td><span className="pf-pill" style={{ color: STATUS_COLORS[o.status] || '#5c6675' }}>{o.status}</span></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )
          )}

          {activeTab === 'income' && (
            income.length === 0 ? <div className="pf-empty">No income data</div> : (
              <div className="pf-table-scroll">
                <table className="pf-table">
                  <thead><tr>
                    <th>Time</th><th>Type</th><th>Symbol</th><th>Amount</th><th>Asset</th>
                  </tr></thead>
                  <tbody>
                    {income.map((inc, i) => (
                      <tr key={i}>
                        <td className="pf-cell-dim">{fmtTime(inc.time)}</td>
                        <td className="pf-cell-type">{INCOME_LABELS[inc.type] || inc.type}</td>
                        <td className="pf-cell-dim">{inc.symbol || '---'}</td>
                        <td className={`pf-cell-mono ${inc.amount >= 0 ? 'pf-green' : 'pf-red'}`}>{fmtSign(inc.amount, 4)}</td>
                        <td className="pf-cell-dim">{inc.asset}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )
          )}
        </div>
      </div>
    </div>
  )
}

export default PortfolioPage
