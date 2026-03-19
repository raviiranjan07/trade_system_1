import React from 'react'

function MLPanel({ ml, mlTrades }) {
  if (!ml) return null

  const totalBps = ml.ml_total_bps || 0
  const bpsClass = totalBps >= 0 ? 'positive' : 'negative'
  const winRate = ((ml.ml_win_rate || 0) * 100).toFixed(1)

  return (
    <div className="ml-panel">
      <div className="ml-panel-header">
        <h3>ML Model {ml.ml_model_loaded ? '(Active)' : '(Inactive)'}</h3>
        <div className={`ml-wallet ${ml.ml_wallet_usd > 5 ? 'positive' : 'negative'}`}>
          ${(ml.ml_wallet_usd || 0).toFixed(2)}
        </div>
      </div>

      <div className="ml-stats-row">
        <div className="ml-stat">
          <span className={`ml-stat-value ${bpsClass}`}>
            {totalBps >= 0 ? '+' : ''}{totalBps.toFixed(1)}
          </span>
          <span className="ml-stat-label">bps</span>
        </div>
        <div className="ml-stat">
          <span className="ml-stat-value">{ml.ml_total_trades || 0}</span>
          <span className="ml-stat-label">trades</span>
        </div>
        <div className="ml-stat">
          <span className="ml-stat-value">{winRate}%</span>
          <span className="ml-stat-label">win rate</span>
        </div>
        <div className="ml-stat">
          <span className="ml-stat-value">
            {ml.ml_avg_bps >= 0 ? '+' : ''}{(ml.ml_avg_bps || 0).toFixed(1)}
          </span>
          <span className="ml-stat-label">avg/trade</span>
        </div>
      </div>

      {ml.ml_has_position && (
        <div className="ml-position-badge">
          IN POSITION: {ml.ml_position_side || '?'}
        </div>
      )}

      {mlTrades && mlTrades.length > 0 && (
        <div className="ml-trades-list">
          <h4>Recent ML Trades</h4>
          <table>
            <thead>
              <tr>
                <th>Dir</th>
                <th>Entry</th>
                <th>Exit</th>
                <th>P&L</th>
                <th>Exit</th>
              </tr>
            </thead>
            <tbody>
              {mlTrades.slice(0, 10).map((t, i) => (
                <tr key={i} className={t.net_profit_bps > 0 ? 'winner' : 'loser'}>
                  <td className={t.direction === 'LONG' ? 'positive' : 'negative'}>
                    {t.direction}
                  </td>
                  <td>{Number(t.entry_price).toFixed(0)}</td>
                  <td>{Number(t.exit_price).toFixed(0)}</td>
                  <td className={t.net_profit_bps > 0 ? 'positive' : 'negative'}>
                    {t.net_profit_bps > 0 ? '+' : ''}{Number(t.net_profit_bps).toFixed(1)}
                  </td>
                  <td>{t.exit_reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

export default MLPanel
