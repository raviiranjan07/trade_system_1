import React from 'react'

function SignalStats({ signals }) {
  const entries = signals ? signals.filter(s => s.action === 'ENTRY').length : 0
  const reentries = signals ? signals.filter(s => s.action === 'REENTRY').length : 0
  const skips = signals ? signals.filter(s => s.action === 'SKIP').length : 0
  const total = entries + reentries + skips

  const entryRate = total > 0 ? ((entries + reentries) / total * 100).toFixed(1) : '0.0'

  return (
    <div className="signal-stats-grid">
      <div className="card signal-stat-card">
        <div className="card-header">Entries</div>
        <div className="card-value positive">{entries}</div>
      </div>
      <div className="card signal-stat-card">
        <div className="card-header">Re-entries</div>
        <div className="card-value" style={{ color: '#f59e0b' }}>{reentries}</div>
      </div>
      <div className="card signal-stat-card">
        <div className="card-header">Skipped</div>
        <div className="card-value" style={{ color: '#8892a0' }}>{skips}</div>
      </div>
    </div>
  )
}

export default SignalStats
