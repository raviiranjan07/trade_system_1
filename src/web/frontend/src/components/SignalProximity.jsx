import React from 'react'

const SIGNAL_ORDER = ['V12_LONG', 'V12_SHORT', 'BEAR_LONG', 'BULL_SHORT']

const SIGNAL_COLORS = {
  V12_LONG: '#00d97e',
  V12_SHORT: '#e63757',
  BEAR_LONG: '#3b82f6',
  BULL_SHORT: '#f59e0b',
}

const SIGNAL_LABELS = {
  V12_LONG: 'V12 LONG',
  V12_SHORT: 'V12 SHORT',
  BEAR_LONG: 'BEAR LONG',
  BULL_SHORT: 'BULL SHORT',
}

function getRsiProgress(cond) {
  // How close is RSI to the threshold (0-100%)
  const { current, threshold, op } = cond
  if (cond.met) return 100

  if (op === '<') {
    // RSI needs to go DOWN from current to threshold
    // At 50 (neutral) = 0%, at threshold = 100%
    const neutral = 50
    if (current >= neutral) return 0
    return Math.round(Math.max(0, (1 - (current - threshold) / (neutral - threshold)) * 100))
  } else {
    // RSI needs to go UP from current to threshold
    const neutral = 50
    if (current <= neutral) return 0
    return Math.round(Math.max(0, (1 - (threshold - current) / (threshold - neutral)) * 100))
  }
}

function SignalProximity({ proximity }) {
  if (!proximity || Object.keys(proximity).length === 0) return null

  return (
    <div className="signal-proximity">
      <div className="proximity-header">Signal Proximity</div>
      {SIGNAL_ORDER.map(key => {
        const sig = proximity[key]
        if (!sig) return null

        const conditions = sig.conditions || []
        const rsiCond = conditions.find(c => c.name === 'RSI')
        const otherConds = conditions.filter(c => c.name !== 'RSI')
        const allMet = conditions.every(c => c.met)
        const rsiProgress = rsiCond ? getRsiProgress(rsiCond) : 0
        const color = SIGNAL_COLORS[key]

        return (
          <div key={key} className={`proximity-row ${allMet ? 'all-met' : ''}`}>
            <div className="proximity-label" style={{ color }}>
              <span className="proximity-name">{SIGNAL_LABELS[key]}</span>
              {rsiCond && (
                <span className="proximity-rsi-info">
                  RSI {rsiCond.current} {rsiCond.op} {rsiCond.threshold}
                </span>
              )}
            </div>
            <div className="proximity-bar-track">
              <div
                className="proximity-bar-fill"
                style={{
                  width: `${rsiProgress}%`,
                  backgroundColor: allMet ? color : `${color}88`,
                }}
              />
            </div>
            <div className="proximity-chips">
              {conditions.map((cond, i) => (
                <span
                  key={i}
                  className={`proximity-chip ${cond.met ? 'met' : 'unmet'}`}
                  title={cond.current !== undefined
                    ? `${cond.name}: ${cond.current} ${cond.op || ''} ${cond.threshold || ''}`
                    : `${cond.name}: ${cond.needed || ''}`}
                >
                  {cond.met ? '\u2713' : '\u2717'}
                  {' '}
                  {cond.name === 'RSI' ? `RSI${cond.op}${cond.threshold}`
                    : cond.name === 'Regime' ? cond.needed
                    : `${cond.name}${cond.op}${cond.threshold}${cond.name === 'EMA' ? '%' : ''}`}
                </span>
              ))}
            </div>
          </div>
        )
      })}
    </div>
  )
}

export default SignalProximity
