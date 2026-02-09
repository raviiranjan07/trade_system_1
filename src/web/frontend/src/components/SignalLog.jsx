import React from 'react'

function SignalLog({ signals, fullHeight }) {
  const formatTime = (isoString) => {
    if (!isoString) return '---'
    const date = new Date(isoString)
    return date.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      hour12: false
    })
  }

  const getSignalIcon = (signal) => {
    if (signal.action === 'ENTRY' || signal.action === 'REENTRY') {
      return signal.direction === 'LONG' ? '↑' : '↓'
    }
    return '○'
  }

  const getSignalTitle = (signal) => {
    if (signal.action === 'REENTRY') {
      return `${signal.direction} Re-entry`
    }
    if (signal.action === 'ENTRY') {
      return `${signal.direction} Entry`
    }
    return 'Signal Skipped'
  }

  const getSignalDetails = (signal) => {
    if (signal.action === 'ENTRY' || signal.action === 'REENTRY') {
      const parts = [`RSI: ${signal.rsi?.toFixed(1) || '---'}`]
      if (signal.atr_ok !== null && signal.atr_ok !== undefined) {
        parts.push(`ATR: ${signal.atr_ok ? 'PASS' : 'FAIL'}`)
      }
      if (signal.ema_ok !== null && signal.ema_ok !== undefined) {
        parts.push(`EMA: ${signal.ema_ok ? 'PASS' : 'FAIL'}`)
      }
      return parts.join(' | ')
    }
    return signal.reason || 'No trade signal'
  }

  const getIconClass = (signal) => {
    if (signal.action === 'REENTRY') return 'reentry'
    if (signal.action === 'ENTRY') return 'trade'
    return 'skip'
  }

  return (
    <div className="card signals-section">
      <div className="card-header">Signal Log</div>
      {!signals || signals.length === 0 ? (
        <div className="signals-empty">No signals yet</div>
      ) : (
        <div className={`signals-list ${fullHeight ? 'full-height' : ''}`}>
          {signals.map((signal, index) => (
            <div key={signal.id || index} className="signal-item">
              <div className={`signal-icon ${getIconClass(signal)}`}>
                {getSignalIcon(signal)}
              </div>
              <div className="signal-content">
                <div className={`signal-title ${signal.action !== 'SKIP' ? 'trade' : ''}`}>
                  {getSignalTitle(signal)}
                </div>
                <div className="signal-details">
                  {getSignalDetails(signal)}
                </div>
              </div>
              <div className="signal-time">
                {formatTime(signal.time)}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default SignalLog
