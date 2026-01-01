import React from 'react'

function SignalLog({ signals }) {
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
    if (signal.action === 'TRADE') {
      return signal.side === 'LONG' ? '↑' : '↓'
    }
    return '○'
  }

  const getSignalTitle = (signal) => {
    if (signal.action === 'TRADE') {
      return `${signal.side} Trade Opened`
    }
    return `Signal Skipped`
  }

  const getSignalDetails = (signal) => {
    if (signal.action === 'TRADE') {
      return `Exp: ${(signal.expectancy * 100).toFixed(2)}% | Dist: ${signal.distance?.toFixed(2) || '---'}`
    }
    return signal.reason || 'No trade signal'
  }

  return (
    <div className="card signals-section">
      <div className="card-header">Signal Log</div>
      {!signals || signals.length === 0 ? (
        <div className="signals-empty">No signals yet</div>
      ) : (
        <div className="signals-list">
          {signals.map((signal, index) => (
            <div key={signal.id || index} className="signal-item">
              <div className={`signal-icon ${signal.action === 'TRADE' ? 'trade' : 'skip'}`}>
                {getSignalIcon(signal)}
              </div>
              <div className="signal-content">
                <div className={`signal-title ${signal.action === 'TRADE' ? 'trade' : ''}`}>
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
