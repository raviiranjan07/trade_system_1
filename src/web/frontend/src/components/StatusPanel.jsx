import React from 'react'

function StatusPanel({ label, value, format, panelType }) {
  const formatValue = () => {
    if (value === null || value === undefined) return '---'

    switch (format) {
      case 'price':
        return `$${Number(value).toLocaleString('en-US', {
          minimumFractionDigits: 2,
          maximumFractionDigits: 2
        })}`
      case 'rsi':
        return Number(value).toFixed(1)
      case 'percentile':
        return `${Number(value).toFixed(1)}%`
      case 'ema':
        return `${Number(value).toFixed(2)}%`
      case 'number':
        return Number(value).toLocaleString('en-US')
      default:
        return value
    }
  }

  const getPanelClass = () => {
    switch (panelType) {
      case 'price': return 'price-panel'
      case 'rsi': return 'rsi-panel'
      case 'atr': return 'atr-panel'
      case 'ema': return 'ema-panel'
      case 'regime': return 'regime-panel'
      case 'bars': return 'bars-panel'
      default: return ''
    }
  }

  // RSI zone coloring
  const getRsiZone = () => {
    if (format !== 'rsi' || value === null || value === undefined) return ''
    const rsi = Number(value)
    if (rsi <= 20) return 'rsi-oversold'
    if (rsi >= 80) return 'rsi-overbought'
    return ''
  }

  return (
    <div className={`card status-panel ${getPanelClass()} ${getRsiZone()}`}>
      <div className="card-header">{label}</div>
      <div className="card-value">{formatValue()}</div>
    </div>
  )
}

export default StatusPanel
