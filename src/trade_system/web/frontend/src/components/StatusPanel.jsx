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
      case 'number':
        return Number(value).toLocaleString('en-US')
      case 'percent':
        return `${(Number(value) * 100).toFixed(1)}%`
      case 'bars':
        return `${value} bars`
      default:
        return value
    }
  }

  const getPanelClass = () => {
    switch (panelType) {
      case 'price':
        return 'price-panel'
      case 'regime':
        return 'regime-panel'
      case 'bars':
        return 'bars-panel'
      case 'next-check':
        return 'next-check-panel'
      default:
        return ''
    }
  }

  return (
    <div className={`card status-panel ${getPanelClass()}`}>
      <div className="card-header">{label}</div>
      <div className="card-value">{formatValue()}</div>
    </div>
  )
}

export default StatusPanel
