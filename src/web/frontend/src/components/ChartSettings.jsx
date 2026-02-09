import React, { useState, useEffect, useRef } from 'react'

const BG_PRESETS = [
  { value: '#0a0e14', label: 'Midnight' },
  { value: '#131722', label: 'TradingView' },
  { value: '#1e222d', label: 'Charcoal' },
  { value: '#1a1a2e', label: 'Navy' },
  { value: '#19232d', label: 'Slate' },
  { value: '#0d1117', label: 'GitHub' },
  { value: '#1b2836', label: 'Ocean' },
  { value: '#000000', label: 'Black' },
]

function ChartSettings({ chartBg, onChangeBg }) {
  const [open, setOpen] = useState(false)
  const ref = useRef(null)

  // Close on outside click
  useEffect(() => {
    if (!open) return
    const handler = (e) => {
      if (ref.current && !ref.current.contains(e.target)) setOpen(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [open])

  return (
    <div className="chart-settings" ref={ref}>
      <button
        className="chart-settings-btn"
        onClick={() => setOpen(v => !v)}
        title="Chart Settings"
      >
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="3" />
          <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
        </svg>
      </button>
      {open && (
        <div className="chart-settings-popup">
          <div className="chart-settings-section">
            <span className="chart-settings-label">Background</span>
            <div className="chart-settings-swatches">
              {BG_PRESETS.map(c => (
                <button
                  key={c.value}
                  className={`color-swatch ${chartBg === c.value ? 'active' : ''}`}
                  style={{ background: c.value }}
                  onClick={() => onChangeBg(c.value)}
                  title={c.label}
                />
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default ChartSettings
