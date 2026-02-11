import React, { useState, useEffect, useRef } from 'react'

const MA_TYPES = [
  { value: 'SMA', label: 'SMA (Simple Moving Average)' },
  { value: 'EMA', label: 'EMA (Exponential Moving Average)' },
  { value: 'WMA', label: 'WMA (Weighted Moving Average)' },
  { value: 'DEMA', label: 'DEMA (Double EMA)' },
]

const DEFAULT_COLORS = [
  '#3b82f6', '#f59e0b', '#ef4444', '#10b981', '#8b5cf6',
  '#06b6d4', '#f97316', '#ec4899', '#14b8a6', '#a855f7',
]

const LINE_WIDTHS = [1, 2, 3, 4]

const SOURCES = [
  { value: 'close', label: 'Close' },
  { value: 'open', label: 'Open' },
  { value: 'high', label: 'High' },
  { value: 'low', label: 'Low' },
  { value: 'hl2', label: 'HL2 (High+Low)/2' },
  { value: 'hlc3', label: 'HLC3 (High+Low+Close)/3' },
  { value: 'ohlc4', label: 'OHLC4' },
]

function IndicatorPanel({ indicators, onAdd, onRemove, onUpdate, onToggle }) {
  const [open, setOpen] = useState(false)
  const [adding, setAdding] = useState(false)
  const [editing, setEditing] = useState(null) // indicator id being edited
  const ref = useRef(null)

  // Form state for adding/editing
  const [formType, setFormType] = useState('SMA')
  const [formPeriod, setFormPeriod] = useState(50)
  const [formColor, setFormColor] = useState('#f59e0b')
  const [formWidth, setFormWidth] = useState(2)
  const [formSource, setFormSource] = useState('close')

  // Close on outside click
  useEffect(() => {
    if (!open) return
    const handler = (e) => {
      if (ref.current && !ref.current.contains(e.target)) {
        setOpen(false)
        setAdding(false)
        setEditing(null)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [open])

  const resetForm = () => {
    setFormType('SMA')
    setFormPeriod(50)
    setFormColor(DEFAULT_COLORS[indicators.length % DEFAULT_COLORS.length])
    setFormWidth(2)
    setFormSource('close')
  }

  const handleAdd = () => {
    const period = Math.max(1, Math.min(500, parseInt(formPeriod) || 50))
    onAdd({
      id: Date.now() + Math.random(),
      type: formType,
      period,
      color: formColor,
      lineWidth: formWidth,
      source: formSource,
      visible: true,
    })
    setAdding(false)
    resetForm()
  }

  const startEdit = (ind) => {
    setEditing(ind.id)
    setFormType(ind.type)
    setFormPeriod(ind.period)
    setFormColor(ind.color)
    setFormWidth(ind.lineWidth)
    setFormSource(ind.source || 'close')
    setAdding(false)
  }

  const handleSaveEdit = () => {
    const period = Math.max(1, Math.min(500, parseInt(formPeriod) || 50))
    onUpdate(editing, {
      type: formType,
      period,
      color: formColor,
      lineWidth: formWidth,
      source: formSource,
    })
    setEditing(null)
    resetForm()
  }

  const handleStartAdd = () => {
    setEditing(null)
    resetForm()
    setAdding(true)
  }

  const renderForm = (isEdit) => (
    <div className="indicator-form">
      <div className="indicator-form-row">
        <label>Type</label>
        <select value={formType} onChange={e => setFormType(e.target.value)}>
          {MA_TYPES.map(t => <option key={t.value} value={t.value}>{t.value}</option>)}
        </select>
      </div>
      <div className="indicator-form-row">
        <label>Period</label>
        <input
          type="number"
          min="1"
          max="500"
          value={formPeriod}
          onChange={e => setFormPeriod(e.target.value)}
        />
      </div>
      <div className="indicator-form-row">
        <label>Source</label>
        <select value={formSource} onChange={e => setFormSource(e.target.value)}>
          {SOURCES.map(s => <option key={s.value} value={s.value}>{s.label}</option>)}
        </select>
      </div>
      <div className="indicator-form-row">
        <label>Color</label>
        <div className="indicator-color-row">
          {DEFAULT_COLORS.map(c => (
            <button
              key={c}
              className={`indicator-color-swatch ${formColor === c ? 'active' : ''}`}
              style={{ background: c }}
              onClick={() => setFormColor(c)}
            />
          ))}
          <input
            type="color"
            value={formColor}
            onChange={e => setFormColor(e.target.value)}
            className="indicator-color-input"
          />
        </div>
      </div>
      <div className="indicator-form-row">
        <label>Width</label>
        <div className="indicator-width-row">
          {LINE_WIDTHS.map(w => (
            <button
              key={w}
              className={`indicator-width-btn ${formWidth === w ? 'active' : ''}`}
              onClick={() => setFormWidth(w)}
            >
              <div className="indicator-width-preview" style={{ height: w }} />
            </button>
          ))}
        </div>
      </div>
      <div className="indicator-form-actions">
        <button className="indicator-btn-cancel" onClick={() => { setAdding(false); setEditing(null) }}>
          Cancel
        </button>
        <button className="indicator-btn-apply" onClick={isEdit ? handleSaveEdit : handleAdd}>
          {isEdit ? 'Save' : 'Add'}
        </button>
      </div>
    </div>
  )

  return (
    <div className="indicator-panel" ref={ref}>
      <button
        className={`indicator-panel-btn ${open ? 'active' : ''}`}
        onClick={() => { setOpen(v => !v); setAdding(false); setEditing(null) }}
        title="Indicators"
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
          <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
        </svg>
        <span className="indicator-panel-label">Indicators</span>
        {indicators.length > 0 && (
          <span className="indicator-badge">{indicators.length}</span>
        )}
      </button>

      {open && (
        <div className="indicator-panel-popup">
          <div className="indicator-panel-header">
            <span>Indicators</span>
            <button className="indicator-add-btn" onClick={handleStartAdd} title="Add Indicator">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" />
              </svg>
            </button>
          </div>

          {/* Active indicators list */}
          <div className="indicator-list">
            {indicators.length === 0 && !adding && (
              <div className="indicator-empty">No indicators added</div>
            )}
            {indicators.map(ind => (
              <div key={ind.id} className={`indicator-item ${!ind.visible ? 'hidden-ind' : ''}`}>
                {editing === ind.id ? (
                  renderForm(true)
                ) : (
                  <>
                    <button
                      className="indicator-eye-btn"
                      onClick={() => onToggle(ind.id)}
                      title={ind.visible ? 'Hide' : 'Show'}
                    >
                      {ind.visible ? (
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
                          <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
                          <circle cx="12" cy="12" r="3" />
                        </svg>
                      ) : (
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
                          <path d="M17.94 17.94A10.07 10.07 0 0112 20c-7 0-11-8-11-8a18.45 18.45 0 015.06-5.94M9.9 4.24A9.12 9.12 0 0112 4c7 0 11 8 11 8a18.5 18.5 0 01-2.16 3.19m-6.72-1.07a3 3 0 11-4.24-4.24" />
                          <line x1="1" y1="1" x2="23" y2="23" />
                        </svg>
                      )}
                    </button>
                    <span className="indicator-dot" style={{ background: ind.color }} />
                    <span className="indicator-name">{ind.type} {ind.period}</span>
                    <span className="indicator-source">{ind.source !== 'close' ? ind.source : ''}</span>
                    <button className="indicator-edit-btn" onClick={() => startEdit(ind)} title="Settings">
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
                        <circle cx="12" cy="12" r="3" />
                        <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
                      </svg>
                    </button>
                    <button className="indicator-remove-btn" onClick={() => onRemove(ind.id)} title="Remove">
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
                      </svg>
                    </button>
                  </>
                )}
              </div>
            ))}
          </div>

          {/* Add form */}
          {adding && renderForm(false)}
        </div>
      )}
    </div>
  )
}

export default IndicatorPanel
