import React from 'react'

const TOOL_GROUPS = [
  // Pointer
  [{ id: 'pointer', label: 'Pointer', icon: (
    <svg viewBox="0 0 24 24" width="18" height="18"><path d="M4 2l14 9-7 1.5-3 7z" fill="currentColor" opacity="0.9"/></svg>
  )}],
  // Lines
  [
    { id: 'hline', label: 'Horizontal Line', icon: (
      <svg viewBox="0 0 24 24" width="18" height="18"><line x1="2" y1="12" x2="22" y2="12" stroke="currentColor" strokeWidth="2"/></svg>
    )},
    { id: 'vline', label: 'Vertical Line', icon: (
      <svg viewBox="0 0 24 24" width="18" height="18"><line x1="12" y1="2" x2="12" y2="22" stroke="currentColor" strokeWidth="2"/></svg>
    )},
    { id: 'trendline', label: 'Trend Line', icon: (
      <svg viewBox="0 0 24 24" width="18" height="18"><line x1="3" y1="19" x2="21" y2="5" stroke="currentColor" strokeWidth="2"/><circle cx="3" cy="19" r="2" fill="currentColor"/><circle cx="21" cy="5" r="2" fill="currentColor"/></svg>
    )},
    { id: 'ray', label: 'Ray', icon: (
      <svg viewBox="0 0 24 24" width="18" height="18"><line x1="3" y1="18" x2="21" y2="6" stroke="currentColor" strokeWidth="2"/><circle cx="3" cy="18" r="2" fill="currentColor"/><polygon points="21,6 17,5 18,9" fill="currentColor"/></svg>
    )},
    { id: 'extended', label: 'Extended Line', icon: (
      <svg viewBox="0 0 24 24" width="18" height="18"><line x1="1" y1="20" x2="23" y2="4" stroke="currentColor" strokeWidth="1.5" strokeDasharray="3 2"/><line x1="6" y1="17" x2="18" y2="7" stroke="currentColor" strokeWidth="2"/><circle cx="6" cy="17" r="2" fill="currentColor"/><circle cx="18" cy="7" r="2" fill="currentColor"/></svg>
    )},
  ],
  // Shapes
  [
    { id: 'rect', label: 'Rectangle', icon: (
      <svg viewBox="0 0 24 24" width="18" height="18"><rect x="3" y="5" width="18" height="14" fill="none" stroke="currentColor" strokeWidth="2" rx="1"/></svg>
    )},
    { id: 'channel', label: 'Parallel Channel', icon: (
      <svg viewBox="0 0 24 24" width="18" height="18"><line x1="2" y1="16" x2="22" y2="10" stroke="currentColor" strokeWidth="2"/><line x1="2" y1="10" x2="22" y2="4" stroke="currentColor" strokeWidth="1.5" strokeDasharray="3 2"/></svg>
    )},
  ],
  // Fibonacci
  [{ id: 'fib', label: 'Fibonacci Retracement', icon: (
    <svg viewBox="0 0 24 24" width="18" height="18"><line x1="2" y1="4" x2="22" y2="4" stroke="currentColor" strokeWidth="1.5" opacity="0.5"/><line x1="2" y1="8" x2="22" y2="8" stroke="currentColor" strokeWidth="1" opacity="0.6"/><line x1="2" y1="12" x2="22" y2="12" stroke="currentColor" strokeWidth="2"/><line x1="2" y1="16" x2="22" y2="16" stroke="currentColor" strokeWidth="1" opacity="0.6"/><line x1="2" y1="20" x2="22" y2="20" stroke="currentColor" strokeWidth="1.5" opacity="0.5"/></svg>
  )}],
  // Measure + Text
  [
    { id: 'measure', label: 'Measure', icon: (
      <svg viewBox="0 0 24 24" width="18" height="18"><rect x="4" y="5" width="16" height="14" fill="none" stroke="currentColor" strokeWidth="1.5" strokeDasharray="3 2"/><line x1="12" y1="7" x2="12" y2="17" stroke="currentColor" strokeWidth="2"/><line x1="9" y1="7" x2="15" y2="7" stroke="currentColor" strokeWidth="1.5"/><line x1="9" y1="17" x2="15" y2="17" stroke="currentColor" strokeWidth="1.5"/></svg>
    )},
    { id: 'text', label: 'Text', icon: (
      <svg viewBox="0 0 24 24" width="18" height="18"><text x="6" y="19" fill="currentColor" fontSize="18" fontWeight="bold" fontFamily="sans-serif">T</text></svg>
    )},
  ],
  // Eraser
  [{ id: 'eraser', label: 'Eraser', icon: (
    <svg viewBox="0 0 24 24" width="18" height="18"><line x1="5" y1="5" x2="19" y2="19" stroke="currentColor" strokeWidth="2"/><line x1="19" y1="5" x2="5" y2="19" stroke="currentColor" strokeWidth="2"/></svg>
  )}],
]

function DrawingToolbar({ activeTool, onToolChange, onClearAll }) {
  return (
    <div className="drawing-toolbar">
      {TOOL_GROUPS.map((group, gi) => (
        <React.Fragment key={gi}>
          {gi > 0 && <div className="drawing-toolbar-divider" />}
          {group.map(tool => (
            <button
              key={tool.id}
              className={`drawing-tool-btn ${activeTool === tool.id ? 'active' : ''}`}
              onClick={() => onToolChange(tool.id)}
              title={tool.label}
            >
              {tool.icon}
            </button>
          ))}
        </React.Fragment>
      ))}
      <div className="drawing-toolbar-divider" />
      <button
        className="drawing-tool-btn clear-btn"
        onClick={onClearAll}
        title="Clear All Drawings"
      >
        <svg viewBox="0 0 24 24" width="18" height="18">
          <path d="M7 4v1H4v2h16V5h-3V4H7z" fill="currentColor" opacity="0.8"/>
          <path d="M5 8l1 12h12l1-12H5zm4 2v8h2v-8H9zm4 0v8h2v-8h-2z" fill="currentColor" opacity="0.8"/>
        </svg>
      </button>
    </div>
  )
}

export default DrawingToolbar
