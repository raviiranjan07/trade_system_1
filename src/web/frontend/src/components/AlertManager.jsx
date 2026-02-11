import React, { useEffect, useRef, useState, useCallback } from 'react'

// Simple beep using AudioContext
const playBeep = (frequency = 880, duration = 150, volume = 0.3) => {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)()
    const osc = ctx.createOscillator()
    const gain = ctx.createGain()
    osc.connect(gain)
    gain.connect(ctx.destination)
    osc.frequency.value = frequency
    osc.type = 'sine'
    gain.gain.value = volume
    osc.start()
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + duration / 1000)
    osc.stop(ctx.currentTime + duration / 1000 + 0.05)
  } catch (e) {
    // AudioContext not available
  }
}

const ALERT_SOUNDS = {
  entry: () => { playBeep(880, 120, 0.25); setTimeout(() => playBeep(1100, 120, 0.25), 150) },
  exit: () => playBeep(660, 200, 0.2),
  signal: () => playBeep(440, 100, 0.15),
}

function AlertManager({ trades, signals, position }) {
  const [permission, setPermission] = useState(
    typeof Notification !== 'undefined' ? Notification.permission : 'denied'
  )
  const [alertsEnabled, setAlertsEnabled] = useState(() => {
    return localStorage.getItem('alertsEnabled') !== 'false'
  })
  const [soundEnabled, setSoundEnabled] = useState(() => {
    return localStorage.getItem('soundEnabled') !== 'false'
  })
  const [showMenu, setShowMenu] = useState(false)
  const menuRef = useRef(null)

  // Track previous counts to detect changes
  const prevTradeCountRef = useRef(trades?.length || 0)
  const prevSignalCountRef = useRef(signals?.length || 0)
  const prevHasPositionRef = useRef(position?.has_position || false)
  const initialLoadRef = useRef(true)

  // Persist prefs
  useEffect(() => { localStorage.setItem('alertsEnabled', alertsEnabled) }, [alertsEnabled])
  useEffect(() => { localStorage.setItem('soundEnabled', soundEnabled) }, [soundEnabled])

  // Close menu on outside click
  useEffect(() => {
    if (!showMenu) return
    const handler = (e) => {
      if (menuRef.current && !menuRef.current.contains(e.target)) setShowMenu(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [showMenu])

  const requestPermission = useCallback(async () => {
    if (typeof Notification === 'undefined') return
    const result = await Notification.requestPermission()
    setPermission(result)
  }, [])

  const sendNotification = useCallback((title, body, tag) => {
    if (!alertsEnabled) return
    if (permission === 'granted') {
      try {
        new Notification(title, { body, tag, icon: '/favicon.ico', silent: true })
      } catch (e) {}
    }
  }, [alertsEnabled, permission])

  // Detect new trades (entry/exit)
  useEffect(() => {
    if (initialLoadRef.current) {
      initialLoadRef.current = false
      prevTradeCountRef.current = trades?.length || 0
      prevHasPositionRef.current = position?.has_position || false
      return
    }

    const currentTradeCount = trades?.length || 0
    const hasPosition = position?.has_position || false
    const hadPosition = prevHasPositionRef.current

    // New trade closed (trade count increased)
    if (currentTradeCount > prevTradeCountRef.current && trades?.length > 0) {
      const latest = trades[0] // newest first
      const pnl = latest.net_profit_bps?.toFixed(1)
      const dir = latest.direction || '?'
      const reason = latest.exit_reason || ''
      const isWin = (latest.net_profit_bps || 0) > 0

      if (soundEnabled) ALERT_SOUNDS.exit()
      sendNotification(
        `Trade Closed: ${dir} ${isWin ? 'WIN' : 'LOSS'}`,
        `${isWin ? '+' : ''}${pnl} bps | ${reason}`,
        'trade-exit'
      )
    }

    // Position opened (no position -> has position)
    if (hasPosition && !hadPosition && position?.side) {
      if (soundEnabled) ALERT_SOUNDS.entry()
      sendNotification(
        `Position Opened: ${position.side}`,
        `Entry: $${position.entry_price?.toLocaleString()}`,
        'trade-entry'
      )
    }

    prevTradeCountRef.current = currentTradeCount
    prevHasPositionRef.current = hasPosition
  }, [trades, position, soundEnabled, alertsEnabled, sendNotification])

  // Detect new signals (ENTRY or REENTRY only, not SKIPs)
  useEffect(() => {
    const currentSignalCount = signals?.length || 0
    if (currentSignalCount > prevSignalCountRef.current && signals?.length > 0) {
      const latest = signals[0]
      if (latest.action === 'ENTRY' || latest.action === 'REENTRY') {
        if (soundEnabled) ALERT_SOUNDS.signal()
        sendNotification(
          `Signal: ${latest.action} ${latest.direction || ''}`,
          `RSI: ${latest.rsi?.toFixed(1)} | ${latest.reason || ''}`,
          'signal'
        )
      }
    }
    prevSignalCountRef.current = currentSignalCount
  }, [signals, soundEnabled, alertsEnabled, sendNotification])

  return (
    <div className="alert-manager" ref={menuRef}>
      <button
        className={`alert-btn ${alertsEnabled ? 'active' : ''}`}
        onClick={() => setShowMenu(v => !v)}
        title="Alert Settings"
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9" />
          <path d="M13.73 21a2 2 0 0 1-3.46 0" />
        </svg>
        {alertsEnabled && <span className="alert-dot-active" />}
      </button>

      {showMenu && (
        <div className="alert-menu">
          <div className="alert-menu-header">Notifications</div>

          <label className="alert-toggle-row">
            <span>Browser Alerts</span>
            <input
              type="checkbox"
              checked={alertsEnabled}
              onChange={(e) => {
                setAlertsEnabled(e.target.checked)
                if (e.target.checked && permission === 'default') requestPermission()
              }}
            />
          </label>

          <label className="alert-toggle-row">
            <span>Sound</span>
            <input
              type="checkbox"
              checked={soundEnabled}
              onChange={(e) => setSoundEnabled(e.target.checked)}
            />
          </label>

          {permission !== 'granted' && alertsEnabled && (
            <button className="alert-permission-btn" onClick={requestPermission}>
              Enable Browser Notifications
            </button>
          )}

          <div className="alert-menu-footer">
            Alerts on: entries, exits, signals
          </div>
        </div>
      )}
    </div>
  )
}

export default AlertManager
