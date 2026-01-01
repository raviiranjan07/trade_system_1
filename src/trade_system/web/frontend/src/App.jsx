import React, { useState, useEffect, useRef } from 'react'
import StatusPanel from './components/StatusPanel'
import PositionCard from './components/PositionCard'
import StatsSection from './components/StatsSection'
import TradesList from './components/TradesList'
import SignalLog from './components/SignalLog'

function App() {
  const [data, setData] = useState(null)
  const [connected, setConnected] = useState(false)
  const wsRef = useRef(null)
  const reconnectTimeoutRef = useRef(null)

  const connectWebSocket = () => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const wsUrl = `${protocol}//${window.location.host}/ws`

    try {
      wsRef.current = new WebSocket(wsUrl)

      wsRef.current.onopen = () => {
        console.log('WebSocket connected')
        setConnected(true)
      }

      wsRef.current.onmessage = (event) => {
        const message = JSON.parse(event.data)
        if (message.type !== 'ping') {
          setData(message)
        }
      }

      wsRef.current.onclose = () => {
        console.log('WebSocket disconnected')
        setConnected(false)
        reconnectTimeoutRef.current = setTimeout(connectWebSocket, 3000)
      }

      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error)
        wsRef.current.close()
      }
    } catch (error) {
      console.error('Failed to connect:', error)
      reconnectTimeoutRef.current = setTimeout(connectWebSocket, 3000)
    }
  }

  useEffect(() => {
    fetch('/api/all')
      .then(res => res.json())
      .then(setData)
      .catch(console.error)

    connectWebSocket()

    return () => {
      if (wsRef.current) {
        wsRef.current.close()
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
      }
    }
  }, [])

  const formatUptime = (seconds) => {
    if (!seconds) return '0s'
    const hours = Math.floor(seconds / 3600)
    const minutes = Math.floor((seconds % 3600) / 60)
    const secs = Math.floor(seconds % 60)
    if (hours > 0) return `${hours}h ${minutes}m ${secs}s`
    if (minutes > 0) return `${minutes}m ${secs}s`
    return `${secs}s`
  }

  if (!data) {
    return (
      <div className="dashboard">
        <div className="loading">
          <div className="loading-spinner"></div>
          <span>Connecting to trading system...</span>
        </div>
      </div>
    )
  }

  const { status, position, stats, trades, signals, config } = data

  return (
    <div className="dashboard">
      {/* Header */}
      <header className="header">
        <h1>
          Paper Trading
          <span className="pair">{config?.pair || 'BTCUSDT'}</span>
        </h1>
        <div className="header-right">
          <span className="uptime">{formatUptime(status?.uptime_seconds)}</span>
          <div className="connection-status">
            <span className={`connection-dot ${connected ? 'connected' : 'disconnected'}`}></span>
            {connected ? 'Live' : 'Reconnecting...'}
          </div>
          <span className={`status-badge ${status?.status?.toLowerCase() || 'starting'}`}>
            {status?.status || 'STARTING'}
          </span>
        </div>
      </header>

      {/* Status Cards - 4 columns */}
      <div className="grid grid-4">
        <StatusPanel
          label="Price"
          value={status?.price}
          format="price"
          panelType="price"
        />
        <StatusPanel
          label="Regime"
          value={status?.regime || '---'}
          panelType="regime"
        />
        <StatusPanel
          label="Bars Processed"
          value={status?.bar_count || 0}
          format="number"
          panelType="bars"
        />
        <StatusPanel
          label="Next Check"
          value={status?.next_check_in || 0}
          format="bars"
          panelType="next-check"
        />
      </div>

      {/* Position */}
      <div style={{ marginTop: '16px' }}>
        <PositionCard position={position} />
      </div>

      {/* Stats */}
      <StatsSection stats={stats} />

      {/* Bottom Section - Trades and Signals side by side */}
      <div className="bottom-section">
        <TradesList trades={trades} />
        <SignalLog signals={signals} />
      </div>
    </div>
  )
}

export default App
