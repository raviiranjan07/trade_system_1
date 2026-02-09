import React, { useState, useEffect, useRef, useCallback } from 'react'
import StatusPanel from './components/StatusPanel'
import PositionCard from './components/PositionCard'
import StatsSection from './components/StatsSection'
import TradesList from './components/TradesList'
import SignalLog from './components/SignalLog'
import TabBar from './components/TabBar'
import EquityCurve from './components/EquityCurve'
import SignalStats from './components/SignalStats'
import TradingViewChart from './components/TradingViewChart'
import CandlestickChart from './components/CandlestickChart'
import RsiPane from './components/RsiPane'
import DrawingToolbar from './components/DrawingToolbar'
import DrawingOverlay from './components/DrawingOverlay'
import ChartLegend from './components/ChartLegend'
import ChartSettings from './components/ChartSettings'
import SignalProximity from './components/SignalProximity'

// IST = UTC + 5:30 = 19800 seconds
const IST_OFFSET = 19800

const shiftIST = (c) => c ? { ...c, time: c.time + IST_OFFSET } : c
const shiftCandlesIST = (arr) => arr ? arr.map(shiftIST) : arr

function App() {
  const [data, setData] = useState(null)
  const [connected, setConnected] = useState(false)
  const [activeTab, setActiveTab] = useState(() => localStorage.getItem('activeTab') || 'trades')
  const [chartView, setChartView] = useState(() => localStorage.getItem('chartView') || 'tradingview')
  const wsRef = useRef(null)
  const reconnectTimeoutRef = useRef(null)
  const chartAreaRef = useRef(null)

  // Bot chart state
  const [chartCandles, setChartCandles] = useState(null)
  const [liveCandle, setLiveCandle] = useState(null)
  const [timeRange, setTimeRange] = useState(null)
  const [hasMoreCandles, setHasMoreCandles] = useState(true)

  // Drawing tools state
  const [drawingTool, setDrawingTool] = useState('pointer')
  const [drawings, setDrawings] = useState(() => {
    try {
      const saved = localStorage.getItem('drawings')
      return saved ? JSON.parse(saved) : []
    } catch { return [] }
  })
  const [chartInstance, setChartInstance] = useState(null)
  const [seriesInstance, setSeriesInstance] = useState(null)
  const [chartContainerEl, setChartContainerEl] = useState(null)

  // RSI chart instance (for crosshair sync)
  const [rsiChartInstance, setRsiChartInstance] = useState(null)
  const [rsiSeriesInstance, setRsiSeriesInstance] = useState(null)

  // Chart settings
  const [chartBg, setChartBg] = useState(() => localStorage.getItem('chartBg') || '#0a0e14')

  const handleChangeBg = useCallback((color) => {
    setChartBg(color)
    localStorage.setItem('chartBg', color)
  }, [])

  // Persist UI preferences to localStorage
  useEffect(() => { localStorage.setItem('activeTab', activeTab) }, [activeTab])
  useEffect(() => { localStorage.setItem('chartView', chartView) }, [chartView])
  useEffect(() => {
    try { localStorage.setItem('drawings', JSON.stringify(drawings)) } catch {}
  }, [drawings])

  // Safety: flush drawings on page unload (belt and suspenders)
  useEffect(() => {
    const handler = () => {
      try { localStorage.setItem('drawings', JSON.stringify(drawings)) } catch {}
    }
    window.addEventListener('beforeunload', handler)
    return () => window.removeEventListener('beforeunload', handler)
  }, [drawings])

  // Re-fetch all data from REST API (called on mount + every reconnect)
  const fetchAllData = () => {
    fetch('/api/all')
      .then(res => res.json())
      .then(d => setData(d))
      .catch(console.error)

    fetch('/api/candles')
      .then(res => res.json())
      .then(d => {
        if (d.candles) setChartCandles(shiftCandlesIST(d.candles))
        if (d.current_candle) setLiveCandle(shiftIST(d.current_candle))
      })
      .catch(console.error)
  }

  const connectWebSocket = () => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const wsUrl = `${protocol}//${window.location.host}/ws`

    try {
      wsRef.current = new WebSocket(wsUrl)

      wsRef.current.onopen = () => {
        console.log('WebSocket connected')
        setConnected(true)
        // Re-fetch all data on every (re)connect so dashboard stays fresh
        fetchAllData()
      }

      wsRef.current.onmessage = (event) => {
        const message = JSON.parse(event.data)
        if (message.type === 'ping') return

        if (message.type === 'candle_update') {
          setLiveCandle(shiftIST(message.candle))
          return
        }

        if (message.type === 'candle_close') {
          // Append closed candle to chart data (IST-shifted)
          const shifted = shiftIST(message.candle)
          setChartCandles(prev => {
            if (!prev) return [shifted]
            return [...prev, shifted]
          })
          setLiveCandle(null)
          return
        }

        // Full state update
        setData(message)
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
    // Connect WS — onopen handler calls fetchAllData() for initial + reconnect
    connectWebSocket()

    return () => {
      if (wsRef.current) wsRef.current.close()
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current)
    }
  }, [])

  const handleFullscreen = useCallback(() => {
    if (!chartAreaRef.current) return
    if (document.fullscreenElement) {
      document.exitFullscreen()
    } else {
      chartAreaRef.current.requestFullscreen()
    }
  }, [])

  const handleTimeRangeChange = useCallback((range) => {
    setTimeRange(range)
  }, [])

  const handleLoadMore = useCallback(() => {
    if (!chartCandles || chartCandles.length === 0 || !hasMoreCandles) return
    // Reverse IST offset for API call (API expects UTC)
    const oldestTime = chartCandles[0].time - IST_OFFSET
    fetch(`/api/candles/history?before=${oldestTime}&limit=1000`)
      .then(res => res.json())
      .then(d => {
        if (d.candles && d.candles.length > 0) {
          setChartCandles(prev => [...shiftCandlesIST(d.candles), ...(prev || [])])
        }
        if (d.has_more === false) setHasMoreCandles(false)
      })
      .catch(console.error)
  }, [chartCandles, hasMoreCandles])

  const handleChartReady = useCallback((chart, series, containerEl) => {
    setChartInstance(chart)
    setSeriesInstance(series)
    setChartContainerEl(containerEl)
  }, [])

  const handleAddDrawing = useCallback((drawing) => {
    setDrawings(prev => [...prev, drawing])
  }, [])

  const handleRemoveDrawing = useCallback((id) => {
    setDrawings(prev => prev.filter(d => d.id !== id))
  }, [])

  const handleUpdateDrawing = useCallback((id, newPoints) => {
    setDrawings(prev => prev.map(d => d.id === id ? { ...d, points: newPoints } : d))
  }, [])

  const handleClearDrawings = useCallback(() => {
    setDrawings([])
  }, [])

  const handleDrawingCancel = useCallback(() => {
    setDrawingTool('pointer')
  }, [])

  const handleRsiChartReady = useCallback((chart, series) => {
    setRsiChartInstance(chart)
    setRsiSeriesInstance(series)
  }, [])

  // Crosshair sync: main chart → RSI pane
  useEffect(() => {
    if (!chartInstance || !rsiChartInstance || !rsiSeriesInstance) return
    const handler = (param) => {
      if (!param.time) {
        try { rsiChartInstance.clearCrosshairPosition() } catch (e) {}
        return
      }
      try {
        rsiChartInstance.setCrosshairPosition(50, param.time, rsiSeriesInstance)
      } catch (e) {}
    }
    chartInstance.subscribeCrosshairMove(handler)
    return () => {
      try { chartInstance.unsubscribeCrosshairMove(handler) } catch (e) {}
    }
  }, [chartInstance, rsiChartInstance, rsiSeriesInstance])

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

  if (status?.status === 'STARTING' || !status?.price) {
    return (
      <div className="dashboard">
        <header className="header">
          <h1>
            V1.3 Bot
            <span className="pair">{config?.pair || 'BTCUSDT'}</span>
            <span className="timeframe">{config?.timeframe || '15m'}</span>
          </h1>
          <div className="header-right">
            <span className={`status-badge starting`}>STARTING</span>
          </div>
        </header>
        <div className="loading">
          <div className="loading-spinner"></div>
          <span>Warming up indicators (need 300+ bars)...</span>
        </div>
      </div>
    )
  }

  return (
    <div className="dashboard">
      {/* Header */}
      <header className="header">
        <h1>
          V1.3 Bot
          <span className="pair">{config?.pair || 'BTCUSDT'}</span>
          <span className="timeframe">{config?.timeframe || '15m'}</span>
          {config?.config_hash && (
            <span className="config-hash">#{config.config_hash}</span>
          )}
        </h1>
        <div className="header-right">
          <span className="uptime">{formatUptime(status?.uptime_seconds)}</span>
          <span className="mode-badge">{config?.mode?.toUpperCase() || 'PAPER'}</span>
          <div className="connection-status">
            <span className={`connection-dot ${connected ? 'connected' : 'disconnected'}`}></span>
            {connected ? 'Live' : 'Reconnecting...'}
          </div>
          <span className={`status-badge ${status?.status?.toLowerCase() || 'starting'}`}>
            {status?.status || 'STARTING'}
          </span>
        </div>
      </header>

      {/* Main Layout: Chart (left) + Sidebar (right) */}
      <div className="main-layout">
        {/* === LEFT: Chart Area === */}
        <div className="chart-area" ref={chartAreaRef}>
          {/* Chart View Toggle */}
          <div className="chart-view-toggle">
            <button
              className={`chart-view-btn ${chartView === 'tradingview' ? 'active' : ''}`}
              onClick={() => setChartView('tradingview')}
            >
              TradingView
            </button>
            <button
              className={`chart-view-btn ${chartView === 'bot' ? 'active' : ''}`}
              onClick={() => setChartView('bot')}
            >
              Bot Chart
            </button>
          </div>

          {/* TradingView Widget */}
          <div className="chart-panel" style={{ display: chartView === 'tradingview' ? 'block' : 'none' }}>
            <TradingViewChart />
          </div>

          {/* Bot Chart (lightweight-charts) */}
          <div className="chart-panel bot-chart-panel" style={{ display: chartView === 'bot' ? 'flex' : 'none' }}>
            <div className={`bot-chart-main ${drawingTool !== 'pointer' ? 'tool-active' : ''}`}>
              <CandlestickChart
                candles={chartCandles}
                currentCandle={liveCandle}
                trades={trades}
                onTimeRangeChange={handleTimeRangeChange}
                onLoadMore={handleLoadMore}
                isVisible={chartView === 'bot'}
                onChartReady={handleChartReady}
                chartBg={chartBg}
              />
              {chartInstance && seriesInstance && chartContainerEl && (
                <DrawingOverlay
                  chart={chartInstance}
                  series={seriesInstance}
                  chartContainerEl={chartContainerEl}
                  activeTool={drawingTool}
                  drawings={drawings}
                  onAddDrawing={handleAddDrawing}
                  onRemoveDrawing={handleRemoveDrawing}
                  onUpdateDrawing={handleUpdateDrawing}
                  onCancel={handleDrawingCancel}
                />
              )}
              <DrawingToolbar
                activeTool={drawingTool}
                onToolChange={setDrawingTool}
                onClearAll={handleClearDrawings}
              />
              {chartInstance && seriesInstance && (
                <ChartLegend
                  chart={chartInstance}
                  series={seriesInstance}
                  candles={chartCandles}
                  liveCandle={liveCandle}
                  pair={config?.pair}
                  timeframe={config?.timeframe}
                />
              )}
            </div>
            <div className="bot-chart-rsi">
              <RsiPane candles={chartCandles} timeRange={timeRange} chartBg={chartBg} onChartReady={handleRsiChartReady} />
            </div>
          </div>

          <button className="fullscreen-btn" onClick={handleFullscreen} title="Toggle Fullscreen">
            &#x26F6;
          </button>
          {chartView === 'bot' && (
            <ChartSettings chartBg={chartBg} onChangeBg={handleChangeBg} />
          )}
        </div>

        {/* === RIGHT: Sidebar === */}
        <div className="sidebar">
          {/* Indicator Cards (2x2) */}
          <div className="grid grid-2 sidebar-indicators">
            <StatusPanel label="Price" value={status?.price} format="price" panelType="price" />
            <StatusPanel label="RSI (14)" value={status?.rsi} format="rsi" panelType="rsi" />
            <StatusPanel label="ATR %" value={status?.atr_percentile} format="percentile" panelType="atr" />
            <StatusPanel label="EMA Sep" value={status?.ema_separation} format="ema" panelType="ema" />
          </div>

          {/* Position Card */}
          <PositionCard position={position} />

          {/* Regime + Bars */}
          <div className="grid grid-2 sidebar-regime">
            <StatusPanel label="Regime" value={status?.regime || '---'} panelType="regime" />
            <StatusPanel label="Bars" value={status?.bar_count || 0} format="number" panelType="bars" />
          </div>

          {/* Signal Proximity */}
          <SignalProximity proximity={data?.proximity} />

          {/* Tab Bar */}
          <TabBar activeTab={activeTab} onTabChange={setActiveTab} />

          {/* Tab Content */}
          <div className="sidebar-tab-content">
            {activeTab === 'trades' && (
              <>
                <StatsSection stats={stats} />
                <EquityCurve trades={trades} />
                <TradesList trades={trades} />
              </>
            )}

            {activeTab === 'signals' && (
              <>
                <SignalStats signals={signals} />
                <SignalLog signals={signals} fullHeight />
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
