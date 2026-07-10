import React, { useEffect, useRef } from 'react'
import { createChart, ColorType, LineSeries } from 'lightweight-charts'

function buildCurveData(tradesArr) {
  if (!tradesArr || tradesArr.length === 0) return []
  // Trades are newest-first, reverse for chronological order
  const sorted = [...tradesArr].reverse()

  let cumBps = 0
  const data = []
  for (const t of sorted) {
    cumBps += t.net_profit_bps || 0
    let ts = 0
    if (t.exit_time) {
      const d = new Date(t.exit_time)
      if (!isNaN(d.getTime())) {
        ts = Math.floor(d.getTime() / 1000)
      }
    }
    if (ts > 0) {
      data.push({ time: ts, value: Math.round(cumBps * 10) / 10 })
    }
  }

  // Deduplicate timestamps (keep last value for each timestamp)
  const deduped = []
  for (let i = 0; i < data.length; i++) {
    if (i === data.length - 1 || data[i].time !== data[i + 1].time) {
      deduped.push(data[i])
    }
  }
  return deduped
}

function EquityCurve({ trades, mlTrades, mlAttnTrades, mlV3Trades }) {
  const containerRef = useRef(null)
  const chartRef = useRef(null)
  const v14SeriesRef = useRef(null)
  const mlSeriesRef = useRef(null)
  const mlAttnSeriesRef = useRef(null)
  const mlV3SeriesRef = useRef(null)
  const zeroLineRef = useRef(null)

  // Chart initialization
  useEffect(() => {
    if (!containerRef.current) return

    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height: 320,
      layout: {
        background: { type: ColorType.Solid, color: '#0a0e14' },
        textColor: '#8892a0',
        fontSize: 11,
      },
      grid: {
        vertLines: { color: '#1a2028' },
        horzLines: { color: '#1a2028' },
      },
      rightPriceScale: {
        borderColor: '#2a3544',
      },
      timeScale: {
        borderColor: '#2a3544',
        timeVisible: true,
        secondsVisible: false,
      },
      crosshair: {
        horzLine: { color: '#3a4555', labelBackgroundColor: '#2a3544' },
        vertLine: { color: '#3a4555', labelBackgroundColor: '#2a3544' },
      },
    })
    chart.applyOptions({ branding: { visible: false } })

    // V1.4 line (blue)
    const v14Series = chart.addSeries(LineSeries, {
      color: '#3b82f6',
      lineWidth: 2,
      lastValueVisible: true,
      priceLineVisible: false,
      crosshairMarkerVisible: true,
    })

    // ML line (purple)
    const mlSeries = chart.addSeries(LineSeries, {
      color: '#8b5cf6',
      lineWidth: 2,
      lastValueVisible: true,
      priceLineVisible: false,
      crosshairMarkerVisible: true,
    })

    // ML V2 line (amber)
    const mlAttnSeries = chart.addSeries(LineSeries, {
      color: '#f59e0b',
      lineWidth: 2,
      lastValueVisible: true,
      priceLineVisible: false,
      crosshairMarkerVisible: true,
    })

    // ML V3 line (green)
    const mlV3Series = chart.addSeries(LineSeries, {
      color: '#10b981',
      lineWidth: 2,
      lastValueVisible: true,
      priceLineVisible: false,
      crosshairMarkerVisible: true,
    })

    // Zero reference line
    const zeroLine = chart.addSeries(LineSeries, {
      color: 'rgba(136, 146, 160, 0.3)',
      lineWidth: 1,
      lineStyle: 2,
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
    })

    chartRef.current = chart
    v14SeriesRef.current = v14Series
    mlSeriesRef.current = mlSeries
    mlAttnSeriesRef.current = mlAttnSeries
    mlV3SeriesRef.current = mlV3Series
    zeroLineRef.current = zeroLine

    const handleResize = () => {
      if (containerRef.current) {
        chart.applyOptions({ width: containerRef.current.clientWidth })
      }
    }
    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      chart.remove()
    }
  }, [])

  // Update data when trades change
  useEffect(() => {
    if (!v14SeriesRef.current) return

    const v14Data = buildCurveData(trades)
    const mlData = buildCurveData(mlTrades)
    const mlAttnData = buildCurveData(mlAttnTrades)
    const mlV3Data = buildCurveData(mlV3Trades)

    if (v14Data.length > 0) {
      v14SeriesRef.current.setData(v14Data)
    }
    if (mlData.length > 0) {
      mlSeriesRef.current.setData(mlData)
    }
    if (mlAttnData.length > 0 && mlAttnSeriesRef.current) {
      mlAttnSeriesRef.current.setData(mlAttnData)
    }
    if (mlV3Data.length > 0 && mlV3SeriesRef.current) {
      mlV3SeriesRef.current.setData(mlV3Data)
    }

    // Zero line spanning full range of all series
    const allTimes = [...v14Data.map(d => d.time), ...mlData.map(d => d.time), ...mlAttnData.map(d => d.time), ...mlV3Data.map(d => d.time)]
    if (allTimes.length > 0) {
      const minTime = Math.min(...allTimes)
      const maxTime = Math.max(...allTimes)
      zeroLineRef.current.setData([
        { time: minTime, value: 0 },
        { time: maxTime, value: 0 },
      ])
    }

    if (chartRef.current) {
      chartRef.current.timeScale().fitContent()
    }
  }, [trades, mlTrades, mlAttnTrades, mlV3Trades])

  const hasTrades = (trades && trades.length > 0) || (mlTrades && mlTrades.length > 0) || (mlAttnTrades && mlAttnTrades.length > 0) || (mlV3Trades && mlV3Trades.length > 0)

  if (!hasTrades) {
    return (
      <div className="card equity-section">
        <div className="card-header">Equity Curve</div>
        <div className="equity-empty">No trades yet</div>
      </div>
    )
  }

  return (
    <div className="card equity-section">
      <div className="card-header">Equity Curve (cumulative bps)</div>
      <div style={{ position: 'relative' }}>
        <div ref={containerRef} style={{ width: '100%' }} />
        <div style={{
          position: 'absolute',
          top: 8,
          right: 12,
          display: 'flex',
          gap: '12px',
          fontSize: '11px',
          color: '#8892a0',
          background: 'rgba(10, 14, 20, 0.75)',
          padding: '4px 8px',
          borderRadius: '4px',
        }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <span style={{ width: 8, height: 8, borderRadius: '50%', background: '#3b82f6', display: 'inline-block' }} />
            V1.4
          </span>
          <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <span style={{ width: 8, height: 8, borderRadius: '50%', background: '#8b5cf6', display: 'inline-block' }} />
            ML
          </span>
          <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <span style={{ width: 8, height: 8, borderRadius: '50%', background: '#f59e0b', display: 'inline-block' }} />
            ML V2
          </span>
          <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <span style={{ width: 8, height: 8, borderRadius: '50%', background: '#10b981', display: 'inline-block' }} />
            ML V3
          </span>
        </div>
      </div>
    </div>
  )
}

export default EquityCurve
