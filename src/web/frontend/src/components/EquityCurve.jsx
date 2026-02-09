import React, { useEffect, useRef } from 'react'
import { createChart, ColorType, LineSeries, BaselineSeries } from 'lightweight-charts'

function EquityCurve({ trades }) {
  const containerRef = useRef(null)
  const chartRef = useRef(null)
  const seriesRef = useRef(null)
  const zeroLineRef = useRef(null)

  // Chart initialization
  useEffect(() => {
    if (!containerRef.current) return

    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height: 250,
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

    // Baseline series: green above 0, red below 0
    const series = chart.addSeries(BaselineSeries, {
      baseValue: { type: 'price', price: 0 },
      topLineColor: '#00d97e',
      topFillColor1: 'rgba(0, 217, 126, 0.12)',
      topFillColor2: 'rgba(0, 217, 126, 0.01)',
      bottomLineColor: '#e63757',
      bottomFillColor1: 'rgba(230, 55, 87, 0.01)',
      bottomFillColor2: 'rgba(230, 55, 87, 0.12)',
      lineWidth: 2,
      lastValueVisible: true,
      priceLineVisible: false,
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
    seriesRef.current = series
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
    if (!seriesRef.current || !trades || trades.length === 0) return

    // Trades are newest-first, reverse for chronological order
    const sorted = [...trades].reverse()

    // Build equity curve: cumulative bps
    let cumBps = 0
    const data = []
    for (const t of sorted) {
      cumBps += t.net_profit_bps || 0
      // Use exit_time as timestamp
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

    if (data.length === 0) return

    // Deduplicate timestamps (keep last value for each timestamp)
    const deduped = []
    for (let i = 0; i < data.length; i++) {
      if (i === data.length - 1 || data[i].time !== data[i + 1].time) {
        deduped.push(data[i])
      }
    }

    seriesRef.current.setData(deduped)

    // Zero line spanning full range
    zeroLineRef.current.setData([
      { time: deduped[0].time, value: 0 },
      { time: deduped[deduped.length - 1].time, value: 0 },
    ])

    chartRef.current.timeScale().fitContent()
  }, [trades])

  if (!trades || trades.length === 0) {
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
      <div ref={containerRef} style={{ width: '100%' }} />
    </div>
  )
}

export default EquityCurve
