import React, { useEffect, useRef, useMemo } from 'react'
import { createChart, ColorType, BaselineSeries, LineSeries } from 'lightweight-charts'

// Compute grid color: lighten background by ~16 RGB steps
const gridFromBg = (hex) => {
  const r = parseInt(hex.slice(1, 3), 16)
  const g = parseInt(hex.slice(3, 5), 16)
  const b = parseInt(hex.slice(5, 7), 16)
  const lr = Math.min(255, r + 16).toString(16).padStart(2, '0')
  const lg = Math.min(255, g + 16).toString(16).padStart(2, '0')
  const lb = Math.min(255, b + 16).toString(16).padStart(2, '0')
  return `#${lr}${lg}${lb}`
}

function RsiPane({ candles, timeRange, chartBg, onChartReady }) {
  const containerRef = useRef(null)
  const chartRef = useRef(null)
  const rsiSeriesRef = useRef(null)
  const syncSeriesRef = useRef(null) // hidden LineSeries for crosshair sync

  // Get latest RSI value
  const latestRsi = useMemo(() => {
    if (!candles || candles.length === 0) return null
    for (let i = candles.length - 1; i >= 0; i--) {
      if (candles[i].rsi != null) return candles[i].rsi
    }
    return null
  }, [candles])

  // Zone-based colors (V1 thresholds: 20/80)
  const rsiColor = latestRsi != null
    ? latestRsi >= 80 ? '#ef5350'    // Overbought - red
    : latestRsi <= 20 ? '#26a69a'    // Oversold - teal
    : '#787B86'                       // Neutral - gray
    : '#787B86'

  // Chart initialization
  useEffect(() => {
    if (!containerRef.current) return

    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height: containerRef.current.clientHeight || 150,
      layout: {
        background: { type: ColorType.Solid, color: '#0a0e14' },
        textColor: '#8892a0',
        fontSize: 10,
      },
      grid: {
        vertLines: { color: '#1a2028' },
        horzLines: { color: '#1a2028' },
      },
      rightPriceScale: {
        borderColor: '#2a3544',
        scaleMargins: { top: 0.05, bottom: 0.05 },
      },
      timeScale: { visible: false },
      crosshair: {
        horzLine: { visible: false },
        vertLine: { visible: true, color: 'rgba(255,255,255,0.15)', width: 1, style: 2, labelVisible: false },
      },
    })
    chart.applyOptions({ branding: { visible: false } })

    // Remove scale borders for seamless look
    chart.priceScale('right').applyOptions({ borderVisible: false })

    // RSI as BaselineSeries — dual-color at midline 50
    const rsiSeries = chart.addSeries(BaselineSeries, {
      baseValue: { type: 'price', price: 50 },
      topLineColor: '#ef5350',
      topFillColor1: 'rgba(239, 83, 80, 0.1)',
      topFillColor2: 'transparent',
      bottomLineColor: '#26a69a',
      bottomFillColor1: 'transparent',
      bottomFillColor2: 'rgba(38, 166, 154, 0.1)',
      lineWidth: 2,
      priceLineVisible: false,
      lastValueVisible: true,
      title: '',
      priceFormat: { type: 'price', precision: 1, minMove: 0.1 },
    })

    // Hidden LineSeries used for crosshair sync (BaselineSeries crashes on setCrosshairPosition)
    const syncSeries = chart.addSeries(LineSeries, {
      color: 'transparent',
      lineWidth: 0,
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
    })

    // Threshold price lines (no separate series needed)
    const lineOpts = { lineWidth: 1, lineStyle: 2, axisLabelVisible: false, title: '' }

    rsiSeries.createPriceLine({ price: 80, color: 'rgba(239, 83, 80, 0.40)', ...lineOpts })
    rsiSeries.createPriceLine({ price: 50, color: 'rgba(120, 123, 134, 0.25)', ...lineOpts })
    rsiSeries.createPriceLine({ price: 20, color: 'rgba(38, 166, 154, 0.40)', ...lineOpts })

    chartRef.current = chart
    rsiSeriesRef.current = rsiSeries
    syncSeriesRef.current = syncSeries

    // Expose chart + sync series (NOT the BaselineSeries) for crosshair sync
    if (onChartReady) onChartReady(chart, syncSeries)

    // ResizeObserver for responsive sizing
    const ro = new ResizeObserver(entries => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect
        if (width > 0 && height > 0) {
          chart.applyOptions({ width, height })
        }
      }
    })
    ro.observe(containerRef.current)

    return () => {
      ro.disconnect()
      if (onChartReady) onChartReady(null, null)
      chart.remove()
    }
  }, [])

  // Apply chart background + grid color when chartBg changes
  useEffect(() => {
    if (!chartRef.current || !chartBg) return
    const grid = gridFromBg(chartBg)
    chartRef.current.applyOptions({
      layout: { background: { type: ColorType.Solid, color: chartBg } },
      grid: { vertLines: { color: grid }, horzLines: { color: grid } },
    })
  }, [chartBg])

  // Sync logical range
  useEffect(() => {
    if (!chartRef.current || !timeRange) return
    try {
      chartRef.current.timeScale().setVisibleLogicalRange(timeRange)
    } catch (e) {}
  }, [timeRange])

  // Set/update RSI data — always setData (small dataset, avoids incremental update issues)
  useEffect(() => {
    if (!rsiSeriesRef.current || !candles || candles.length === 0) return

    const rsiData = candles
      .filter(c => c.rsi != null && c.time != null)
      .map(c => ({ time: c.time, value: c.rsi }))

    if (rsiData.length === 0) return

    try {
      rsiSeriesRef.current.setData(rsiData)
      if (syncSeriesRef.current) syncSeriesRef.current.setData(rsiData)
    } catch (e) {
      // ignore
    }
  }, [candles])

  return (
    <div className="rsi-pane">
      <div className="rsi-header" style={chartBg ? { background: chartBg } : undefined}>
        <span className="rsi-dot" style={{ background: rsiColor }} />
        <span className="rsi-label">RSI</span>
        <span className="rsi-period">(14)</span>
        <span className="rsi-value" style={{ color: rsiColor }}>
          {latestRsi != null ? latestRsi.toFixed(1) : '—'}
        </span>
      </div>
      <div ref={containerRef} className="rsi-chart-container" />
    </div>
  )
}

export default RsiPane
