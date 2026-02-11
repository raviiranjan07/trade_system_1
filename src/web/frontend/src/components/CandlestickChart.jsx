import React, { useEffect, useRef } from 'react'
import { createChart, ColorType, CrosshairMode, CandlestickSeries, LineSeries, HistogramSeries } from 'lightweight-charts'
import { calculateMA } from '../utils/maCalc'

const IST_OFFSET = 19800 // UTC+5:30 for trade markers

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

function CandlestickChart({ candles, currentCandle, trades, indicators, onTimeRangeChange, onLoadMore, isVisible, onChartReady, chartBg }) {
  const containerRef = useRef(null)
  const chartRef = useRef(null)
  const candleSeriesRef = useRef(null)
  const volumeSeriesRef = useRef(null)
  const lastTimeRef = useRef(0)
  const firstTimeRef = useRef(0)
  const loadingMoreRef = useRef(false)
  const suppressScrollRef = useRef(false)

  // Dynamic indicator series: { [id]: LineSeries }
  const indicatorSeriesRef = useRef({})

  // Refs for callbacks to avoid stale closures in chart subscription
  const onLoadMoreRef = useRef(onLoadMore)
  const onTimeRangeChangeRef = useRef(onTimeRangeChange)
  useEffect(() => { onLoadMoreRef.current = onLoadMore }, [onLoadMore])
  useEffect(() => { onTimeRangeChangeRef.current = onTimeRangeChange }, [onTimeRangeChange])

  // Chart initialization (runs once)
  useEffect(() => {
    if (!containerRef.current) return

    const w = containerRef.current.clientWidth || 800
    const h = containerRef.current.clientHeight || 400

    const chart = createChart(containerRef.current, {
      width: w,
      height: h,
      layout: {
        background: { type: ColorType.Solid, color: '#0a0e14' },
        textColor: '#8892a0',
        fontSize: 12,
      },
      grid: {
        vertLines: { color: '#1a2028' },
        horzLines: { color: '#1a2028' },
      },
      crosshair: { mode: CrosshairMode.Normal },
      rightPriceScale: {
        borderVisible: false,
      },
      timeScale: {
        borderVisible: false,
        timeVisible: true,
        secondsVisible: false,
      },
    })
    chart.applyOptions({ branding: { visible: false } })

    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: '#00d97e',
      downColor: '#e63757',
      borderUpColor: '#00d97e',
      borderDownColor: '#e63757',
      wickUpColor: '#00d97e',
      wickDownColor: '#e63757',
    })

    const volumeSeries = chart.addSeries(HistogramSeries, {
      priceFormat: { type: 'volume' },
      priceScaleId: 'volume',
    })
    chart.priceScale('volume').applyOptions({
      scaleMargins: { top: 0.85, bottom: 0 },
    })

    // Expose time range changes for RSI sync + lazy loading (uses refs to avoid stale closures)
    chart.timeScale().subscribeVisibleLogicalRangeChange((logicalRange) => {
      if (logicalRange) {
        if (onTimeRangeChangeRef.current) onTimeRangeChangeRef.current(logicalRange)

        if (suppressScrollRef.current) return

        if (logicalRange.from < 50 && onLoadMoreRef.current && !loadingMoreRef.current) {
          loadingMoreRef.current = true
          onLoadMoreRef.current()
          // Safety: reset after 5s even if fetch fails or returns empty
          setTimeout(() => { loadingMoreRef.current = false }, 5000)
        }
      }
    })

    chartRef.current = chart
    candleSeriesRef.current = candleSeries
    volumeSeriesRef.current = volumeSeries

    if (onChartReady) onChartReady(chart, candleSeries, containerRef.current)

    // ResizeObserver for responsive chart sizing (more reliable than window resize)
    const ro = new ResizeObserver(entries => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect
        if (width > 0 && height > 0 && chartRef.current) {
          chartRef.current.applyOptions({ width, height })
        }
      }
    })
    ro.observe(containerRef.current)

    return () => {
      ro.disconnect()
      indicatorSeriesRef.current = {}
      if (onChartReady) onChartReady(null, null, null)
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

  // Set/update historical data
  useEffect(() => {
    if (!candleSeriesRef.current || !candles || candles.length === 0) return

    const valid = candles.filter(c =>
      c.time != null && c.open != null && c.high != null && c.low != null && c.close != null
    )
    if (valid.length === 0) return

    const newLastTime = valid[valid.length - 1].time
    const newFirstTime = valid[0].time

    const isFirstLoad = lastTimeRef.current === 0
    const isPrepend = !isFirstLoad && newFirstTime < firstTimeRef.current
    const isAppend = !isFirstLoad && !isPrepend && newLastTime >= lastTimeRef.current

    if (isFirstLoad || isPrepend) {
      const ts = chartRef.current?.timeScale()
      const visibleRange = ts && !isFirstLoad ? ts.getVisibleLogicalRange() : null
      const prependCount = isPrepend ? valid.findIndex(c => c.time >= firstTimeRef.current) : 0

      candleSeriesRef.current.setData(
        valid.map(c => ({
          time: c.time, open: c.open, high: c.high, low: c.low, close: c.close,
        }))
      )

      volumeSeriesRef.current.setData(
        valid.map(c => ({
          time: c.time,
          value: c.volume,
          color: c.close >= c.open ? 'rgba(0,217,126,0.12)' : 'rgba(230,55,87,0.12)',
        }))
      )

      if (isFirstLoad) {
        chartRef.current.timeScale().fitContent()
      } else if (visibleRange && prependCount > 0) {
        suppressScrollRef.current = true
        try {
          chartRef.current.timeScale().setVisibleLogicalRange({
            from: visibleRange.from + prependCount,
            to: visibleRange.to + prependCount,
          })
        } catch (e) {
          // ignore
        }
        setTimeout(() => {
          suppressScrollRef.current = false
          loadingMoreRef.current = false
        }, 500)
      } else {
        loadingMoreRef.current = false
      }
    } else if (isAppend) {
      const c = valid[valid.length - 1]
      try {
        candleSeriesRef.current.update({
          time: c.time, open: c.open, high: c.high, low: c.low, close: c.close,
        })
        volumeSeriesRef.current.update({
          time: c.time,
          value: c.volume,
          color: c.close >= c.open ? 'rgba(0,217,126,0.12)' : 'rgba(230,55,87,0.12)',
        })
      } catch (e) {
        // Live candle update may have advanced the series beyond lastTimeRef.
        // Do an immediate full setData to resync instead of deferring.
        try {
          candleSeriesRef.current.setData(
            valid.map(v => ({ time: v.time, open: v.open, high: v.high, low: v.low, close: v.close }))
          )
          volumeSeriesRef.current.setData(
            valid.map(v => ({
              time: v.time, value: v.volume,
              color: v.close >= v.open ? 'rgba(0,217,126,0.12)' : 'rgba(230,55,87,0.12)',
            }))
          )
        } catch (e2) {
          // ignore
        }
      }
    }

    lastTimeRef.current = newLastTime
    firstTimeRef.current = newFirstTime
  }, [candles])

  // Dynamic indicator series management
  useEffect(() => {
    if (!chartRef.current || !candles || candles.length === 0) return
    if (!indicators) return

    const chart = chartRef.current
    const currentSeries = indicatorSeriesRef.current
    const activeIds = new Set(indicators.map(i => String(i.id)))

    // Remove series for deleted indicators
    for (const id of Object.keys(currentSeries)) {
      if (!activeIds.has(id)) {
        try { chart.removeSeries(currentSeries[id]) } catch (e) {}
        delete currentSeries[id]
      }
    }

    // Filter valid candles for MA calculation
    const valid = candles.filter(c =>
      c.time != null && c.open != null && c.high != null && c.low != null && c.close != null
    )
    if (valid.length === 0) return

    // Add or update each indicator
    for (const ind of indicators) {
      const id = String(ind.id)
      let series = currentSeries[id]

      if (!series) {
        series = chart.addSeries(LineSeries, {
          color: ind.color,
          lineWidth: ind.lineWidth || 2,
          priceLineVisible: false,
          lastValueVisible: false,
          visible: ind.visible !== false,
        })
        currentSeries[id] = series
      } else {
        series.applyOptions({
          color: ind.color,
          lineWidth: ind.lineWidth || 2,
          visible: ind.visible !== false,
        })
      }

      // Calculate and set data
      if (ind.visible !== false) {
        const data = calculateMA(ind.type, valid, ind.period, ind.source || 'close')
        try { series.setData(data) } catch (e) {}
      }
    }
  }, [indicators, candles])

  // Trade markers (entry/exit arrows)
  useEffect(() => {
    if (!candleSeriesRef.current || !trades || trades.length === 0) return

    const markers = []
    for (const t of trades) {
      if (t.entry_time) {
        const entryTimeUTC = typeof t.entry_time === 'number' ? t.entry_time : Math.floor(new Date(t.entry_time).getTime() / 1000)
        const isLong = t.side === 'LONG'
        markers.push({
          time: entryTimeUTC + IST_OFFSET,
          position: isLong ? 'belowBar' : 'aboveBar',
          color: isLong ? '#00d97e' : '#e63757',
          shape: isLong ? 'arrowUp' : 'arrowDown',
          text: isLong ? 'L' : 'S',
        })
      }
      if (t.exit_time) {
        const exitTimeUTC = typeof t.exit_time === 'number' ? t.exit_time : Math.floor(new Date(t.exit_time).getTime() / 1000)
        markers.push({
          time: exitTimeUTC + IST_OFFSET,
          position: 'inBar',
          color: '#f59e0b',
          shape: 'circle',
          text: 'X',
        })
      }
    }

    markers.sort((a, b) => a.time - b.time)

    try {
      candleSeriesRef.current.setMarkers(markers)
    } catch (e) {
      // Markers may fail if candle data hasn't loaded yet
    }
  }, [trades, candles])

  // Live candle update
  useEffect(() => {
    if (!candleSeriesRef.current || !currentCandle) return
    if (currentCandle.open == null || currentCandle.close == null) return

    try {
      candleSeriesRef.current.update({
        time: currentCandle.time,
        open: currentCandle.open,
        high: currentCandle.high,
        low: currentCandle.low,
        close: currentCandle.close,
      })
      if (currentCandle.volume != null) {
        volumeSeriesRef.current.update({
          time: currentCandle.time,
          value: currentCandle.volume,
          color: currentCandle.close >= currentCandle.open
            ? 'rgba(0,217,126,0.12)' : 'rgba(230,55,87,0.12)',
        })
      }
    } catch (e) {
      // Ignore stale update errors after reconnect
    }
  }, [currentCandle])

  // Fix chart size when tab becomes visible
  useEffect(() => {
    if (isVisible && chartRef.current && containerRef.current) {
      setTimeout(() => {
        const w = containerRef.current.clientWidth
        const h = containerRef.current.clientHeight
        if (w > 0 && h > 0) chartRef.current.applyOptions({ width: w, height: h })
      }, 50)
    }
  }, [isVisible])

  return <div ref={containerRef} style={{ position: 'absolute', inset: 0 }} />
}

export default CandlestickChart
