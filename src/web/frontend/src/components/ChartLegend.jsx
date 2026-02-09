import React, { useEffect, useMemo, useState } from 'react'

const fmtPrice = (v) => {
  if (v == null) return '—'
  return v.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
}

const fmtVol = (v) => {
  if (v == null) return '—'
  if (v >= 1e9) return (v / 1e9).toFixed(2) + 'B'
  if (v >= 1e6) return (v / 1e6).toFixed(2) + 'M'
  if (v >= 1e3) return (v / 1e3).toFixed(2) + 'K'
  return v.toFixed(2)
}

function ChartLegend({ chart, series, candles, liveCandle, pair, timeframe }) {
  const [hoverData, setHoverData] = useState(null)

  // O(1) lookup: time -> candle (for SMA + volume)
  const candleMap = useMemo(() => {
    if (!candles) return new Map()
    const map = new Map()
    for (const c of candles) {
      if (c.time != null) map.set(c.time, c)
    }
    return map
  }, [candles])

  // Subscribe to crosshair move for hover values
  useEffect(() => {
    if (!chart || !series) return

    const handleMove = (param) => {
      if (param.time) {
        const ohlc = param.seriesData?.get(series)
        const candle = candleMap.get(param.time)
        if (ohlc) {
          setHoverData({
            open: ohlc.open,
            high: ohlc.high,
            low: ohlc.low,
            close: ohlc.close,
            sma: candle?.sma,
            volume: candle?.volume,
          })
          return
        }
      }
      setHoverData(null)
    }

    chart.subscribeCrosshairMove(handleMove)
    return () => chart.unsubscribeCrosshairMove(handleMove)
  }, [chart, series, candleMap])

  // Latest closed candle
  const latest = candles && candles.length > 0 ? candles[candles.length - 1] : null

  // Display: hovered candle > live candle > latest closed candle
  const d = hoverData || (liveCandle && liveCandle.open != null ? {
    open: liveCandle.open,
    high: liveCandle.high,
    low: liveCandle.low,
    close: liveCandle.close,
    sma: latest?.sma,
    volume: liveCandle.volume ?? latest?.volume,
  } : latest ? {
    open: latest.open,
    high: latest.high,
    low: latest.low,
    close: latest.close,
    sma: latest.sma,
    volume: latest.volume,
  } : null)

  if (!d || d.open == null) return null

  const isUp = d.close >= d.open
  const change = d.open ? ((d.close - d.open) / d.open * 100) : 0
  const changeStr = (change >= 0 ? '+' : '') + change.toFixed(2) + '%'

  return (
    <div className="chart-legend">
      <div className="legend-row legend-symbol-row">
        <span className="legend-pair">{pair || 'BTCUSDT'}</span>
        <span className="legend-sep">·</span>
        <span className="legend-tf">{timeframe || '15m'}</span>
      </div>
      <div className="legend-row legend-ohlc-row">
        <span className="legend-lbl">O</span>
        <span className={`legend-val ${isUp ? 'up' : 'dn'}`}>{fmtPrice(d.open)}</span>
        <span className="legend-lbl">H</span>
        <span className={`legend-val ${isUp ? 'up' : 'dn'}`}>{fmtPrice(d.high)}</span>
        <span className="legend-lbl">L</span>
        <span className={`legend-val ${isUp ? 'up' : 'dn'}`}>{fmtPrice(d.low)}</span>
        <span className="legend-lbl">C</span>
        <span className={`legend-val ${isUp ? 'up' : 'dn'}`}>{fmtPrice(d.close)}</span>
        <span className={`legend-change ${isUp ? 'up' : 'dn'}`}>{changeStr}</span>
      </div>
      {d.sma != null && (
        <div className="legend-row legend-ind-row">
          <span className="legend-dot" style={{ background: '#3b82f6' }} />
          <span className="legend-ind-name">SMA 200</span>
          <span className="legend-ind-val">{fmtPrice(d.sma)}</span>
        </div>
      )}
      <div className="legend-row legend-ind-row">
        <span className="legend-dot" style={{ background: 'rgba(0,217,126,0.5)' }} />
        <span className="legend-ind-name">Vol</span>
        <span className="legend-ind-val">{fmtVol(d.volume)}</span>
      </div>
    </div>
  )
}

export default ChartLegend
