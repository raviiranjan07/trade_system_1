/**
 * Moving Average calculations for dynamic indicator overlay.
 * Supports SMA, EMA, WMA, DEMA with configurable source (close/open/high/low/hl2/hlc3/ohlc4).
 */

export const getSource = (candle, source) => {
  switch (source) {
    case 'open': return candle.open
    case 'high': return candle.high
    case 'low': return candle.low
    case 'hl2': return (candle.high + candle.low) / 2
    case 'hlc3': return (candle.high + candle.low + candle.close) / 3
    case 'ohlc4': return (candle.open + candle.high + candle.low + candle.close) / 4
    default: return candle.close
  }
}

export const calcSMA = (candles, period, source) => {
  const result = []
  if (candles.length < period) return result
  let sum = 0
  for (let i = 0; i < period; i++) {
    sum += getSource(candles[i], source)
  }
  result.push({ time: candles[period - 1].time, value: sum / period })
  for (let i = period; i < candles.length; i++) {
    sum += getSource(candles[i], source) - getSource(candles[i - period], source)
    result.push({ time: candles[i].time, value: sum / period })
  }
  return result
}

export const calcEMA = (candles, period, source) => {
  if (candles.length < period) return []
  const k = 2 / (period + 1)
  let sum = 0
  for (let i = 0; i < period; i++) {
    sum += getSource(candles[i], source)
  }
  let ema = sum / period
  const result = [{ time: candles[period - 1].time, value: ema }]
  for (let i = period; i < candles.length; i++) {
    ema = getSource(candles[i], source) * k + ema * (1 - k)
    result.push({ time: candles[i].time, value: ema })
  }
  return result
}

export const calcWMA = (candles, period, source) => {
  const result = []
  if (candles.length < period) return result
  const denom = period * (period + 1) / 2
  for (let i = period - 1; i < candles.length; i++) {
    let sum = 0
    for (let j = 0; j < period; j++) {
      sum += getSource(candles[i - period + 1 + j], source) * (j + 1)
    }
    result.push({ time: candles[i].time, value: sum / denom })
  }
  return result
}

export const calcDEMA = (candles, period, source) => {
  const ema1 = calcEMA(candles, period, source)
  if (ema1.length < period) return []
  const k = 2 / (period + 1)
  let sum = 0
  for (let i = 0; i < period; i++) {
    sum += ema1[i].value
  }
  let ema2 = sum / period
  const result = [{
    time: ema1[period - 1].time,
    value: 2 * ema1[period - 1].value - ema2,
  }]
  for (let i = period; i < ema1.length; i++) {
    ema2 = ema1[i].value * k + ema2 * (1 - k)
    result.push({ time: ema1[i].time, value: 2 * ema1[i].value - ema2 })
  }
  return result
}

export const calculateMA = (type, candles, period, source = 'close') => {
  switch (type) {
    case 'EMA': return calcEMA(candles, period, source)
    case 'WMA': return calcWMA(candles, period, source)
    case 'DEMA': return calcDEMA(candles, period, source)
    default: return calcSMA(candles, period, source)
  }
}
