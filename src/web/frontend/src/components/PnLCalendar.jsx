import React, { useMemo, useState } from 'react'

const MONTHS_FULL = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
const DAY_HEADERS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

const getColor = (bps, maxAbs) => {
  if (bps === 0 || maxAbs === 0) return 'transparent'
  const intensity = Math.min(Math.abs(bps) / maxAbs, 1)
  if (bps > 0) {
    const alpha = 0.15 + intensity * 0.85
    return `rgba(0, 217, 126, ${alpha.toFixed(2)})`
  }
  const alpha = 0.15 + intensity * 0.85
  return `rgba(230, 55, 87, ${alpha.toFixed(2)})`
}

function PnLCalendar({ mlTrades, mlAttnTrades, mlV3Trades }) {
  // Build daily PnL map
  const dailyMap = useMemo(() => {
    const allTrades = [...(mlTrades || []), ...(mlAttnTrades || []), ...(mlV3Trades || [])]
    if (allTrades.length === 0) return {}
    const daily = {}
    for (const t of allTrades) {
      if (!t.exit_time) continue
      const d = new Date(t.exit_time)
      if (isNaN(d.getTime())) continue
      const ist = new Date(d.getTime() + 5.5 * 60 * 60 * 1000)
      const key = ist.toISOString().slice(0, 10)
      if (!daily[key]) daily[key] = { bps: 0, count: 0 }
      daily[key].bps += t.net_profit_bps || 0
      daily[key].count++
    }
    return daily
  }, [mlTrades, mlAttnTrades, mlV3Trades])

  // Available months from trades
  const availableMonths = useMemo(() => {
    const months = new Set()
    for (const key of Object.keys(dailyMap)) {
      months.add(key.slice(0, 7)) // YYYY-MM
    }
    // Also add current month
    const now = new Date(Date.now() + 5.5 * 60 * 60 * 1000)
    months.add(now.toISOString().slice(0, 7))
    return [...months].sort()
  }, [dailyMap])

  // Default to current month
  const [selectedMonth, setSelectedMonth] = useState(() => {
    const now = new Date(Date.now() + 5.5 * 60 * 60 * 1000)
    return now.toISOString().slice(0, 7)
  })

  // Build calendar grid for selected month
  const { weeks, monthStats, maxAbs } = useMemo(() => {
    const [year, month] = selectedMonth.split('-').map(Number)
    const firstDay = new Date(year, month - 1, 1)
    const lastDay = new Date(year, month, 0) // last day of month
    const daysInMonth = lastDay.getDate()

    // Monday = 0, Sunday = 6 (ISO weekday)
    const firstWeekday = (firstDay.getDay() + 6) % 7 // convert Sun=0 to Mon=0

    // Collect all bps values for color scaling
    const allBps = []
    for (let d = 1; d <= daysInMonth; d++) {
      const key = `${selectedMonth}-${String(d).padStart(2, '0')}`
      if (dailyMap[key]) allBps.push(dailyMap[key].bps)
    }
    const maxAbsVal = allBps.length > 0 ? Math.max(...allBps.map(Math.abs), 1) : 1

    // Build weeks
    const weeksList = []
    let currentWeek = new Array(firstWeekday).fill(null) // pad start
    let weekBps = 0

    for (let d = 1; d <= daysInMonth; d++) {
      const key = `${selectedMonth}-${String(d).padStart(2, '0')}`
      const data = dailyMap[key] || null
      const todayStr = new Date(Date.now() + 5.5 * 60 * 60 * 1000).toISOString().slice(0, 10)

      currentWeek.push({
        day: d,
        date: key,
        bps: data ? data.bps : null,
        count: data ? data.count : 0,
        isToday: key === todayStr,
        isFuture: key > todayStr,
      })

      if (data) weekBps += data.bps

      // End of week (Sunday) or end of month
      if (currentWeek.length === 7 || d === daysInMonth) {
        // Pad end of last week
        while (currentWeek.length < 7) currentWeek.push(null)
        weeksList.push({ days: currentWeek, totalBps: weekBps })
        currentWeek = []
        weekBps = 0
      }
    }

    // Month stats
    let totalBps = 0, greenDays = 0, redDays = 0, tradeDays = 0
    for (let d = 1; d <= daysInMonth; d++) {
      const key = `${selectedMonth}-${String(d).padStart(2, '0')}`
      if (dailyMap[key]) {
        tradeDays++
        totalBps += dailyMap[key].bps
        if (dailyMap[key].bps > 0) greenDays++
        else if (dailyMap[key].bps < 0) redDays++
      }
    }

    return {
      weeks: weeksList,
      monthStats: { totalBps, greenDays, redDays, tradeDays },
      maxAbs: maxAbsVal,
    }
  }, [selectedMonth, dailyMap])

  const [year, month] = selectedMonth.split('-').map(Number)

  const prevMonth = () => {
    const d = new Date(year, month - 2, 1)
    setSelectedMonth(`${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}`)
  }

  const nextMonth = () => {
    const d = new Date(year, month, 1)
    setSelectedMonth(`${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}`)
  }

  const fmtBps = (v) => {
    if (v == null) return ''
    return `${v >= 0 ? '+' : ''}${v.toFixed(1)}`
  }

  return (
    <div className="card cal-section">
      {/* Header with month navigation */}
      <div className="cal-header">
        <button className="cal-nav-btn" onClick={prevMonth}>&lt;</button>
        <div className="cal-title">
          <span className="cal-month-name">{MONTHS_FULL[month - 1]} {year}</span>
          <span className="cal-summary">
            <span className="positive">{monthStats.greenDays}</span>
            {' / '}
            <span className="negative">{monthStats.redDays}</span>
            {' / '}
            {monthStats.tradeDays} days =
            <span className={monthStats.totalBps >= 0 ? 'positive' : 'negative'}>
              {' '}{fmtBps(monthStats.totalBps)} bps
            </span>
          </span>
        </div>
        <button className="cal-nav-btn" onClick={nextMonth}>&gt;</button>
      </div>

      {/* Day headers */}
      <div className="cal-day-headers">
        {DAY_HEADERS.map(d => (
          <div key={d} className="cal-day-hdr">{d}</div>
        ))}
        <div className="cal-day-hdr cal-week-hdr">Week</div>
      </div>

      {/* Calendar grid */}
      <div className="cal-month-grid">
        {weeks.map((week, wi) => (
          <div key={wi} className="cal-row">
            {week.days.map((cell, di) => {
              if (!cell) {
                return <div key={di} className="cal-day empty" />
              }
              const hasTrade = cell.bps !== null
              const bg = hasTrade ? getColor(cell.bps, maxAbs) : 'transparent'

              return (
                <div
                  key={di}
                  className={`cal-day ${hasTrade ? 'has-trade' : ''} ${cell.isToday ? 'today' : ''} ${cell.isFuture ? 'future' : ''}`}
                  style={{ background: bg }}
                  title={hasTrade ? `${cell.date}: ${fmtBps(cell.bps)} bps (${cell.count} trades)` : cell.date}
                >
                  <span className="cal-day-num">{cell.day}</span>
                  {hasTrade && (
                    <span className={`cal-day-bps ${cell.bps >= 0 ? 'positive' : 'negative'}`}>
                      {fmtBps(cell.bps)}
                    </span>
                  )}
                </div>
              )
            })}
            {/* Weekly total */}
            <div className={`cal-day cal-week-total ${week.totalBps >= 0 ? 'week-positive' : 'week-negative'}`}>
              {week.totalBps !== 0 ? fmtBps(week.totalBps) : ''}
            </div>
          </div>
        ))}
      </div>

      {/* Monthly total bar */}
      <div className={`cal-month-total ${monthStats.totalBps >= 0 ? 'positive' : 'negative'}`}>
        Month Total: {fmtBps(monthStats.totalBps)} bps
      </div>
    </div>
  )
}

export default PnLCalendar
