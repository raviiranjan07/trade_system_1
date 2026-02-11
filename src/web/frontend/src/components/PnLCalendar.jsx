import React, { useMemo } from 'react'

// Color scale for PnL: green for positive, red for negative, intensity by magnitude
const getColor = (bps, maxAbs) => {
  if (bps === 0 || maxAbs === 0) return 'var(--bg-elevated)'
  const intensity = Math.min(Math.abs(bps) / maxAbs, 1)
  if (bps > 0) {
    // Green scale: dim to bright
    const alpha = 0.15 + intensity * 0.85
    return `rgba(0, 217, 126, ${alpha.toFixed(2)})`
  }
  // Red scale: dim to bright
  const alpha = 0.15 + intensity * 0.85
  return `rgba(230, 55, 87, ${alpha.toFixed(2)})`
}

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
const DAYS = ['Mon', '', 'Wed', '', 'Fri', '', '']

function PnLCalendar({ trades }) {
  const { weeks, dailyMap, maxAbs, monthLabels } = useMemo(() => {
    if (!trades || trades.length === 0) return { weeks: [], dailyMap: {}, maxAbs: 0, monthLabels: [] }

    // Sort oldest first
    const sorted = [...trades].reverse()

    // Group by date (IST offset applied - trades have exit_time as ISO string)
    const daily = {}
    for (const t of sorted) {
      if (!t.exit_time) continue
      const d = new Date(t.exit_time)
      if (isNaN(d.getTime())) continue
      // Add IST offset for display
      const ist = new Date(d.getTime() + 5.5 * 60 * 60 * 1000)
      const key = ist.toISOString().slice(0, 10) // YYYY-MM-DD
      if (!daily[key]) daily[key] = 0
      daily[key] += t.net_profit_bps || 0
    }

    const dates = Object.keys(daily).sort()
    if (dates.length === 0) return { weeks: [], dailyMap: {}, maxAbs: 0, monthLabels: [] }

    // Find date range: from first trade's Monday to today
    const firstDate = new Date(dates[0] + 'T00:00:00')
    const today = new Date()
    const todayStr = today.toISOString().slice(0, 10)

    // Go back to Monday of first week
    const startDay = firstDate.getDay() // 0=Sun
    const mondayOffset = startDay === 0 ? 6 : startDay - 1
    const startDate = new Date(firstDate)
    startDate.setDate(startDate.getDate() - mondayOffset)

    // Build weeks grid
    const weeksList = []
    const months = []
    let current = new Date(startDate)
    let weekIdx = 0
    let lastMonth = -1

    while (current <= today || weeksList.length % 1 !== 0) {
      const week = []
      for (let d = 0; d < 7; d++) {
        const dateStr = current.toISOString().slice(0, 10)
        const month = current.getMonth()
        if (month !== lastMonth && d === 0) {
          months.push({ weekIdx, month })
          lastMonth = month
        }
        week.push({
          date: dateStr,
          bps: daily[dateStr] || 0,
          hasData: dateStr in daily,
          isFuture: dateStr > todayStr,
        })
        current.setDate(current.getDate() + 1)
      }
      weeksList.push(week)
      weekIdx++
      if (current > today && current.getDay() === 1) break // finish on a Monday boundary after today
    }

    // Max absolute for color scaling
    const vals = Object.values(daily)
    const maxAbsVal = Math.max(...vals.map(Math.abs), 1)

    return { weeks: weeksList, dailyMap: daily, maxAbs: maxAbsVal, monthLabels: months }
  }, [trades])

  if (weeks.length === 0) {
    return (
      <div className="card cal-section">
        <div className="card-header">PnL Calendar</div>
        <div className="perf-empty">No trades yet</div>
      </div>
    )
  }

  // Summary stats
  const totalDays = Object.keys(dailyMap).length
  const greenDays = Object.values(dailyMap).filter(v => v > 0).length
  const redDays = Object.values(dailyMap).filter(v => v < 0).length
  const totalBps = Object.values(dailyMap).reduce((s, v) => s + v, 0)

  return (
    <div className="card cal-section">
      <div className="card-header">
        PnL Calendar
        <span className="cal-summary">
          <span className="positive">{greenDays}</span> green /
          <span className="negative"> {redDays}</span> red /
          {totalDays} days =
          <span className={totalBps >= 0 ? 'positive' : 'negative'}>
            {' '}{totalBps >= 0 ? '+' : ''}{totalBps.toFixed(1)} bps
          </span>
        </span>
      </div>

      <div className="cal-container">
        {/* Day labels */}
        <div className="cal-day-labels">
          {DAYS.map((d, i) => (
            <div key={i} className="cal-day-label">{d}</div>
          ))}
        </div>

        <div className="cal-grid-wrapper">
          {/* Month labels */}
          <div className="cal-month-labels">
            {monthLabels.map((m, i) => (
              <div
                key={i}
                className="cal-month-label"
                style={{ left: m.weekIdx * 15 }}
              >
                {MONTHS[m.month]}
              </div>
            ))}
          </div>

          {/* Grid */}
          <div className="cal-grid">
            {weeks.map((week, wi) => (
              <div key={wi} className="cal-week">
                {week.map((day, di) => (
                  <div
                    key={di}
                    className={`cal-cell ${day.isFuture ? 'future' : ''} ${day.hasData ? 'has-data' : ''}`}
                    style={{
                      background: day.isFuture ? 'transparent' : day.hasData ? getColor(day.bps, maxAbs) : 'var(--bg-card)',
                      borderColor: day.isFuture ? 'transparent' : 'var(--border-color)',
                    }}
                    title={day.isFuture ? '' : `${day.date}: ${day.hasData ? (day.bps >= 0 ? '+' : '') + day.bps.toFixed(1) + ' bps' : 'No trades'}`}
                  />
                ))}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Legend */}
      <div className="cal-legend">
        <span className="cal-legend-label">Less</span>
        <div className="cal-legend-cell" style={{ background: 'rgba(230, 55, 87, 0.85)' }} />
        <div className="cal-legend-cell" style={{ background: 'rgba(230, 55, 87, 0.4)' }} />
        <div className="cal-legend-cell" style={{ background: 'var(--bg-card)' }} />
        <div className="cal-legend-cell" style={{ background: 'rgba(0, 217, 126, 0.4)' }} />
        <div className="cal-legend-cell" style={{ background: 'rgba(0, 217, 126, 0.85)' }} />
        <span className="cal-legend-label">More</span>
      </div>
    </div>
  )
}

export default PnLCalendar
