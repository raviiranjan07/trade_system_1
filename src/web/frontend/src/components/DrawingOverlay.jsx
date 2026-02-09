import React, { useEffect, useRef, useCallback, useState } from 'react'

const ONE_POINT = ['hline', 'vline', 'text']
const TWO_POINT = ['trendline', 'ray', 'extended', 'rect', 'fib', 'measure']
const THREE_POINT = ['channel']

const COLORS = {
  hline: '#2962FF', vline: '#7B1FA2', trendline: '#FF6D00', ray: '#FF6D00',
  extended: '#FF6D00', rect: '#2962FF', channel: '#00897B', fib: '#FF6D00',
  measure: '#9C27B0', text: '#E0E0E0',
}

const FIB_LEVELS = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
const FIB_LABELS = ['0%', '23.6%', '38.2%', '50%', '61.8%', '78.6%', '100%']

function ptLineDist(px, py, p1, p2) {
  const dx = p2.x - p1.x, dy = p2.y - p1.y
  const len2 = dx * dx + dy * dy
  if (len2 === 0) return Math.hypot(px - p1.x, py - p1.y)
  let t = ((px - p1.x) * dx + (py - p1.y) * dy) / len2
  t = Math.max(0, Math.min(1, t))
  return Math.hypot(px - (p1.x + t * dx), py - (p1.y + t * dy))
}

function hitTest(d, point, toPixel, threshold) {
  const px = d.points.map(p => toPixel(p.time, p.price)).filter(Boolean)
  if (d.type === 'hline' && px[0]) return Math.abs(point.y - px[0].y) < threshold
  if (d.type === 'vline' && px[0]) return Math.abs(point.x - px[0].x) < threshold
  if (['trendline', 'ray', 'extended'].includes(d.type) && px[0] && px[1])
    return ptLineDist(point.x, point.y, px[0], px[1]) < threshold
  if (['rect', 'measure'].includes(d.type) && px[0] && px[1]) {
    const x1 = Math.min(px[0].x, px[1].x), x2 = Math.max(px[0].x, px[1].x)
    const y1 = Math.min(px[0].y, px[1].y), y2 = Math.max(px[0].y, px[1].y)
    return point.x >= x1 - threshold && point.x <= x2 + threshold &&
           point.y >= y1 - threshold && point.y <= y2 + threshold
  }
  if (['fib', 'channel'].includes(d.type)) {
    for (const p of px) {
      if (Math.hypot(point.x - p.x, point.y - p.y) < threshold * 2) return true
    }
    if (d.type === 'channel' && px[0] && px[1] && ptLineDist(point.x, point.y, px[0], px[1]) < threshold) return true
  }
  if (d.type === 'text' && px[0]) return Math.hypot(point.x - px[0].x, point.y - px[0].y) < threshold * 2
  return false
}

function DrawingOverlay({ chart, series, chartContainerEl, activeTool, drawings, onAddDrawing, onRemoveDrawing, onUpdateDrawing, onCancel }) {
  const canvasRef = useRef(null)
  const frameRef = useRef(null)
  const pendingPointsRef = useRef([])
  const mousePriceRef = useRef(null)
  const drawingsRef = useRef(drawings)
  const activeToolRef = useRef(activeTool)
  const selectedIdRef = useRef(null)
  const [selectedId, setSelectedId] = useState(null)

  // Drag state
  const isDraggingRef = useRef(false)
  const dragInfoRef = useRef(null)
  const dragCleanupRef = useRef(null)
  const lastCursorRef = useRef('')

  useEffect(() => { drawingsRef.current = drawings }, [drawings])
  useEffect(() => {
    activeToolRef.current = activeTool
    pendingPointsRef.current = []
    mousePriceRef.current = null
    if (activeTool !== 'pointer') { setSelectedId(null); setCursorOnChart('') }
  }, [activeTool])
  useEffect(() => { selectedIdRef.current = selectedId }, [selectedId])

  // Set cursor on chart canvases
  const setCursorOnChart = useCallback((cursor) => {
    if (cursor === lastCursorRef.current) return
    lastCursorRef.current = cursor
    if (!chartContainerEl) return
    const canvases = chartContainerEl.querySelectorAll('canvas')
    for (const c of canvases) c.style.cursor = cursor
  }, [chartContainerEl])

  // Find chart pane bounds (largest canvas = main chart area)
  const getPaneBounds = useCallback(() => {
    if (!chartContainerEl) return null
    const allCanvas = chartContainerEl.querySelectorAll('canvas')
    if (allCanvas.length === 0) return null
    let main = null, maxA = 0
    for (const c of allCanvas) {
      const r = c.getBoundingClientRect()
      const a = r.width * r.height
      if (a > maxA) { maxA = a; main = c }
    }
    if (!main) return null
    const parent = chartContainerEl.parentElement || chartContainerEl
    const pr = parent.getBoundingClientRect()
    const cr = main.getBoundingClientRect()
    return { left: cr.left - pr.left, top: cr.top - pr.top, width: cr.width, height: cr.height, screenLeft: cr.left, screenTop: cr.top }
  }, [chartContainerEl])

  // Get time/pixel mapping from visible range (for extrapolation beyond data)
  const getTimePixelRatio = useCallback(() => {
    if (!chart) return null
    try {
      const ts = chart.timeScale()
      const vr = ts.getVisibleRange()
      if (!vr || vr.from === vr.to) return null
      const fromPx = ts.timeToCoordinate(vr.from)
      const toPx = ts.timeToCoordinate(vr.to)
      if (fromPx == null || toPx == null || toPx === fromPx) return null
      return { fromTime: vr.from, toTime: vr.to, fromPx, toPx }
    } catch (e) { return null }
  }, [chart])

  // time -> pixel x (works beyond data range via extrapolation)
  const timeToX = useCallback((time) => {
    if (!chart) return null
    try {
      const x = chart.timeScale().timeToCoordinate(time)
      if (x != null) return x
    } catch (e) {}
    const r = getTimePixelRatio()
    if (!r) return null
    const pxPerTime = (r.toPx - r.fromPx) / (r.toTime - r.fromTime)
    return r.fromPx + (time - r.fromTime) * pxPerTime
  }, [chart, getTimePixelRatio])

  // pixel x -> time (works beyond data range via extrapolation)
  const xToTime = useCallback((x) => {
    if (!chart) return null
    try {
      const t = chart.timeScale().coordinateToTime(x)
      if (t != null) return t
    } catch (e) {}
    const r = getTimePixelRatio()
    if (!r) return null
    const timePerPx = (r.toTime - r.fromTime) / (r.toPx - r.fromPx)
    return Math.round(r.fromTime + (x - r.fromPx) * timePerPx)
  }, [chart, getTimePixelRatio])

  const toPixel = useCallback((time, price) => {
    if (!chart || !series) return null
    const x = timeToX(time)
    let y = null
    try { y = series.priceToCoordinate(price) } catch (e) {}
    if (x == null || y == null) return null
    return { x, y }
  }, [chart, series, timeToX])

  // Convert pane pixel coords to {time, price}
  const paneToTP = useCallback((paneX, paneY) => {
    const time = xToTime(paneX)
    let price = null
    try { price = series.coordinateToPrice(paneY) } catch (e) {}
    if (time == null || price == null) return null
    return { time, price }
  }, [series, xToTime])

  // Extract time+price from chart event param
  const extractTP = useCallback((param) => {
    if (!param || !param.point) return null
    let time = xToTime(param.point.x)
    if (time == null) time = param.time
    if (time == null) return null
    let price = null
    try { price = series.coordinateToPrice(param.point.y) } catch (e) {}
    if (price == null) return null
    return { time, price }
  }, [series, xToTime])

  const drawAnchor = (ctx, x, y, sel) => {
    ctx.save()
    ctx.fillStyle = sel ? '#2962FF' : '#fff'
    ctx.strokeStyle = sel ? '#fff' : '#555'
    ctx.lineWidth = 1.5
    ctx.beginPath()
    ctx.arc(x, y, sel ? 5 : 3, 0, Math.PI * 2)
    ctx.fill()
    ctx.stroke()
    ctx.restore()
  }

  const drawShape = useCallback((ctx, d, bounds, sel) => {
    const { type, points, color } = d
    const px = points.map(p => toPixel(p.time, p.price))
    const c = color || COLORS[type] || '#2962FF'
    ctx.save()
    ctx.strokeStyle = c
    ctx.fillStyle = c
    ctx.lineWidth = sel ? 2.5 : 1.5
    ctx.setLineDash([])
    ctx.font = '11px Inter, sans-serif'
    ctx.textAlign = 'left'

    switch (type) {
      case 'hline': {
        if (!px[0]) break
        ctx.beginPath(); ctx.moveTo(0, px[0].y); ctx.lineTo(bounds.width, px[0].y); ctx.stroke()
        const lbl = points[0].price.toFixed(1)
        ctx.font = '10px monospace'
        const tw = ctx.measureText(lbl).width
        ctx.fillStyle = c; ctx.globalAlpha = 0.85
        ctx.fillRect(bounds.width - tw - 10, px[0].y - 9, tw + 8, 18)
        ctx.globalAlpha = 1; ctx.fillStyle = '#fff'
        ctx.fillText(lbl, bounds.width - tw - 6, px[0].y + 4)
        if (sel) drawAnchor(ctx, bounds.width / 2, px[0].y, true)
        break
      }
      case 'vline': {
        if (!px[0]) break
        ctx.beginPath(); ctx.moveTo(px[0].x, 0); ctx.lineTo(px[0].x, bounds.height); ctx.stroke()
        if (sel) drawAnchor(ctx, px[0].x, bounds.height / 2, true)
        break
      }
      case 'trendline': {
        if (!px[0] || !px[1]) break
        ctx.beginPath(); ctx.moveTo(px[0].x, px[0].y); ctx.lineTo(px[1].x, px[1].y); ctx.stroke()
        drawAnchor(ctx, px[0].x, px[0].y, sel)
        drawAnchor(ctx, px[1].x, px[1].y, sel)
        break
      }
      case 'ray': {
        if (!px[0] || !px[1]) break
        const dx = px[1].x - px[0].x, dy = px[1].y - px[0].y
        const len = Math.hypot(dx, dy)
        if (len === 0) break
        const s = Math.max(bounds.width, bounds.height) * 3 / len
        ctx.beginPath(); ctx.moveTo(px[0].x, px[0].y); ctx.lineTo(px[0].x + dx * s, px[0].y + dy * s); ctx.stroke()
        drawAnchor(ctx, px[0].x, px[0].y, sel)
        drawAnchor(ctx, px[1].x, px[1].y, sel)
        break
      }
      case 'extended': {
        if (!px[0] || !px[1]) break
        const dx = px[1].x - px[0].x, dy = px[1].y - px[0].y
        const len = Math.hypot(dx, dy)
        if (len === 0) break
        const s = Math.max(bounds.width, bounds.height) * 3 / len
        ctx.beginPath()
        ctx.moveTo(px[0].x - dx * s, px[0].y - dy * s)
        ctx.lineTo(px[0].x + dx * s, px[0].y + dy * s)
        ctx.stroke()
        drawAnchor(ctx, px[0].x, px[0].y, sel)
        drawAnchor(ctx, px[1].x, px[1].y, sel)
        break
      }
      case 'rect': {
        if (!px[0] || !px[1]) break
        const x1 = Math.min(px[0].x, px[1].x), y1 = Math.min(px[0].y, px[1].y)
        const w = Math.abs(px[1].x - px[0].x), h = Math.abs(px[1].y - px[0].y)
        ctx.globalAlpha = 0.12; ctx.fillRect(x1, y1, w, h)
        ctx.globalAlpha = 1; ctx.strokeRect(x1, y1, w, h)
        if (sel) {
          drawAnchor(ctx, px[0].x, px[0].y, true)
          drawAnchor(ctx, px[1].x, px[1].y, true)
        }
        break
      }
      case 'channel': {
        if (!px[0] || !px[1]) break
        ctx.beginPath(); ctx.moveTo(px[0].x, px[0].y); ctx.lineTo(px[1].x, px[1].y); ctx.stroke()
        if (px[2]) {
          const dx = px[1].x - px[0].x, dy = px[1].y - px[0].y
          const len2 = dx * dx + dy * dy
          if (len2 > 0) {
            const nx = -dy, ny = dx, nLen = Math.sqrt(len2)
            const ox = px[2].x - px[0].x, oy = px[2].y - px[0].y
            const dist = (ox * nx + oy * ny) / nLen
            const offX = (nx / nLen) * dist, offY = (ny / nLen) * dist
            ctx.setLineDash([5, 3])
            ctx.beginPath(); ctx.moveTo(px[0].x + offX, px[0].y + offY); ctx.lineTo(px[1].x + offX, px[1].y + offY); ctx.stroke()
            ctx.setLineDash([])
            ctx.globalAlpha = 0.06
            ctx.beginPath()
            ctx.moveTo(px[0].x, px[0].y); ctx.lineTo(px[1].x, px[1].y)
            ctx.lineTo(px[1].x + offX, px[1].y + offY); ctx.lineTo(px[0].x + offX, px[0].y + offY)
            ctx.closePath(); ctx.fill()
            ctx.globalAlpha = 1
          }
        }
        if (sel) px.filter(Boolean).forEach(p => drawAnchor(ctx, p.x, p.y, true))
        break
      }
      case 'fib': {
        if (!px[0] || !px[1]) break
        const p1 = points[0].price, p2 = points[1].price
        const xL = Math.min(px[0].x, px[1].x), xR = bounds.width
        for (let i = 0; i < FIB_LEVELS.length; i++) {
          const price = p1 + (p2 - p1) * FIB_LEVELS[i]
          const fp = toPixel(points[0].time, price)
          if (!fp) continue
          ctx.globalAlpha = (i === 0 || i === FIB_LEVELS.length - 1) ? 0.9 : 0.5
          ctx.strokeStyle = c
          ctx.setLineDash(FIB_LEVELS[i] === 0.5 ? [4, 4] : [])
          ctx.beginPath(); ctx.moveTo(xL, fp.y); ctx.lineTo(xR, fp.y); ctx.stroke()
          ctx.fillStyle = c; ctx.globalAlpha = 0.85
          ctx.font = '10px Inter, sans-serif'
          ctx.fillText(`${FIB_LABELS[i]}  ${price.toFixed(0)}`, xL + 6, fp.y - 4)
        }
        const f382 = toPixel(points[0].time, p1 + (p2 - p1) * 0.382)
        const f618 = toPixel(points[0].time, p1 + (p2 - p1) * 0.618)
        if (f382 && f618) {
          ctx.globalAlpha = 0.06; ctx.fillStyle = c
          ctx.fillRect(xL, Math.min(f382.y, f618.y), xR - xL, Math.abs(f618.y - f382.y))
        }
        ctx.globalAlpha = 1; ctx.setLineDash([])
        if (sel) { drawAnchor(ctx, px[0].x, px[0].y, true); drawAnchor(ctx, px[1].x, px[1].y, true) }
        break
      }
      case 'measure': {
        if (!px[0] || !px[1]) break
        const x1 = Math.min(px[0].x, px[1].x), y1 = Math.min(px[0].y, px[1].y)
        const w = Math.abs(px[1].x - px[0].x), h = Math.abs(px[1].y - px[0].y)
        const pDiff = points[1].price - points[0].price
        const isUp = pDiff >= 0
        ctx.fillStyle = isUp ? 'rgba(0,217,126,0.08)' : 'rgba(230,55,87,0.08)'
        ctx.fillRect(x1, y1, w, h)
        ctx.strokeStyle = isUp ? '#00d97e' : '#e63757'
        ctx.setLineDash([3, 3]); ctx.strokeRect(x1, y1, w, h); ctx.setLineDash([])
        const pct = (pDiff / points[0].price) * 100
        const bps = Math.abs(pct * 100)
        const cx = x1 + w / 2, cy = y1 + h / 2
        const lw = 150, lh = 48
        ctx.fillStyle = 'rgba(10,14,20,0.92)'
        ctx.fillRect(cx - lw / 2, cy - lh / 2, lw, lh)
        ctx.strokeStyle = isUp ? '#00d97e' : '#e63757'
        ctx.lineWidth = 1; ctx.strokeRect(cx - lw / 2, cy - lh / 2, lw, lh)
        ctx.textAlign = 'center'
        ctx.fillStyle = isUp ? '#00d97e' : '#e63757'
        ctx.font = 'bold 13px Inter, sans-serif'
        ctx.fillText(`${isUp ? '+' : ''}${pDiff.toFixed(1)}`, cx, cy - 6)
        ctx.font = '11px Inter, sans-serif'
        ctx.fillText(`${isUp ? '+' : ''}${pct.toFixed(2)}%  (${bps.toFixed(0)} bps)`, cx, cy + 12)
        ctx.textAlign = 'left'; ctx.lineWidth = 1.5
        break
      }
      case 'text': {
        if (!px[0]) break
        ctx.fillStyle = c
        ctx.font = '14px Inter, sans-serif'
        ctx.fillText(d.text || '', px[0].x, px[0].y)
        if (sel) drawAnchor(ctx, px[0].x, px[0].y + 6, true)
        break
      }
    }
    ctx.restore()
  }, [toPixel])

  // Main redraw
  const redraw = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas || !chart || !series) return
    const bounds = getPaneBounds()
    if (!bounds) return
    const dpr = window.devicePixelRatio || 1
    const nw = Math.round(bounds.width * dpr), nh = Math.round(bounds.height * dpr)
    if (canvas.width !== nw || canvas.height !== nh) {
      canvas.width = nw; canvas.height = nh
      canvas.style.width = bounds.width + 'px'
      canvas.style.height = bounds.height + 'px'
    }
    canvas.style.left = bounds.left + 'px'
    canvas.style.top = bounds.top + 'px'
    const ctx = canvas.getContext('2d')
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, bounds.width, bounds.height)

    for (const d of drawingsRef.current) {
      drawShape(ctx, d, bounds, d.id === selectedIdRef.current)
    }

    const pending = pendingPointsRef.current
    const mouse = mousePriceRef.current
    const tool = activeToolRef.current
    if (pending.length > 0 && mouse && tool && tool !== 'pointer' && tool !== 'eraser') {
      ctx.globalAlpha = 0.55
      drawShape(ctx, { type: tool, points: [...pending, mouse], color: COLORS[tool] }, bounds, false)
      ctx.globalAlpha = 1
    }
  }, [chart, series, getPaneBounds, drawShape])

  const scheduleRedraw = useCallback(() => {
    if (frameRef.current) cancelAnimationFrame(frameRef.current)
    frameRef.current = requestAnimationFrame(redraw)
  }, [redraw])

  // Chart scroll/zoom
  useEffect(() => {
    if (!chart) return
    const h = () => scheduleRedraw()
    chart.timeScale().subscribeVisibleLogicalRangeChange(h)
    return () => { chart.timeScale().unsubscribeVisibleLogicalRangeChange(h); if (frameRef.current) cancelAnimationFrame(frameRef.current) }
  }, [chart, scheduleRedraw])

  useEffect(() => { scheduleRedraw() }, [drawings, activeTool, selectedId, scheduleRedraw])

  // Delayed redraws: drawings loaded from localStorage may fail toPixel()
  // if chart data hasn't arrived yet. Retry a few times to catch it.
  useEffect(() => {
    if (!chart || !series || drawingsRef.current.length === 0) return
    const timers = [200, 600, 1500, 3000].map(ms =>
      setTimeout(scheduleRedraw, ms)
    )
    return () => timers.forEach(clearTimeout)
  }, [chart, series, scheduleRedraw])

  useEffect(() => {
    if (!chartContainerEl) return
    const ro = new ResizeObserver(() => scheduleRedraw())
    ro.observe(chartContainerEl)
    return () => ro.disconnect()
  }, [chartContainerEl, scheduleRedraw])

  useEffect(() => {
    const h = () => setTimeout(scheduleRedraw, 100)
    document.addEventListener('fullscreenchange', h)
    return () => document.removeEventListener('fullscreenchange', h)
  }, [scheduleRedraw])

  // ========== DRAG/MOVE SUPPORT ==========
  useEffect(() => {
    if (!chart || !series || !chartContainerEl) return

    const handlePointerDown = (e) => {
      if (activeToolRef.current !== 'pointer') return
      if (!selectedIdRef.current) return

      const bounds = getPaneBounds()
      if (!bounds) return

      // Convert screen coords to chart pane coords
      const paneX = e.clientX - bounds.screenLeft
      const paneY = e.clientY - bounds.screenTop

      // Check within pane
      if (paneX < 0 || paneY < 0 || paneX > bounds.width || paneY > bounds.height) return

      const sel = drawingsRef.current.find(d => d.id === selectedIdRef.current)
      if (!sel) return

      // Check near anchor point (single anchor drag)
      const anchors = sel.points.map(p => toPixel(p.time, p.price)).filter(Boolean)
      let anchorIdx = -1
      for (let i = 0; i < anchors.length; i++) {
        if (Math.hypot(paneX - anchors[i].x, paneY - anchors[i].y) < 12) {
          anchorIdx = i
          break
        }
      }

      // If not near anchor, check if near the shape body (full move)
      if (anchorIdx === -1) {
        if (hitTest(sel, { x: paneX, y: paneY }, toPixel, 12)) {
          anchorIdx = -2 // move all
        }
      }

      if (anchorIdx === -1) return // not on the drawing

      // Intercept! Prevent chart from panning
      e.stopPropagation()
      e.preventDefault()

      isDraggingRef.current = true
      dragInfoRef.current = {
        id: sel.id,
        anchorIdx,
        startPoints: sel.points.map(p => ({ ...p })),
        startPaneX: paneX,
        startPaneY: paneY,
      }
      document.body.style.cursor = 'grabbing'

      const handleDragMove = (e2) => {
        if (!isDraggingRef.current) return
        const info = dragInfoRef.current
        if (!info) return

        const b = getPaneBounds()
        if (!b) return
        const mx = e2.clientX - b.screenLeft
        const my = e2.clientY - b.screenTop

        if (info.anchorIdx === -2) {
          // Move all points by pixel delta
          const dx = mx - info.startPaneX
          const dy = my - info.startPaneY
          const newPoints = info.startPoints.map(p => {
            const pp = toPixel(p.time, p.price)
            if (!pp) return p
            const tp = paneToTP(pp.x + dx, pp.y + dy)
            if (!tp) return p
            return tp
          })
          onUpdateDrawing(info.id, newPoints)
        } else {
          // Move single anchor
          const tp = paneToTP(mx, my)
          if (!tp) return
          const newPoints = info.startPoints.map((p, i) =>
            i === info.anchorIdx ? tp : { ...p }
          )
          onUpdateDrawing(info.id, newPoints)
        }
        scheduleRedraw()
      }

      const handleDragEnd = () => {
        isDraggingRef.current = false
        dragInfoRef.current = null
        document.body.style.cursor = ''
        window.removeEventListener('pointermove', handleDragMove)
        window.removeEventListener('pointerup', handleDragEnd)
        dragCleanupRef.current = null
      }

      window.addEventListener('pointermove', handleDragMove)
      window.addEventListener('pointerup', handleDragEnd)
      dragCleanupRef.current = handleDragEnd
    }

    // Capture phase: intercept before chart's own handlers
    chartContainerEl.addEventListener('pointerdown', handlePointerDown, true)
    return () => {
      chartContainerEl.removeEventListener('pointerdown', handlePointerDown, true)
      if (dragCleanupRef.current) { dragCleanupRef.current(); dragCleanupRef.current = null }
    }
  }, [chart, series, chartContainerEl, getPaneBounds, toPixel, paneToTP, onUpdateDrawing, scheduleRedraw])

  // Click handler (selection + drawing placement)
  useEffect(() => {
    if (!chart || !series) return
    const handleClick = (param) => {
      if (isDraggingRef.current) return // ignore click after drag
      const tool = activeToolRef.current
      if (!tool) return
      const tp = extractTP(param)
      const point = param.point

      // Pointer: select drawing
      if (tool === 'pointer') {
        if (!point || !tp) { setSelectedId(null); setCursorOnChart(''); return }
        for (const d of [...drawingsRef.current].reverse()) {
          if (hitTest(d, point, toPixel, 15)) { setSelectedId(d.id); return }
        }
        setSelectedId(null)
        setCursorOnChart('')
        return
      }

      if (!tp) return
      const { time, price } = tp

      if (tool === 'eraser') {
        if (!point) return
        for (const d of [...drawingsRef.current].reverse()) {
          if (hitTest(d, point, toPixel, 15)) { onRemoveDrawing(d.id); return }
        }
        return
      }

      if (tool === 'text') {
        const text = window.prompt('Enter text:')
        if (text) onAddDrawing({ id: Date.now() + Math.random(), type: 'text', points: [{ time, price }], color: COLORS.text, text })
        return
      }

      const newPending = [...pendingPointsRef.current, { time, price }]
      const needed = THREE_POINT.includes(tool) ? 3 : TWO_POINT.includes(tool) ? 2 : 1
      if (newPending.length >= needed) {
        onAddDrawing({ id: Date.now() + Math.random(), type: tool, points: newPending, color: COLORS[tool] })
        pendingPointsRef.current = []
      } else {
        pendingPointsRef.current = newPending
      }
      scheduleRedraw()
    }
    chart.subscribeClick(handleClick)
    return () => chart.unsubscribeClick(handleClick)
  }, [chart, series, extractTP, toPixel, setCursorOnChart, onAddDrawing, onRemoveDrawing, scheduleRedraw])

  // Crosshair move: preview + cursor feedback
  useEffect(() => {
    if (!chart || !series) return
    const handleMove = (param) => {
      const tool = activeToolRef.current

      // Cursor feedback in pointer mode
      if (tool === 'pointer' && selectedIdRef.current != null && param.point && !isDraggingRef.current) {
        const sel = drawingsRef.current.find(d => d.id === selectedIdRef.current)
        if (sel) {
          // Check near anchor
          const anchors = sel.points.map(p => toPixel(p.time, p.price)).filter(Boolean)
          let nearAnchor = false
          for (const a of anchors) {
            if (Math.hypot(param.point.x - a.x, param.point.y - a.y) < 12) { nearAnchor = true; break }
          }
          if (nearAnchor) setCursorOnChart('grab')
          else if (hitTest(sel, param.point, toPixel, 12)) setCursorOnChart('move')
          else setCursorOnChart('')
        }
      } else if (tool === 'pointer') {
        setCursorOnChart('')
      }

      // Preview for drawing tools
      if (!tool || tool === 'pointer' || tool === 'eraser') return
      if (pendingPointsRef.current.length === 0) return
      const tp = extractTP(param)
      if (!tp) { mousePriceRef.current = null; return }
      mousePriceRef.current = tp
      scheduleRedraw()
    }
    chart.subscribeCrosshairMove(handleMove)
    return () => chart.unsubscribeCrosshairMove(handleMove)
  }, [chart, series, extractTP, toPixel, setCursorOnChart, scheduleRedraw])

  // Keyboard
  useEffect(() => {
    const handler = (e) => {
      if (e.key === 'Escape') {
        pendingPointsRef.current = []
        mousePriceRef.current = null
        setSelectedId(null)
        setCursorOnChart('')
        scheduleRedraw()
        if (onCancel) onCancel()
      }
      if ((e.key === 'Delete' || e.key === 'Backspace') && selectedIdRef.current != null) {
        e.preventDefault()
        onRemoveDrawing(selectedIdRef.current)
        setSelectedId(null)
        setCursorOnChart('')
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [onCancel, onRemoveDrawing, setCursorOnChart, scheduleRedraw])

  return <canvas ref={canvasRef} className="drawing-canvas" />
}

export default DrawingOverlay
