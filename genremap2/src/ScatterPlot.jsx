// ScatterPlot.jsx — D3-powered 2D PCA visualization (supervised)
import { useEffect, useRef, useCallback } from 'react'
import * as d3 from 'd3'

const GENRE_COLORS = {
  'blues':     '#00e5c3',
  'classical': '#6e8efb',
  'country':   '#ffd166',
  'disco':     '#e879f9',
  'hiphop':    '#f97316',
  'jazz':      '#a78bfa',
  'metal':     '#ff5f7e',
  'pop':       '#fb7185',
  'reggae':    '#34d399',
  'rock':      '#38bdf8',
  'unknown':   '#7a7a9a',
}

const MARGIN = { top: 24, right: 24, bottom: 24, left: 24 }

export default function ScatterPlot({ songs, selectedId, onSelect, highlightGenre }) {
  const svgRef = useRef(null)
  const zoomRef = useRef(null)
  const audioRef = useRef(null)  // Audio element for playback

  // ── Audio playback helpers ────────────────────────────────────────────────
  const playAudio = useCallback((filename) => {
    // Stop any currently playing audio
    if (audioRef.current) {
      audioRef.current.pause()
      audioRef.current.currentTime = 0
    }

    // Create new audio element
    const audio = new Audio(`/api/audio/${filename}`)
    audio.loop = true  // Loop the 30s clip
    audio.volume = 0.6
    audioRef.current = audio

    audio.play().catch(e => {
      console.warn('Audio playback failed:', e)
    })
  }, [])

  const stopAudio = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.pause()
      audioRef.current.currentTime = 0
      audioRef.current = null
    }
  }, [])

  // Play/stop audio based on selection changes
  useEffect(() => {
    if (selectedId !== null && selectedId !== undefined) {
      const song = songs.find(s => s.id === selectedId)
      if (song) {
        playAudio(song.filename)
      }
    } else {
      stopAudio()
    }
  }, [selectedId, songs, playAudio, stopAudio])

  // Cleanup on unmount
  useEffect(() => {
    return () => stopAudio()
  }, [stopAudio])

  const draw = useCallback(() => {
    if (!songs.length || !svgRef.current) return

    const container = svgRef.current.parentElement
    const W = container.clientWidth
    const H = container.clientHeight
    const innerW = W - MARGIN.left - MARGIN.right
    const innerH = H - MARGIN.top - MARGIN.bottom

    // ── Scales ────────────────────────────────────────────────────────────────
    const xExtent = d3.extent(songs, d => d.x)
    const yExtent = d3.extent(songs, d => d.y)
    const padding = 0.08

    const xScale = d3.scaleLinear()
      .domain([xExtent[0] - padding * (xExtent[1] - xExtent[0]),
               xExtent[1] + padding * (xExtent[1] - xExtent[0])])
      .range([0, innerW])

    const yScale = d3.scaleLinear()
      .domain([yExtent[0] - padding * (yExtent[1] - yExtent[0]),
               yExtent[1] + padding * (yExtent[1] - yExtent[0])])
      .range([innerH, 0])

    // ── SVG setup ─────────────────────────────────────────────────────────────
    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()
    svg.attr('width', W).attr('height', H)

    // Subtle grid
    const g = svg.append('g')
      .attr('transform', `translate(${MARGIN.left},${MARGIN.top})`)

    const gridG = g.append('g').attr('class', 'grid')
    gridG.append('g')
      .attr('class', 'grid-x')
      .attr('transform', `translate(0,${innerH})`)
      .call(
        d3.axisBottom(xScale).tickSize(-innerH).tickFormat('').ticks(8)
      )
    gridG.append('g')
      .attr('class', 'grid-y')
      .call(
        d3.axisLeft(yScale).tickSize(-innerW).tickFormat('').ticks(8)
      )

    svg.selectAll('.grid line')
      .style('stroke', '#1e1e2a')
      .style('stroke-width', '1px')
    svg.selectAll('.grid .domain').remove()

    // ── Zoom ──────────────────────────────────────────────────────────────────
    const zoomG = g.append('g').attr('class', 'zoom-layer')

    const zoom = d3.zoom()
      .scaleExtent([0.4, 12])
      .on('zoom', (event) => {
        zoomG.attr('transform', event.transform)
      })

    // Click on background to deselect
    svg.call(zoom)
      .on('click', (event) => {
        // Only deselect if clicking directly on the background (not a dot)
        if (event.target === svg.node() || event.target.tagName === 'rect') {
          onSelect(null)
        }
      })

    zoomRef.current = { zoom, svg }

    // ── Dots ──────────────────────────────────────────────────────────────────
    const dots = zoomG.selectAll('circle')
      .data(songs, d => d.id)
      .join('circle')
      .attr('cx', d => xScale(d.x))
      .attr('cy', d => yScale(d.y))
      .attr('r', d => d.id === selectedId ? 7 : 4)
      .attr('fill', d => GENRE_COLORS[d.genre] || GENRE_COLORS['unknown'])
      .attr('fill-opacity', d => {
        if (highlightGenre !== null && d.genre !== highlightGenre) return 0.08
        if (d.id === selectedId) return 1
        return 0.72
      })
      .attr('stroke', d => d.id === selectedId
        ? '#fff'
        : GENRE_COLORS[d.genre] || GENRE_COLORS['unknown'])
      .attr('stroke-width', d => d.id === selectedId ? 2 : 0.5)
      .attr('stroke-opacity', d => d.id === selectedId ? 1 : 0.3)
      .style('cursor', 'pointer')
      .style('transition', 'fill-opacity 0.2s, r 0.2s')

    dots.on('mouseenter', function (event, d) {
        d3.select(this)
          .attr('r', 7)
          .attr('fill-opacity', 1)
          .attr('stroke-width', 2)
          .attr('stroke', '#fff')

        // Tooltip (no audio on hover anymore)
        const [px, py] = d3.pointer(event, svg.node())
        tooltip
          .style('opacity', 1)
          .style('left', `${px + 14}px`)
          .style('top', `${py - 10}px`)
          .html(`
            <div class="tt-filename">${d.filename}</div>
            <div class="tt-meta">Genre: ${d.genre}</div>
            ${d.id === selectedId ? '<div class="tt-audio">🔊 playing...</div>' : ''}
          `)
      })
      .on('mouseleave', function (_, d) {
        d3.select(this)
          .attr('r', d.id === selectedId ? 7 : 4)
          .attr('fill-opacity', () => {
            if (highlightGenre !== null && d.genre !== highlightGenre) return 0.08
            return d.id === selectedId ? 1 : 0.72
          })
          .attr('stroke-width', d.id === selectedId ? 2 : 0.5)
          .attr('stroke', d.id === selectedId ? '#fff'
            : GENRE_COLORS[d.genre] || GENRE_COLORS['unknown'])

        tooltip.style('opacity', 0)
      })
      .on('click', (event, d) => {
        event.stopPropagation()  // Prevent background click from firing
        onSelect(d)
      })

    // ── Tooltip ───────────────────────────────────────────────────────────────
    d3.select(svgRef.current.parentElement).select('.gm-tooltip').remove()
    const tooltip = d3.select(svgRef.current.parentElement)
      .append('div')
      .attr('class', 'gm-tooltip')

  }, [songs, selectedId, highlightGenre, onSelect])

  // Redraw when data or selection changes
  useEffect(() => { draw() }, [draw])

  // Redraw on resize
  useEffect(() => {
    const ro = new ResizeObserver(() => draw())
    if (svgRef.current?.parentElement) ro.observe(svgRef.current.parentElement)
    return () => ro.disconnect()
  }, [draw])

  return (
    <svg
      ref={svgRef}
      style={{ display: 'block', width: '100%', height: '100%' }}
    />
  )
}
