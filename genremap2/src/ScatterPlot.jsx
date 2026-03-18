// ScatterPlot.jsx — Shows actual boundary neighbor with genre deviation analysis
import { useEffect, useRef, useCallback, useState } from 'react'
import * as d3 from 'd3'
import { api } from './api'

const GENRE_COLORS = {
  'blues': '#00e5c3', 'classical': '#6e8efb', 'country': '#ffd166',
  'disco': '#e879f9', 'hiphop': '#f97316', 'jazz': '#a78bfa',
  'metal': '#ff5f7e', 'pop': '#fb7185', 'reggae': '#34d399',
  'rock': '#38bdf8', 'unknown': '#7a7a9a', 'uploaded': '#7a7a9a',
}

const MARGIN = { top: 24, right: 24, bottom: 24, left: 24 }

// Human-readable feature labels
const FEATURE_LABELS = {
  'tempo': 'Tempo', 'spectral_centroid_mean': 'Brightness',
  'spectral_bandwidth_mean': 'Freq spread', 'rolloff_mean': 'Spectral edge',
  'zero_crossing_rate_mean': 'Noisiness', 'rms_mean': 'Loudness',
  'rms_var': 'Dynamics', 'chroma_stft_mean': 'Harmony',
  'harmony_mean': 'Tonal energy', 'perceptr_mean': 'Percussion',
  'mfcc1_mean': 'Warmth', 'mfcc2_mean': 'Timbre', 'mfcc3_mean': 'Texture',
}

export default function ScatterPlot({ songs, selectedId, onSelect, highlightGenre, boundaryThreshold }) {
  const svgRef = useRef(null)
  const zoomRef = useRef(null)
  const audioRef = useRef(null)
  const [scaler, setScaler] = useState(null)
  const [genreMeans, setGenreMeans] = useState(null)

  // Fetch scaler and genre means on mount
  useEffect(() => {
    async function fetchData() {
      try {
        const [scalerData, meansData] = await Promise.all([
          api.getScaler(),
          api.getGenreMeans()
        ])
        setScaler(scalerData)
        setGenreMeans(meansData)
      } catch (e) {
        console.warn('Failed to fetch scaler/means:', e)
      }
    }
    fetchData()
  }, [])

  // Normalize a feature value
  const normalize = useCallback((value, featureIdx) => {
    if (!scaler) return value
    return (value - scaler.mean[featureIdx]) / scaler.scale[featureIdx]
  }, [scaler])

  // Compute top deviations from genre mean
  const computeGenreDeviations = useCallback((song, topN = 3) => {
    if (!song?.features || !scaler || !genreMeans || !genreMeans[song.genre]) return []
    
    const genreMean = genreMeans[song.genre]
    const diffs = []
    
    for (let i = 0; i < scaler.features.length; i++) {
      const featureName = scaler.features[i]
      const songVal = song.features[featureName]
      const genreMeanVal = genreMean[featureName]
      
      if (songVal === undefined || genreMeanVal === undefined) continue
      
      // Both are already in normalized space
      const songNorm = normalize(songVal, i)
      const deviation = songNorm - genreMeanVal
      
      diffs.push({
        name: featureName,
        label: FEATURE_LABELS[featureName] || featureName.replace(/_/g, ' '),
        deviation: deviation,
        absDeviation: Math.abs(deviation)
      })
    }
    
    return diffs.sort((a, b) => b.absDeviation - a.absDeviation).slice(0, topN)
  }, [scaler, genreMeans, normalize])

  // Compute top alignments (features where song is MOST typical for its genre)
  const computeGenreAlignments = useCallback((song, topN = 3) => {
    if (!song?.features || !scaler || !genreMeans || !genreMeans[song.genre]) return []
    
    const genreMean = genreMeans[song.genre]
    const alignments = []
    
    for (let i = 0; i < scaler.features.length; i++) {
      const featureName = scaler.features[i]
      const songVal = song.features[featureName]
      const genreMeanVal = genreMean[featureName]
      
      if (songVal === undefined || genreMeanVal === undefined) continue
      
      const songNorm = normalize(songVal, i)
      const deviation = Math.abs(songNorm - genreMeanVal)
      
      alignments.push({
        name: featureName,
        label: FEATURE_LABELS[featureName] || featureName.replace(/_/g, ' '),
        deviation: deviation
      })
    }
    
    // Sort by SMALLEST deviation (most aligned)
    return alignments.sort((a, b) => a.deviation - b.deviation).slice(0, topN)
  }, [scaler, genreMeans, normalize])

  const playAudio = useCallback((filename) => {
    if (audioRef.current) {
      audioRef.current.pause()
      audioRef.current.currentTime = 0
    }
    const audio = new Audio(`/api/audio/${filename}`)
    audio.loop = true
    audio.volume = 0.6
    audioRef.current = audio
    audio.play().catch(e => console.warn('Audio playback failed:', e))
  }, [])

  const stopAudio = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.pause()
      audioRef.current.currentTime = 0
      audioRef.current = null
    }
  }, [])

  useEffect(() => {
    if (selectedId !== null && selectedId !== undefined) {
      const song = songs.find(s => s.id === selectedId)
      if (song) playAudio(song.filename)
    } else {
      stopAudio()
    }
  }, [selectedId, songs, playAudio, stopAudio])

  useEffect(() => () => stopAudio(), [stopAudio])

  const draw = useCallback(() => {
    if (!songs.length || !svgRef.current) return

    const container = svgRef.current.parentElement
    const W = container.clientWidth
    const H = container.clientHeight
    const innerW = W - MARGIN.left - MARGIN.right
    const innerH = H - MARGIN.top - MARGIN.bottom

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

    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()
    svg.attr('width', W).attr('height', H)

    const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`)

    const gridG = g.append('g').attr('class', 'grid')
    gridG.append('g').attr('class', 'grid-x').attr('transform', `translate(0,${innerH})`)
      .call(d3.axisBottom(xScale).tickSize(-innerH).tickFormat('').ticks(8))
    gridG.append('g').attr('class', 'grid-y')
      .call(d3.axisLeft(yScale).tickSize(-innerW).tickFormat('').ticks(8))

    svg.selectAll('.grid line').style('stroke', '#1e1e2a').style('stroke-width', '1px')
    svg.selectAll('.grid .domain').remove()

    const zoomG = g.append('g').attr('class', 'zoom-layer')

    const zoom = d3.zoom().scaleExtent([0.4, 12])
      .on('zoom', e => zoomG.attr('transform', e.transform))

    svg.call(zoom).on('click', e => {
      if (e.target === svg.node() || e.target.tagName === 'rect') onSelect(null)
    })

    zoomRef.current = { zoom, svg }

    // Get ACTUAL boundary neighbor from backend data
    const selectedSong = songs.find(s => s.id === selectedId)
    const boundaryNeighbor = selectedSong && selectedSong.boundary_neighbor_id >= 0
      ? songs.find(s => s.id === selectedSong.boundary_neighbor_id)
      : null

    // Draw connection line
    if (boundaryNeighbor) {
      zoomG.append('line')
        .attr('x1', xScale(selectedSong.x)).attr('y1', yScale(selectedSong.y))
        .attr('x2', xScale(boundaryNeighbor.x)).attr('y2', yScale(boundaryNeighbor.y))
        .attr('stroke', '#fff').attr('stroke-width', 2)
        .attr('stroke-dasharray', '5,5').attr('opacity', 0.6)
        .style('pointer-events', 'none')
      
      const midX = (selectedSong.x + boundaryNeighbor.x) / 2
      const midY = (selectedSong.y + boundaryNeighbor.y) / 2
      
      zoomG.append('text')
        .attr('x', xScale(midX)).attr('y', yScale(midY))
        .attr('text-anchor', 'middle').attr('dy', -8)
        .style('font-family', 'var(--font-mono)').style('font-size', '10px')
        .style('fill', '#fff').style('pointer-events', 'none')
        .style('text-shadow', '0 0 4px #000, 0 0 8px #000')
        .text('boundary neighbor')
    }

    const dots = zoomG.selectAll('circle')
      .data(songs, d => d.id)
      .join('circle')
      .attr('cx', d => xScale(d.x)).attr('cy', d => yScale(d.y))
      .attr('r', d => (d.id === selectedId || (boundaryNeighbor && d.id === boundaryNeighbor.id)) ? 7 : 4)
      .attr('fill', d => GENRE_COLORS[d.genre] || GENRE_COLORS['unknown'])
      .attr('fill-opacity', d => {
        const isFiltered = boundaryThreshold !== 10 && d.boundary_score > boundaryThreshold
        if (isFiltered) return 0.08
        if (highlightGenre !== null && d.genre !== highlightGenre) return 0.08
        if (d.id === selectedId || (boundaryNeighbor && d.id === boundaryNeighbor.id)) return 1
        return 0.72
      })
      .attr('stroke', d => (d.id === selectedId || (boundaryNeighbor && d.id === boundaryNeighbor.id)) ? '#fff' : GENRE_COLORS[d.genre])
      .attr('stroke-width', d => (d.id === selectedId || (boundaryNeighbor && d.id === boundaryNeighbor.id)) ? 2 : 0.5)
      .attr('stroke-opacity', d => (d.id === selectedId || (boundaryNeighbor && d.id === boundaryNeighbor.id)) ? 1 : 0.3)
      .style('cursor', 'pointer')

    d3.select(svgRef.current.parentElement).select('.gm-tooltip').remove()
    const tooltip = d3.select(svgRef.current.parentElement).append('div')
      .attr('class', 'gm-tooltip')
      .style('min-width', '260px')
      .style('max-width', '500px')

    dots.on('mouseenter', function (event, d) {
        d3.select(this).attr('r', 7).attr('fill-opacity', 1).attr('stroke-width', 2).attr('stroke', '#fff')

        const [px, py] = d3.pointer(event, svg.node())
        
        let content = `
          <div class="tt-filename">${d.filename}</div>
          <div class="tt-meta" style="color:${GENRE_COLORS[d.genre]}">${d.genre}</div>
        `
        
        if (d.id === selectedId) {
          content += '<div class="tt-audio">🔊 playing...</div>'
        }
        
        // If this is the boundary neighbor, show comparison with selected song
        if (boundaryNeighbor && d.id === boundaryNeighbor.id) {
          content += '<div class="tt-boundary-sep"></div>'
          content += '<div class="tt-boundary-title">⚡ Boundary Neighbor</div>'
          
          // Show how SELECTED SONG deviates from its own genre mean
          const selectedDeviations = computeGenreDeviations(selectedSong, 3)
          const selectedAlignments = computeGenreAlignments(selectedSong, 3)
          if (selectedDeviations.length > 0 || selectedAlignments.length > 0) {
            content += `<div class="tt-diffs-title">${selectedSong.genre} song:</div>`
            content += '<div style="display: flex; gap: 20px;">'
            
            // Left column: deviations
            if (selectedDeviations.length > 0) {
              content += '<div style="flex: 1;">'
              content += '<div style="font-size: 10px; opacity: 0.6; margin-bottom: 4px;">Differs in:</div>'
              selectedDeviations.forEach(dev => {
                const sign = dev.deviation > 0 ? '+' : ''
                const color = dev.deviation > 0 ? 'var(--accent)' : '#6e8efb'
                content += `
                  <div class="tt-diff-row">
                    <span class="tt-diff-label">${dev.label}</span>
                    <span class="tt-diff-value" style="color:${color}">${sign}${dev.deviation.toFixed(1)}σ</span>
                  </div>
                `
              })
              content += '</div>'
            }
            
            // Right column: alignments
            if (selectedAlignments.length > 0) {
              content += '<div style="flex: 1;">'
              content += '<div style="font-size: 10px; opacity: 0.6; margin-bottom: 4px;">Aligned in:</div>'
              selectedAlignments.forEach(align => {
                content += `
                  <div class="tt-diff-row">
                    <span class="tt-diff-label">${align.label}</span>
                    <span class="tt-diff-value" style="color:#4ade80">✓</span>
                  </div>
                `
              })
              content += '</div>'
            }
            
            content += '</div>'
          }
          
          // Show how THIS BOUNDARY NEIGHBOR deviates from its own genre mean
          content += '<div class="tt-boundary-sep"></div>'
          const neighborDeviations = computeGenreDeviations(d, 3)
          const neighborAlignments = computeGenreAlignments(d, 3)
          if (neighborDeviations.length > 0 || neighborAlignments.length > 0) {
            content += `<div class="tt-diffs-title">${d.genre} song:</div>`
            content += '<div style="display: flex; gap: 20px;">'
            
            // Left column: deviations
            if (neighborDeviations.length > 0) {
              content += '<div style="flex: 1;">'
              content += '<div style="font-size: 10px; opacity: 0.6; margin-bottom: 4px;">Differs in:</div>'
              neighborDeviations.forEach(dev => {
                const sign = dev.deviation > 0 ? '+' : ''
                const color = dev.deviation > 0 ? 'var(--accent)' : '#6e8efb'
                content += `
                  <div class="tt-diff-row">
                    <span class="tt-diff-label">${dev.label}</span>
                    <span class="tt-diff-value" style="color:${color}">${sign}${dev.deviation.toFixed(1)}σ</span>
                  </div>
                `
              })
              content += '</div>'
            }
            
            // Right column: alignments
            if (neighborAlignments.length > 0) {
              content += '<div style="flex: 1;">'
              content += '<div style="font-size: 10px; opacity: 0.6; margin-bottom: 4px;">Aligned in:</div>'
              neighborAlignments.forEach(align => {
                content += `
                  <div class="tt-diff-row">
                    <span class="tt-diff-label">${align.label}</span>
                    <span class="tt-diff-value" style="color:#4ade80">✓</span>
                  </div>
                `
              })
              content += '</div>'
            }
            
            content += '</div>'
          }
        } else if (d.id !== selectedId) {
          // For non-selected, non-boundary songs, show their genre deviations and alignments
          const deviations = computeGenreDeviations(d, 3)
          const alignments = computeGenreAlignments(d, 3)
          if (deviations.length > 0 || alignments.length > 0) {
            content += '<div class="tt-boundary-sep"></div>'
            content += `<div class="tt-diffs-title">${d.genre} song:</div>`
            content += '<div style="display: flex; gap: 20px;">'
            
            // Left column: deviations
            if (deviations.length > 0) {
              content += '<div style="flex: 1;">'
              content += '<div style="font-size: 10px; opacity: 0.6; margin-bottom: 4px;">Differs in:</div>'
              deviations.forEach(dev => {
                const sign = dev.deviation > 0 ? '+' : ''
                const color = dev.deviation > 0 ? 'var(--accent)' : '#6e8efb'
                content += `
                  <div class="tt-diff-row">
                    <span class="tt-diff-label">${dev.label}</span>
                    <span class="tt-diff-value" style="color:${color}">${sign}${dev.deviation.toFixed(1)}σ</span>
                  </div>
                `
              })
              content += '</div>'
            }
            
            // Right column: alignments
            if (alignments.length > 0) {
              content += '<div style="flex: 1;">'
              content += '<div style="font-size: 10px; opacity: 0.6; margin-bottom: 4px;">Aligned in:</div>'
              alignments.forEach(align => {
                content += `
                  <div class="tt-diff-row">
                    <span class="tt-diff-label">${align.label}</span>
                    <span class="tt-diff-value" style="color:#4ade80">✓</span>
                  </div>
                `
              })
              content += '</div>'
            }
            
            content += '</div>'
          }
        }
        
        if (d.boundary_score != null) {
          content += `<div class="tt-meta">Boundary: ${d.boundary_score.toFixed(2)}</div>`
        }
        
        tooltip.style('opacity', 1).style('left', `${px + 14}px`).style('top', `${py - 10}px`).html(content)
      })
      .on('mouseleave', function (_, d) {
        d3.select(this)
          .attr('r', (d.id === selectedId || (boundaryNeighbor && d.id === boundaryNeighbor.id)) ? 7 : 4)
          .attr('fill-opacity', () => {
            const isFiltered = boundaryThreshold !== 10 && d.boundary_score > boundaryThreshold
            if (isFiltered) return 0.08
            if (highlightGenre !== null && d.genre !== highlightGenre) return 0.08
            if (d.id === selectedId || (boundaryNeighbor && d.id === boundaryNeighbor.id)) return 1
            return 0.72
          })
          .attr('stroke-width', (d.id === selectedId || (boundaryNeighbor && d.id === boundaryNeighbor.id)) ? 2 : 0.5)
          .attr('stroke', (d.id === selectedId || (boundaryNeighbor && d.id === boundaryNeighbor.id)) ? '#fff' : GENRE_COLORS[d.genre])
        tooltip.style('opacity', 0)
      })
      .on('click', (event, d) => { event.stopPropagation(); onSelect(d) })

  }, [songs, selectedId, highlightGenre, onSelect, boundaryThreshold, computeGenreDeviations])

  useEffect(() => { draw() }, [draw])

  useEffect(() => {
    const ro = new ResizeObserver(() => draw())
    if (svgRef.current?.parentElement) ro.observe(svgRef.current.parentElement)
    return () => ro.disconnect()
  }, [draw])

  return <svg ref={svgRef} style={{ display: 'block', width: '100%', height: '100%' }} />
}