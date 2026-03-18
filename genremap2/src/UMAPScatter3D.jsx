// UMAPScatter3D.jsx — UMAP explorer with 2D + 3D views and info panel
import { useEffect, useRef, useState, useCallback } from 'react'
import * as THREE from 'three'
import * as d3 from 'd3'
import styles from './UMAPScatter3D.module.css'
import { api } from './api'

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

const VIEWS = [
  { id: 'umap2d', label: '2D' },
  { id: 'umap3d', label: '3D' },
]

const FEATURE_GROUPS = [
  {
    group: 'Rhythm',
    features: [
      { name: 'tempo', desc: 'Beats per minute — how fast the song is' },
      { name: 'zero_crossing_rate_mean', desc: 'How often the audio signal crosses zero — higher in noisy or percussive sounds' },
      { name: 'zero_crossing_rate_var', desc: 'How much the zero crossing rate varies throughout the song' },
    ]
  },
  {
    group: 'Energy & Loudness',
    features: [
      { name: 'rms_mean', desc: 'Average loudness/energy of the song' },
      { name: 'rms_var', desc: 'How much the loudness fluctuates — high in dynamic songs, low in consistent ones' },
      { name: 'harmony_mean', desc: 'Average energy in the harmonic (tonal) part of the signal' },
      { name: 'harmony_var', desc: 'How much the harmonic content varies' },
      { name: 'perceptr_mean', desc: 'Average energy in the percussive part of the signal' },
      { name: 'perceptr_var', desc: 'How much the percussive content varies' },
    ]
  },
  {
    group: 'Spectral Shape',
    features: [
      { name: 'spectral_centroid_mean', desc: 'Average "brightness" — where the center of mass of the frequency spectrum sits. High = bright/trebly, low = dark/bassy' },
      { name: 'spectral_centroid_var', desc: 'How much the brightness changes over time' },
      { name: 'spectral_bandwidth_mean', desc: 'Average spread of frequencies around the centroid — wider means more complex sound' },
      { name: 'spectral_bandwidth_var', desc: 'How much the frequency spread varies' },
      { name: 'rolloff_mean', desc: 'The frequency below which 85% of the energy sits — distinguishes bright vs dark sounds' },
      { name: 'rolloff_var', desc: 'How much the rolloff frequency varies' },
      { name: 'chroma_stft_mean', desc: 'Average distribution of energy across the 12 musical pitch classes (C, C#, D...) — captures harmony and key' },
      { name: 'chroma_stft_var', desc: 'How much the pitch class distribution changes — high in harmonically varied songs' },
    ]
  },
  {
    group: 'Timbre (MFCCs)',
    description: 'Mel-Frequency Cepstral Coefficients capture the texture and timbre of sound — what makes a guitar sound different from a violin. Each coefficient captures a different aspect of tonal character. Lower-numbered MFCCs describe broad tonal shape; higher-numbered ones describe finer texture details. Each has a mean (average value) and var (how much it varies).',
    features: Array.from({ length: 20 }, (_, i) => ({
      name: `mfcc${i+1}_mean / mfcc${i+1}_var`,
      desc: i === 0 ? 'Overall tonal brightness/darkness' :
            i === 1 ? 'Broad spectral shape' :
            i < 5 ? 'Mid-level tonal characteristics' :
            i < 10 ? 'Mid-level timbral texture' :
            'Fine-grained timbral detail'
    }))
  }
]

function hexToThreeColor(hex) { return new THREE.Color(hex) }

// ── Info Panel ────────────────────────────────────────────────────────────────
function InfoPanel({ onClose }) {
  const [tab, setTab] = useState('umap')
  return (
    <div className={styles.infoPanel}>
      <div className={styles.infoPanelHeader}>
        <div className={styles.infoPanelTabs}>
          <button className={`${styles.infoTab} ${tab === 'umap' ? styles.infoTabActive : ''}`} onClick={() => setTab('umap')}>What is UMAP?</button>
          <button className={`${styles.infoTab} ${tab === 'features' ? styles.infoTabActive : ''}`} onClick={() => setTab('features')}>Feature Glossary</button>
        </div>
        <button className={styles.infoPanelClose} onClick={onClose}>✕</button>
      </div>
      <div className={styles.infoPanelBody}>
        {tab === 'umap' && (
          <div className={styles.umapExplainer}>
            <p>
              <strong>UMAP</strong> (Uniform Manifold Approximation and Projection) is a dimensionality reduction technique that compresses high-dimensional data down to 2 or 3 dimensions for visualization.
            </p>
            <p>
              Each song in this dataset is described by <strong>57 audio features</strong> — things like tempo, brightness, and timbral texture. UMAP takes all 57 of those numbers and finds a 3D layout where <strong>songs that sound similar end up close together</strong>, and songs that sound different end up far apart.
            </p>
            <p>
              Unlike the PCA/LDA map, UMAP does not try to maximize separation between genres — it purely tries to preserve acoustic similarity. This means you may see unexpected neighbors: a jazz song sitting near a blues song, or a pop song near a disco song, because they genuinely share acoustic characteristics.
            </p>
            <div className={styles.umapViewExplainer}>
              <div className={styles.umapViewBox}>
                <span className={styles.umapViewLabel}>2D view</span>
                <span>A flat scatter plot of the first two UMAP dimensions. Easy to read at a glance. Zoom and pan to explore clusters.</span>
              </div>
              <div className={styles.umapViewBox}>
                <span className={styles.umapViewLabel}>3D view</span>
                <span>The full 3D layout you can rotate freely. Use this to see structure that the 2D view might flatten out.</span>
              </div>
            </div>
          </div>
        )}
        {tab === 'features' && (
          <div className={styles.glossary}>
            {FEATURE_GROUPS.map(group => (
              <div key={group.group} className={styles.glossaryGroup}>
                <div className={styles.glossaryGroupName}>{group.group}</div>
                {group.description && (
                  <div className={styles.glossaryGroupDesc}>{group.description}</div>
                )}
                {group.features.map(f => (
                  <div key={f.name} className={styles.glossaryRow}>
                    <span className={styles.glossaryFeature}>{f.name}</span>
                    <span className={styles.glossaryDesc}>{f.desc}</span>
                  </div>
                ))}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

// ── 2D Pairplot ───────────────────────────────────────────────────────────────
function PairPlot({ songs, highlightGenre, selectedId, onSelect }) {
  const svgRef = useRef(null)

  const draw = useCallback(() => {
    if (!songs.length || !svgRef.current) return
    const container = svgRef.current.parentElement
    const W = container.clientWidth
    const H = container.clientHeight
    const M = { top: 20, right: 20, bottom: 40, left: 48 }
    const iW = W - M.left - M.right
    const iH = H - M.top - M.bottom

    const xExt = d3.extent(songs, d => d.x3d)
    const yExt = d3.extent(songs, d => d.y3d)
    const pad = 0.08

    const xScale = d3.scaleLinear()
      .domain([xExt[0] - pad*(xExt[1]-xExt[0]), xExt[1] + pad*(xExt[1]-xExt[0])])
      .range([0, iW])
    const yScale = d3.scaleLinear()
      .domain([yExt[0] - pad*(yExt[1]-yExt[0]), yExt[1] + pad*(yExt[1]-yExt[0])])
      .range([iH, 0])

    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()
    svg.attr('width', W).attr('height', H)

    const g = svg.append('g').attr('transform', `translate(${M.left},${M.top})`)

    g.append('g').call(d3.axisBottom(xScale).tickSize(-iH).tickFormat('').ticks(6))
      .attr('transform', `translate(0,${iH})`)
      .selectAll('line').style('stroke', '#1e1e2a').style('stroke-width', '1px')
    g.append('g').call(d3.axisLeft(yScale).tickSize(-iW).tickFormat('').ticks(6))
      .selectAll('line').style('stroke', '#1e1e2a').style('stroke-width', '1px')
    svg.selectAll('.domain').remove()

    g.append('text')
      .attr('x', iW / 2).attr('y', iH + 32)
      .attr('text-anchor', 'middle')
      .attr('font-family', 'Space Mono, monospace')
      .attr('font-size', 10).attr('fill', '#44445a')
      .text('UMAP 1')
    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -iH / 2).attr('y', -36)
      .attr('text-anchor', 'middle')
      .attr('font-family', 'Space Mono, monospace')
      .attr('font-size', 10).attr('fill', '#44445a')
      .text('UMAP 2')

    const zoomG = g.append('g')
    const zoom = d3.zoom().scaleExtent([0.4, 12])
      .on('zoom', e => zoomG.attr('transform', e.transform))
    svg.call(zoom).on('click', e => {
      if (e.target === svg.node()) onSelect(null)
    })

    d3.select(svgRef.current.parentElement).select('.gm-tooltip-umap').remove()
    const tooltip = d3.select(svgRef.current.parentElement)
      .append('div').attr('class', 'gm-tooltip gm-tooltip-umap')

    zoomG.selectAll('circle')
      .data(songs, d => d.id)
      .join('circle')
      .attr('cx', d => xScale(d.x3d))
      .attr('cy', d => yScale(d.y3d))
      .attr('r', d => d.id === selectedId ? 7 : 4)
      .attr('fill', d => GENRE_COLORS[d.genre] || GENRE_COLORS['unknown'])
      .attr('fill-opacity', d => {
        if (highlightGenre !== null && d.genre !== highlightGenre) return 0.08
        return d.id === selectedId ? 1 : 0.72
      })
      .attr('stroke', d => d.id === selectedId ? '#fff' : GENRE_COLORS[d.genre] || GENRE_COLORS['unknown'])
      .attr('stroke-width', d => d.id === selectedId ? 2 : 0.5)
      .attr('stroke-opacity', d => d.id === selectedId ? 1 : 0.3)
      .style('cursor', 'pointer')
      .on('mouseenter', function(event, d) {
        d3.select(this).attr('r', 7).attr('fill-opacity', 1).attr('stroke', '#fff').attr('stroke-width', 2)
        const [px, py] = d3.pointer(event, svg.node())
        tooltip.style('opacity', 1).style('left', `${px+14}px`).style('top', `${py-10}px`)
          .html(`<div class="tt-filename">${d.filename}</div><div class="tt-meta" style="color:${GENRE_COLORS[d.genre]}">${d.genre}</div><div class="tt-audio">${d.id === selectedId ? '🔊 playing...' : 'Click to play'}</div>`)
      })
      .on('mouseleave', function(_, d) {
        d3.select(this)
          .attr('r', d.id === selectedId ? 7 : 4)
          .attr('fill-opacity', highlightGenre !== null && d.genre !== highlightGenre ? 0.08 : d.id === selectedId ? 1 : 0.72)
          .attr('stroke', d.id === selectedId ? '#fff' : GENRE_COLORS[d.genre] || GENRE_COLORS['unknown'])
          .attr('stroke-width', d.id === selectedId ? 2 : 0.5)
        tooltip.style('opacity', 0)
      })
      .on('click', (event, d) => { event.stopPropagation(); onSelect(d) })
  }, [songs, highlightGenre, selectedId, onSelect])

  useEffect(() => { draw() }, [draw])
  useEffect(() => {
    const ro = new ResizeObserver(() => draw())
    if (svgRef.current?.parentElement) ro.observe(svgRef.current.parentElement)
    return () => ro.disconnect()
  }, [draw])

  return <svg ref={svgRef} style={{ display: 'block', width: '100%', height: '100%' }} />
}

// ── 3D View ───────────────────────────────────────────────────────────────────
function View3D({ songs, highlightGenre, selectedId, onSelect }) {
  const mountRef = useRef(null)
  const rendererRef = useRef(null)
  const frameRef = useRef(null)
  const isDragging = useRef(false)
  const lastMouse = useRef({ x: 0, y: 0 })
  const spherical = useRef({ theta: 0.4, phi: 1.1, radius: 5 })
  const pointsRef = useRef(null)
  const songDataRef = useRef([])
  const cameraRef = useRef(null)
  const [hovered, setHovered] = useState(null)
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 })

  useEffect(() => {
    if (!songs.length || !mountRef.current) return
    const W = mountRef.current.clientWidth
    const H = mountRef.current.clientHeight

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    renderer.setSize(W, H)
    renderer.setPixelRatio(window.devicePixelRatio)
    renderer.setClearColor(0x000000, 0)
    mountRef.current.appendChild(renderer.domElement)
    rendererRef.current = renderer

    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(50, W / H, 0.01, 100)
    cameraRef.current = camera

    const xs = songs.map(s => s.x3d), ys = songs.map(s => s.y3d), zs = songs.map(s => s.z3d)
    const cx = (Math.max(...xs)+Math.min(...xs))/2
    const cy = (Math.max(...ys)+Math.min(...ys))/2
    const cz = (Math.max(...zs)+Math.min(...zs))/2
    const range = Math.max(Math.max(...xs)-Math.min(...xs), Math.max(...ys)-Math.min(...ys), Math.max(...zs)-Math.min(...zs))/2||1
    const norm = (v,c) => (v-c)/range

    songDataRef.current = songs.map(s => ({ ...s, nx: norm(s.x3d,cx), ny: norm(s.y3d,cy), nz: norm(s.z3d,cz) }))

    const positions = [], colors = []
    for (const s of songDataRef.current) {
      positions.push(s.nx, s.ny, s.nz)
      const c = hexToThreeColor(GENRE_COLORS[s.genre]||GENRE_COLORS['unknown'])
      colors.push(c.r, c.g, c.b)
    }

    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
    geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3))
    const mat = new THREE.PointsMaterial({ size: 0.035, vertexColors: true, sizeAttenuation: true, transparent: true, opacity: 0.85 })
    const points = new THREE.Points(geo, mat)
    scene.add(points)
    pointsRef.current = points

    const axMat = new THREE.LineBasicMaterial({ color: 0x2a2a38, transparent: true, opacity: 0.5 })
    for (const [s,e] of [[[-1.5,0,0],[1.5,0,0]],[[0,-1.5,0],[0,1.5,0]],[[0,0,-1.5],[0,0,1.5]]]) {
      const g = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(...s), new THREE.Vector3(...e)])
      scene.add(new THREE.Line(g, axMat))
    }

    const updateCam = () => {
      const {theta,phi,radius} = spherical.current
      camera.position.set(radius*Math.sin(phi)*Math.sin(theta), radius*Math.cos(phi), radius*Math.sin(phi)*Math.cos(theta))
      camera.lookAt(0,0,0)
    }
    const animate = () => { frameRef.current = requestAnimationFrame(animate); updateCam(); renderer.render(scene,camera) }
    animate()

    const ro = new ResizeObserver(() => {
      if (!mountRef.current) return
      const w = mountRef.current.clientWidth, h = mountRef.current.clientHeight
      renderer.setSize(w,h); camera.aspect=w/h; camera.updateProjectionMatrix()
    })
    ro.observe(mountRef.current)

    return () => {
      cancelAnimationFrame(frameRef.current); ro.disconnect(); renderer.dispose()
      if (mountRef.current && renderer.domElement.parentNode === mountRef.current)
        mountRef.current.removeChild(renderer.domElement)
    }
  }, [songs])

  useEffect(() => {
    if (!pointsRef.current || !songDataRef.current.length) return
    const colors = []
    for (const s of songDataRef.current) {
      const c = hexToThreeColor(GENRE_COLORS[s.genre]||GENRE_COLORS['unknown'])
      const dimmed = highlightGenre !== null && s.genre !== highlightGenre
      colors.push(dimmed?c.r*0.15:c.r, dimmed?c.g*0.15:c.g, dimmed?c.b*0.15:c.b)
    }
    pointsRef.current.geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors,3))
    pointsRef.current.geometry.attributes.color.needsUpdate = true
  }, [highlightGenre])

  const onMouseDown = useCallback(e => { isDragging.current=true; lastMouse.current={x:e.clientX,y:e.clientY} }, [])
  const onMouseUp = useCallback(() => { isDragging.current=false }, [])
  const onWheel = useCallback(e => { spherical.current.radius=Math.max(2,Math.min(12,spherical.current.radius+e.deltaY*0.005)) }, [])

  const onMouseMove = useCallback(e => {
    setMousePos({x:e.clientX,y:e.clientY})
    if (isDragging.current) {
      const dx=e.clientX-lastMouse.current.x, dy=e.clientY-lastMouse.current.y
      lastMouse.current={x:e.clientX,y:e.clientY}
      spherical.current.theta -= dx*0.008
      spherical.current.phi = Math.max(0.1,Math.min(Math.PI-0.1,spherical.current.phi+dy*0.008))
      return
    }
    if (!mountRef.current||!cameraRef.current||!songDataRef.current.length) return
    const rect=mountRef.current.getBoundingClientRect()
    const mx=((e.clientX-rect.left)/rect.width)*2-1
    const my=-((e.clientY-rect.top)/rect.height)*2+1
    const raycaster=new THREE.Raycaster()
    raycaster.setFromCamera({x:mx,y:my},cameraRef.current)
    raycaster.params.Points.threshold=0.04
    if (pointsRef.current) {
      const hits=raycaster.intersectObject(pointsRef.current)
      setHovered(hits.length>0?songDataRef.current[hits[0].index]:null)
    }
  }, [])

  const onClick = useCallback(e => {
    if (!mountRef.current||!cameraRef.current||!songDataRef.current.length) return
    const rect=mountRef.current.getBoundingClientRect()
    const mx=((e.clientX-rect.left)/rect.width)*2-1
    const my=-((e.clientY-rect.top)/rect.height)*2+1
    const raycaster=new THREE.Raycaster()
    raycaster.setFromCamera({x:mx,y:my},cameraRef.current)
    raycaster.params.Points.threshold=0.04
    if (pointsRef.current) {
      const hits=raycaster.intersectObject(pointsRef.current)
      if (hits.length>0) onSelect(songDataRef.current[hits[0].index])
      else onSelect(null)
    }
  }, [onSelect])

  return (
    <div style={{position:'relative',width:'100%',height:'100%'}}>
      <div ref={mountRef} style={{width:'100%',height:'100%',cursor:'grab'}}
        onMouseDown={onMouseDown} onMouseMove={onMouseMove} onMouseUp={onMouseUp}
        onMouseLeave={onMouseUp} onWheel={onWheel} onClick={onClick}
      />
      {hovered && (
        <div className={styles.tooltip} style={{left:mousePos.x+14,top:mousePos.y-10,borderColor:GENRE_COLORS[hovered.genre]||GENRE_COLORS['unknown'],position:'fixed'}}>
          <div className={styles.ttFilename}>{hovered.filename}</div>
          <div className={styles.ttGenre} style={{color:GENRE_COLORS[hovered.genre]||GENRE_COLORS['unknown']}}>{hovered.genre}</div>
          <div className={styles.ttAudio}>{hovered.id===selectedId?'🔊 playing...':'Click to play'}</div>
        </div>
      )}
    </div>
  )
}

// ── Main component ────────────────────────────────────────────────────────────
export default function UMAPScatter3D({ highlightGenre }) {
  const [umapSongs, setUmapSongs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [activeView, setActiveView] = useState('umap2d')
  const [selectedSong, setSelectedSong] = useState(null)
  const [showInfo, setShowInfo] = useState(false)
  const fetchedRef = useRef(false)
  const audioRef = useRef(null)

  useEffect(() => {
    if (fetchedRef.current) return
    fetchedRef.current = true
    async function load() {
      try {
        setLoading(true)
        const data = await api.getUMAP3D()
        setUmapSongs(data)
      } catch (e) {
        setError(e.message)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  const handleSelect = useCallback((song) => {
    if (audioRef.current) { audioRef.current.pause(); audioRef.current.currentTime=0; audioRef.current=null }
    if (!song || song.id === selectedSong?.id) { setSelectedSong(null); return }
    setSelectedSong(song)
    const audio = new Audio(`/api/audio/${song.filename}`)
    audio.loop = true; audio.volume = 0.6
    audioRef.current = audio
    audio.play().catch(e => console.warn('Audio failed:', e))
  }, [selectedSong])

  useEffect(() => () => { audioRef.current?.pause() }, [])

  const songs = umapSongs || []

  if (loading) return (
    <div className={styles.center}>
      <div className={styles.spinner} />
      <span className={styles.loadText}>Computing UMAP...</span>
      <span className={styles.loadHint}>This may take 10–30 seconds</span>
    </div>
  )

  if (error) return (
    <div className={styles.center}>
      <div className={styles.errorMsg}><span>⚠</span>{error}</div>
    </div>
  )

  return (
    <div className={styles.explorerWrapper}>
      {/* Sub-nav */}
      <div className={styles.subNav}>
        <span className={styles.subNavLabel}>UMAP PROJECTION</span>
        <div className={styles.subNavBtns}>
          {VIEWS.map(v => (
            <button key={v.id}
              className={`${styles.subBtn} ${activeView === v.id ? styles.subBtnActive : ''}`}
              onClick={() => setActiveView(v.id)}
            >{v.label}</button>
          ))}
        </div>
        <button className={`${styles.infoBtn} ${showInfo ? styles.infoBtnActive : ''}`} onClick={() => setShowInfo(p => !p)}>
          ? Info
        </button>
        {selectedSong && (
          <div className={styles.nowPlaying}>
            <span className={styles.nowPlayingDot} />
            <span>{selectedSong.filename}</span>
            <button className={styles.stopBtn} onClick={() => handleSelect(null)}>✕</button>
          </div>
        )}
      </div>

      {/* Info panel */}
      {showInfo && <InfoPanel onClose={() => setShowInfo(false)} />}

      {/* Visualization */}
      <div className={styles.vizArea}>
        {activeView === 'umap2d'
          ? <PairPlot songs={songs} highlightGenre={highlightGenre} selectedId={selectedSong?.id} onSelect={handleSelect} />
          : <View3D songs={songs} highlightGenre={highlightGenre} selectedId={selectedSong?.id} onSelect={handleSelect} />
        }
      </div>

      <div className={styles.hintBar}>
        {activeView === 'umap3d'
          ? 'Drag to rotate · Scroll to zoom · Click a point to play audio'
          : 'Scroll to zoom · Drag to pan · Click a point to play audio · Click again to stop'
        }
      </div>
    </div>
  )
}