// App.jsx — Integrated version
import { useState, useEffect, useCallback, useMemo } from 'react'
import ScatterPlot from './ScatterPlot'
import GenreLegend from './ClusterLegend'
import SongDetail from './SongDetail'
import UploadZone from './UploadZone'
import { api } from './api'
import styles from './App.module.css'
import FeatureByGenre from './FeatureByGenre'
import FeatureCorrelation from './FeatureCorrelation'
import UMAPScatter3D from './UMAPScatter3D'
import InfoPanel from './InfoPanel'

const VISUALIZATIONS = [
  { id: 'scatter',    label: 'Bounary Exploration Map',       available: true },
  { id: 'violin',     label: 'Feature by Genre',    available: true },
  { id: 'heatmap',    label: 'Feature Correlation', available: true },
  { id: 'umap3d',     label: 'UMAP 3D',             available: true },
]

export default function App() {
  const [songs, setSongs]               = useState([])
  const [features, setFeatures]         = useState([])
  const [selectedSong, setSelectedSong] = useState(null)
  const [highlightGenre, setHighlightGenre] = useState(null)
  const [activeViz, setActiveViz]       = useState('scatter')
  const [uploading, setUploading]       = useState(false)
  const [uploadMsg, setUploadMsg]       = useState(null)
  const [loading, setLoading]           = useState(true)
  const [error, setError]               = useState(null)
  const [boundaryThreshold, setBoundaryThreshold] = useState(10)
  const [showInfo, setShowInfo]         = useState(false)

  // ── Load data ─────────────────────────────────────────────────────────────
  useEffect(() => {
    async function load() {
      try {
        const [songsData, featuresData] = await Promise.all([
          api.getSongs(),
          api.getFeatures(),
        ])
        setSongs(songsData)
        setFeatures(featuresData)
      } catch (e) {
        setError(e.message)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  // ── Upload ────────────────────────────────────────────────────────────────
  const handleUpload = useCallback(async (file, genre = 'uploaded') => {
  setUploading(true)
  setUploadMsg(null)
  try {
    const newSong = await api.uploadSong(file, genre)
    
    // Reload all songs since boundary scores were recomputed
    const songsData = await api.getSongs()
    setSongs(songsData)
    
    // Select the newly uploaded song
    const uploadedSong = songsData.find(s => s.id === newSong.id)
    setSelectedSong(uploadedSong)
    
    setUploadMsg({ 
      type: 'ok', 
      text: `Added as ${genre} (boundary score: ${newSong.boundary_score?.toFixed(2) || 'N/A'})` 
    })
  } catch (e) {
    setUploadMsg({ type: 'err', text: e.message })
  } finally {
    setUploading(false)
    setTimeout(() => setUploadMsg(null), 5000)
  }
}, [])

  // ── Derived State (Merged Logic) ──────────────────────────────────────────
  const genreCount = useMemo(() => [...new Set(songs.map(s => s.genre))].length, [songs])
  
  // Script 2 logic: calculating how many songs pass the boundary filter
  const boundaryCases = useMemo(() => songs.filter(s => 
    s.boundary_score !== null && s.boundary_score !== undefined && s.boundary_score <= boundaryThreshold
  ), [songs, boundaryThreshold])

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div className={styles.app}>
      {/* ── Header ── */}
      <header className={styles.header}>
        <div className={styles.wordmark}>
          <span className={styles.wordmarkAccent}>Genre</span>scape
        </div>

        <div className={styles.stats}>
          <Stat label="songs" value={songs.length} />
          <Stat label="filtered" value={boundaryCases.length} />
          <Stat label="genres" value={genreCount || '—'} />
        </div>

        <div className={styles.vizSelector}>
          {VISUALIZATIONS.map(v => (
            <button
              key={v.id}
              className={`${styles.vizBtn} ${activeViz === v.id ? styles.vizActive : ''} ${!v.available ? styles.vizDisabled : ''}`}
              onClick={() => v.available && setActiveViz(v.id)}
            >
              {v.label}
              {!v.available && <span className={styles.soon}>soon</span>}
            </button>
          ))}
          <button
            className={`${styles.infoBtn} ${showInfo ? styles.infoBtnActive : ''}`}
            onClick={() => setShowInfo(p => !p)}
          >
            ? Info
          </button>
        </div>
      </header>

      {/* ── Info panel (from Script 1) ── */}
      {showInfo && <InfoPanel activeViz={activeViz} onClose={() => setShowInfo(false)} />}

      <div className={styles.body}>
        {/* Left sidebar — legend */}
        <aside className={styles.sidebar}>
          {songs.length > 0 && (
            <GenreLegend
              songs={songs}
              highlightGenre={highlightGenre}
              onHover={setHighlightGenre}
            />
          )}
        </aside>

        {/* Center — visualization */}
        <main className={styles.canvas}>
          {loading && (
            <div className={styles.center}>
              <div className={styles.loadSpinner} />
              <span className={styles.loadText}>Loading dataset...</span>
            </div>
          )}
          {error && (
            <div className={styles.center}>
              <div className={styles.errorMsg}>
                <span className={styles.errorIcon}>⚠</span>
                {error}
                <span className={styles.errorHint}>Make sure FastAPI is on port 8000</span>
              </div>
            </div>
          )}
          {!loading && !error && (
            <>
              {activeViz === 'scatter' && (
                <ScatterPlot
                  songs={songs}
                  selectedId={selectedSong?.id}
                  onSelect={setSelectedSong}
                  highlightGenre={highlightGenre}
                  boundaryThreshold={boundaryThreshold} 
                />
              )}
              {activeViz === 'violin' && <FeatureByGenre songs={songs} />}
              {activeViz === 'heatmap' && <FeatureCorrelation songs={songs} />}
              {activeViz === 'umap3d' && (
                <UMAPScatter3D songs={songs} highlightGenre={highlightGenre} />
              )}
            </>
          )}
        </main>

        {/* Right sidebar — upload + detail */}
        <aside className={styles.rightPanel}>
          <div className={styles.filterSection}>
            <div className={styles.sectionLabel}>BOUNDARY FILTER</div>
            <div className={styles.filterControl}>
              <label className={styles.filterLabel}>
                <span>Max score: {boundaryThreshold === 10 ? 'All' : boundaryThreshold.toFixed(1)}</span>
              </label>
              <input
                type="range"
                min="0.5"
                max="10"
                step="0.1"
                value={boundaryThreshold}
                onChange={(e) => setBoundaryThreshold(parseFloat(e.target.value))}
                className={styles.filterSlider}
              />
              <div className={styles.filterTicks}>
                <span>Boundary</span>
                <span>Clear</span>
              </div>
            </div>
          </div>

          <div className={styles.divider} />

          <div className={styles.uploadSection}>
            <div className={styles.sectionLabel}>ADD SONG</div>
            <UploadZone onUpload={handleUpload} uploading={uploading} />
            {uploadMsg && (
              <div className={`${styles.uploadMsg} ${styles[uploadMsg.type]}`}>
                {uploadMsg.text}
              </div>
            )}
          </div>

          <div className={styles.divider} />

          <div className={styles.detailSection}>
            <div className={styles.sectionLabel}>SONG DETAIL</div>
            {selectedSong
              ? <SongDetail song={selectedSong} onClose={() => setSelectedSong(null)} />
              : <div className={styles.emptyDetail}>Click any point to inspect</div>
            }
          </div>
        </aside>
      </div>

      {/* ── Dynamic Footer (from Script 1) ── */}
      <footer className={styles.footer}>
        {activeViz === 'scatter' && <span>Scroll to zoom · Drag to pan · Click to select & play</span>}
        {activeViz === 'violin' && <span>Select a feature · Hover a box to see stats</span>}
        {activeViz === 'heatmap' && <span>Hover cell to inspect correlation</span>}
        {activeViz === 'umap3d' && <span>Drag to rotate · Scroll to zoom · Proximity = Similarity</span>}
      </footer>
    </div>
  )
}

function Stat({ label, value }) {
  return (
    <div className={styles.stat}>
      <span className={styles.statValue}>{value}</span>
      <span className={styles.statLabel}>{label}</span>
    </div>
  )
}