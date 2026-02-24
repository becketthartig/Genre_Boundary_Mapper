// App.jsx — root component, supervised genre visualization
import { useState, useEffect, useCallback } from 'react'
import ScatterPlot from './ScatterPlot'
import GenreLegend from './ClusterLegend'
import SongDetail from './SongDetail'
import UploadZone from './UploadZone'
import { api } from './api'
import styles from './App.module.css'

const VISUALIZATIONS = [
  { id: 'scatter',    label: 'Genre PCA Map',            available: true  },
  { id: 'violin',     label: 'Feature by Genre',         available: false },
  { id: 'heatmap',    label: 'Feature Correlation',      available: false },
]

export default function App() {
  const [songs, setSongs]             = useState([])
  const [features, setFeatures]       = useState([])
  const [selectedSong, setSelectedSong] = useState(null)
  const [highlightGenre, setHighlightGenre] = useState(null)
  const [activeViz, setActiveViz]     = useState('scatter')
  const [uploading, setUploading]     = useState(false)
  const [uploadMsg, setUploadMsg]     = useState(null)   // { type: 'ok'|'err', text }
  const [loading, setLoading]         = useState(true)
  const [error, setError]             = useState(null)
  const [boundaryThreshold, setBoundaryThreshold] = useState(10)  // Show all by default

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
  const handleUpload = useCallback(async (file) => {
    setUploading(true)
    setUploadMsg(null)
    try {
      const newSong = await api.uploadSong(file)
      setSongs(prev => [...prev, newSong])
      setSelectedSong(newSong)
      setUploadMsg({ type: 'ok', text: `Added (genre: ${newSong.genre})` })
    } catch (e) {
      setUploadMsg({ type: 'err', text: e.message })
    } finally {
      setUploading(false)
      setTimeout(() => setUploadMsg(null), 4000)
    }
  }, [])

  // ── Stats ─────────────────────────────────────────────────────────────────
  const genreCount = [...new Set(songs.map(s => s.genre))].length
  
  // Filter songs by boundary threshold
  const filteredSongs = songs.filter(s => 
    s.boundary_score === null || s.boundary_score === undefined || s.boundary_score <= boundaryThreshold
  )

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div className={styles.app}>
      {/* ── Header ── */}
      <header className={styles.header}>
        <div className={styles.wordmark}>
          <span className={styles.wordmarkAccent}>Genre</span>Map
        </div>

        <div className={styles.stats}>
          <Stat label="songs" value={songs.length} />
          <Stat label="shown" value={filteredSongs.length} />
          <Stat label="genres" value={genreCount || '—'} />
        </div>

        <div className={styles.vizSelector}>
          {VISUALIZATIONS.map(v => (
            <button
              key={v.id}
              className={`${styles.vizBtn} ${activeViz === v.id ? styles.vizActive : ''} ${!v.available ? styles.vizDisabled : ''}`}
              onClick={() => v.available && setActiveViz(v.id)}
              title={!v.available ? 'Coming soon' : undefined}
            >
              {v.label}
              {!v.available && <span className={styles.soon}>soon</span>}
            </button>
          ))}
        </div>
      </header>

      {/* ── Main canvas ── */}
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
                <span className={styles.errorHint}>
                  Make sure the FastAPI backend is running on port 8000
                </span>
              </div>
            </div>
          )}
          {!loading && !error && activeViz === 'scatter' && (
            <ScatterPlot
              songs={filteredSongs}
              selectedId={selectedSong?.id}
              onSelect={setSelectedSong}
              highlightGenre={highlightGenre}
            />
          )}
          {!loading && !error && activeViz !== 'scatter' && (
            <div className={styles.center}>
              <div className={styles.comingSoon}>
                <div className={styles.comingSoonLabel}>coming soon</div>
                <div className={styles.comingSoonTitle}>
                  {VISUALIZATIONS.find(v => v.id === activeViz)?.label}
                </div>
              </div>
            </div>
          )}
        </main>

        {/* Right sidebar — upload + detail */}
        <aside className={styles.rightPanel}>
          <div className={styles.filterSection}>
            <div className={styles.sectionLabel}>BOUNDARY FILTER</div>
            <div className={styles.filterControl}>
              <label className={styles.filterLabel}>
                <span>Max score: {boundaryThreshold === 10 ? 'All' : boundaryThreshold.toFixed(1)}</span>
                <span className={styles.filterHint}>
                  {boundaryThreshold < 1.5 ? 'Strong boundary cases' : 
                   boundaryThreshold < 3 ? 'Moderate boundaries' : 
                   'All songs'}
                </span>
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

      {/* ── Footer ── */}
      <footer className={styles.footer}>
        <span>Scroll to zoom · Drag to pan · Click to select & play audio</span>
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
