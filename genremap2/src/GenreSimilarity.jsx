// GenreSimilarity.jsx — Interactive genre-genre similarity heatmap
import { useMemo, useState, useEffect } from 'react'
import styles from './GenreSimilarity.module.css'

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
}

function simColor(sim) {
  // Blue (low similarity) -> Teal (high similarity)
  const t = sim
  return `rgba(0, 229, 195, ${0.08 + t * 0.82})`
}

function textColor(sim) {
  return sim > 0.5 ? 'var(--text-primary)' : 'var(--text-dim)'
}

export default function GenreSimilarity() {
  const [similarityData, setSimilarityData] = useState(null)
  const [hoveredCell, setHoveredCell] = useState(null)
  const [method, setMethod] = useState('combined_similarity')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  // Load genre similarity data
  useEffect(() => {
    async function load() {
      try {
        const res = await fetch('/genre_similarity.json')
        if (!res.ok) throw new Error('Genre similarity data not found. Run compute_genre_similarity.py first.')
        const data = await res.json()
        setSimilarityData(data)
      } catch (e) {
        setError(e.message)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  const genres = similarityData?.genres || []
  
  const matrix = useMemo(() => {
    if (!similarityData || !method) return []
    const methodData = similarityData[method]
    if (!methodData) return []
    
    return genres.map(g1 => 
      genres.map(g2 => methodData[g1]?.[g2] || 0)
    )
  }, [similarityData, genres, method])

  const hoveredRow = hoveredCell?.row
  const hoveredCol = hoveredCell?.col

  if (loading) {
    return (
      <div className={styles.container}>
        <div className={styles.loading}>Loading genre similarity data...</div>
      </div>
    )
  }

  if (error) {
    return (
      <div className={styles.container}>
        <div className={styles.error}>
          <span className={styles.errorIcon}>⚠</span>
          {error}
          <div className={styles.errorHint}>
            Run: <code>python compute_genre_similarity.py --csv Data/features_30_sec.csv</code>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className={styles.container}>
      {/* Header controls */}
      <div className={styles.controls}>
        <span className={styles.controlLabel}>GENRE SIMILARITY</span>
        <div className={styles.methodBtns}>
          <span className={styles.controlLabel} style={{ marginRight: 4 }}>METHOD</span>
          <button
            className={`${styles.methodBtn} ${method === 'combined_similarity' ? styles.methodActive : ''}`}
            onClick={() => setMethod('combined_similarity')}
          >
            Combined
          </button>
          <button
            className={`${styles.methodBtn} ${method === 'wasserstein_similarity' ? styles.methodActive : ''}`}
            onClick={() => setMethod('wasserstein_similarity')}
          >
            Distribution
          </button>
          <button
            className={`${styles.methodBtn} ${method === 'boundary_overlap' ? styles.methodActive : ''}`}
            onClick={() => setMethod('boundary_overlap')}
          >
            Boundary
          </button>
          <button
            className={`${styles.methodBtn} ${method === 'centroid_correlation' ? styles.methodActive : ''}`}
            onClick={() => setMethod('centroid_correlation')}
          >
            Centroid
          </button>
        </div>
      </div>

      {/* Info bar */}
      <div className={styles.infoBar}>
        {hoveredCell ? (
          <>
            <span 
              className={styles.infoGenre}
              style={{ color: GENRE_COLORS[genres[hoveredRow]] }}
            >
              {genres[hoveredRow]}
            </span>
            <span className={styles.infoSep}>↔</span>
            <span 
              className={styles.infoGenre}
              style={{ color: GENRE_COLORS[genres[hoveredCol]] }}
            >
              {genres[hoveredCol]}
            </span>
            <span className={styles.infoSep}>→</span>
            <span className={styles.infoSim}>
              similarity = {matrix[hoveredRow][hoveredCol].toFixed(4)}
            </span>
            <span className={styles.infoBadge}>
              {matrix[hoveredRow][hoveredCol] > 0.7
                ? 'very similar'
                : matrix[hoveredRow][hoveredCol] > 0.5
                ? 'similar'
                : matrix[hoveredRow][hoveredCol] > 0.3
                ? 'somewhat similar'
                : 'dissimilar'}
            </span>
          </>
        ) : (
          <span className={styles.infoHint}>Hover a cell to inspect similarity</span>
        )}
      </div>

      {/* Scrollable heatmap */}
      <div className={styles.heatmapScroll}>
        <div className={styles.heatmapWrapper}>
          {/* Top axis */}
          <div className={styles.topAxisSpacer} />
          <div className={styles.topAxis}>
            {genres.map((g, j) => (
              <div
                key={g}
                className={`${styles.axisLabelTop} ${hoveredCol === j ? styles.axisHighlight : ''}`}
              >
                <span style={{ color: GENRE_COLORS[g] }}>{g}</span>
              </div>
            ))}
          </div>

          {/* Grid rows */}
          {matrix.map((row, i) => (
            <div key={genres[i]} className={styles.row}>
              {/* Left axis label */}
              <div
                className={`${styles.axisLabelLeft} ${hoveredRow === i ? styles.axisHighlight : ''}`}
                style={{ color: GENRE_COLORS[genres[i]] }}
              >
                {genres[i]}
              </div>

              {/* Cells */}
              {row.map((sim, j) => {
                const isHovered = hoveredRow === i && hoveredCol === j
                const isDiag = i === j
                const isRowHighlight = hoveredRow === i || hoveredCol === i
                const isColHighlight = hoveredCol === j || hoveredRow === j

                return (
                  <div
                    key={j}
                    className={`${styles.cell} ${isDiag ? styles.diagCell : ''} ${isHovered ? styles.hoveredCell : ''}`}
                    style={{
                      background: isDiag ? 'var(--bg-raised)' : simColor(sim),
                      opacity: hoveredCell && !isRowHighlight && !isColHighlight ? 0.3 : 1,
                    }}
                    onMouseEnter={() => setHoveredCell({ row: i, col: j })}
                    onMouseLeave={() => setHoveredCell(null)}
                  >
                    {!isDiag && (
                      <span
                        className={styles.cellValue}
                        style={{ color: textColor(sim) }}
                      >
                        {sim.toFixed(2)}
                      </span>
                    )}
                    {isDiag && (
                      <span className={styles.diagMark}>—</span>
                    )}
                  </div>
                )
              })}
            </div>
          ))}
        </div>
      </div>

      {/* Method descriptions */}
      <div className={styles.methodInfo}>
        {method === 'combined_similarity' && (
          <p>
            <strong>Combined:</strong> Weighted average of all methods (50% distribution + 30% boundary + 20% centroid)
          </p>
        )}
        {method === 'wasserstein_similarity' && (
          <p>
            <strong>Distribution:</strong> Wasserstein distance between feature distributions. Measures how overlapping the genres are across all 57 features.
          </p>
        )}
        {method === 'boundary_overlap' && (
          <p>
            <strong>Boundary:</strong> Proportion of boundary cases from one genre that have their nearest neighbor in another genre. High values = genres that get confused.
          </p>
        )}
        {method === 'centroid_correlation' && (
          <p>
            <strong>Centroid:</strong> Pearson correlation between mean feature vectors. Simple baseline measure.
          </p>
        )}
      </div>

      {/* Color scale legend */}
      <div className={styles.scaleLegend}>
        <span className={styles.scaleLabel}>0.0 dissimilar</span>
        <div className={styles.scaleBar}>
          <div className={styles.scaleGradient} />
        </div>
        <span className={styles.scaleLabel} style={{ color: 'var(--accent)' }}>similar 1.0</span>
      </div>
    </div>
  )
}
