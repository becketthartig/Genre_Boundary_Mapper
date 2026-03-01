// FeatureCorrelation.jsx — Pearson correlation heatmap across all audio features
import { useMemo, useState } from 'react'
import styles from './FeatureCorrelation.module.css'

function pearson(a, b) {
  const n = a.length
  if (n === 0) return 0
  const meanA = a.reduce((s, v) => s + v, 0) / n
  const meanB = b.reduce((s, v) => s + v, 0) / n
  let num = 0, da = 0, db = 0
  for (let i = 0; i < n; i++) {
    const ra = a[i] - meanA
    const rb = b[i] - meanB
    num += ra * rb
    da += ra * ra
    db += rb * rb
  }
  const denom = Math.sqrt(da * db)
  return denom === 0 ? 0 : num / denom
}

function corrColor(r) {
  // negative: blue (#6e8efb), zero: dark bg, positive: teal (#00e5c3)
  if (r > 0) {
    const t = r
    return `rgba(0, 229, 195, ${0.08 + t * 0.82})`
  } else {
    const t = -r
    return `rgba(110, 142, 251, ${0.08 + t * 0.82})`
  }
}

function textColor(r) {
  return Math.abs(r) > 0.5 ? 'var(--text-primary)' : 'var(--text-dim)'
}

export default function FeatureCorrelation({ songs }) {
  const [hoveredCell, setHoveredCell] = useState(null)  // { row, col }
  const [sortBy, setSortBy] = useState('name')  // 'name' | 'variance'

  const featureNames = useMemo(() => {
    const first = songs.find(s => s.features && Object.keys(s.features).length > 0)
    return first ? Object.keys(first.features).sort() : []
  }, [songs])

  // Build feature vectors
  const featureVectors = useMemo(() => {
    const map = {}
    for (const name of featureNames) {
      map[name] = songs
        .filter(s => s.features?.[name] != null)
        .map(s => s.features[name])
    }
    return map
  }, [songs, featureNames])

  // Optionally sort by variance (most variable features first)
  const sortedFeatures = useMemo(() => {
    if (sortBy === 'name') return [...featureNames]
    return [...featureNames].sort((a, b) => {
      const va = featureVectors[a]
      const vb = featureVectors[b]
      const varA = va.reduce((s, v, _, arr) => {
        const m = arr.reduce((a, b) => a + b, 0) / arr.length
        return s + (v - m) ** 2
      }, 0)
      const varB = vb.reduce((s, v, _, arr) => {
        const m = arr.reduce((a, b) => a + b, 0) / arr.length
        return s + (v - m) ** 2
      }, 0)
      return varB - varA
    })
  }, [featureNames, featureVectors, sortBy])

  // Compute full correlation matrix
  const corrMatrix = useMemo(() => {
    return sortedFeatures.map(a =>
      sortedFeatures.map(b => {
        const va = featureVectors[a]
        const vb = featureVectors[b]
        const minLen = Math.min(va.length, vb.length)
        return pearson(va.slice(0, minLen), vb.slice(0, minLen))
      })
    )
  }, [sortedFeatures, featureVectors])

  const n = sortedFeatures.length

  const hoveredRow = hoveredCell?.row
  const hoveredCol = hoveredCell?.col

  return (
    <div className={styles.container}>
      {/* Header controls */}
      <div className={styles.controls}>
        <span className={styles.controlLabel}>FEATURE CORRELATION</span>
        <div className={styles.sortBtns}>
          <span className={styles.controlLabel} style={{ marginRight: 4 }}>SORT</span>
          <button
            className={`${styles.sortBtn} ${sortBy === 'name' ? styles.sortActive : ''}`}
            onClick={() => setSortBy('name')}
          >
            A–Z
          </button>
          <button
            className={`${styles.sortBtn} ${sortBy === 'variance' ? styles.sortActive : ''}`}
            onClick={() => setSortBy('variance')}
          >
            Variance
          </button>
        </div>
      </div>

      {/* Hover info bar */}
      <div className={styles.infoBar}>
        {hoveredCell ? (
          <>
            <span className={styles.infoFeature} style={{ color: 'var(--accent)' }}>
              {sortedFeatures[hoveredRow]}
            </span>
            <span className={styles.infoSep}>×</span>
            <span className={styles.infoFeature} style={{ color: '#6e8efb' }}>
              {sortedFeatures[hoveredCol]}
            </span>
            <span className={styles.infoSep}>→</span>
            <span
              className={styles.infoCorr}
              style={{
                color: corrMatrix[hoveredRow][hoveredCol] > 0 ? 'var(--accent)' : '#6e8efb'
              }}
            >
              r = {corrMatrix[hoveredRow][hoveredCol].toFixed(4)}
            </span>
            <span className={styles.infoBadge}>
              {Math.abs(corrMatrix[hoveredRow][hoveredCol]) > 0.7
                ? 'strong'
                : Math.abs(corrMatrix[hoveredRow][hoveredCol]) > 0.4
                ? 'moderate'
                : 'weak'}
            </span>
          </>
        ) : (
          <span className={styles.infoHint}>Hover a cell to inspect correlation</span>
        )}
      </div>

      {/* Scrollable heatmap */}
      <div className={styles.heatmapScroll}>
        <div className={styles.heatmapWrapper}>
          {/* Top axis labels */}
          <div className={styles.topAxisSpacer} />
          <div className={styles.topAxis}>
            {sortedFeatures.map((f, j) => (
              <div
                key={f}
                className={`${styles.axisLabelTop} ${hoveredCol === j ? styles.axisHighlight : ''}`}
              >
                <span>{f.replace(/_/g, ' ')}</span>
              </div>
            ))}
          </div>

          {/* Grid rows */}
          {corrMatrix.map((row, i) => (
            <div key={sortedFeatures[i]} className={styles.row}>
              {/* Left axis label */}
              <div
                className={`${styles.axisLabelLeft} ${hoveredRow === i ? styles.axisHighlight : ''}`}
              >
                {sortedFeatures[i].replace(/_/g, ' ')}
              </div>

              {/* Cells */}
              {row.map((r, j) => {
                const isHovered = hoveredRow === i && hoveredCol === j
                const isDiag = i === j
                const isRowHighlight = hoveredRow === i || hoveredCol === i
                const isColHighlight = hoveredCol === j || hoveredRow === j

                return (
                  <div
                    key={j}
                    className={`${styles.cell} ${isDiag ? styles.diagCell : ''} ${isHovered ? styles.hoveredCell : ''}`}
                    style={{
                      background: isDiag ? 'var(--bg-raised)' : corrColor(r),
                      opacity: hoveredCell && !isRowHighlight && !isColHighlight ? 0.3 : 1,
                    }}
                    onMouseEnter={() => setHoveredCell({ row: i, col: j })}
                    onMouseLeave={() => setHoveredCell(null)}
                  >
                    {!isDiag && (
                      <span
                        className={styles.cellValue}
                        style={{ color: textColor(r) }}
                      >
                        {r.toFixed(2)}
                      </span>
                    )}
                    {isDiag && (
                      <span className={styles.diagMark}>1</span>
                    )}
                  </div>
                )
              })}
            </div>
          ))}
        </div>
      </div>

      {/* Color scale legend */}
      <div className={styles.scaleLegend}>
        <span className={styles.scaleLabel} style={{ color: '#6e8efb' }}>−1 negative</span>
        <div className={styles.scaleBar}>
          <div className={styles.scaleGradient} />
        </div>
        <span className={styles.scaleLabel} style={{ color: 'var(--accent)' }}>positive +1</span>
      </div>
    </div>
  )
}
