// FeatureByGenre.jsx — Box plot showing feature distributions per genre
import { useState, useMemo } from 'react'
import styles from './FeatureByGenre.module.css'

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

function getColor(genre) {
  return GENRE_COLORS[genre] || GENRE_COLORS['unknown']
}

function computeStats(values) {
  if (!values.length) return null
  const sorted = [...values].sort((a, b) => a - b)
  const n = sorted.length
  const q1 = sorted[Math.floor(n * 0.25)]
  const median = sorted[Math.floor(n * 0.5)]
  const q3 = sorted[Math.floor(n * 0.75)]
  const iqr = q3 - q1
  const lowerFence = q1 - 1.5 * iqr
  const upperFence = q3 + 1.5 * iqr
  const min = sorted.find(v => v >= lowerFence) ?? sorted[0]
  const max = [...sorted].reverse().find(v => v <= upperFence) ?? sorted[n - 1]
  const mean = values.reduce((a, b) => a + b, 0) / n
  return { min, q1, median, q3, max, mean }
}

export default function FeatureByGenre({ songs }) {
  const featureNames = useMemo(() => {
    const first = songs.find(s => s.features && Object.keys(s.features).length > 0)
    return first ? Object.keys(first.features).sort() : []
  }, [songs])

  const [selectedFeature, setSelectedFeature] = useState(() => {
    // Pick a nice default
    const preferred = ['tempo', 'rms_mean', 'spectral_centroid_mean', 'mfcc1_mean']
    return preferred.find(f => featureNames.includes(f)) || featureNames[0] || ''
  })

  const [hoveredGenre, setHoveredGenre] = useState(null)

  const genres = useMemo(() => [...new Set(songs.map(s => s.genre))].sort(), [songs])

  const statsPerGenre = useMemo(() => {
    return genres.map(genre => {
      const values = songs
        .filter(s => s.genre === genre && s.features?.[selectedFeature] != null)
        .map(s => s.features[selectedFeature])
      return { genre, stats: computeStats(values), count: values.length }
    }).filter(g => g.stats !== null)
  }, [songs, genres, selectedFeature])

  // Global min/max for scale
  const globalMin = useMemo(() =>
    Math.min(...statsPerGenre.map(g => g.stats.min)), [statsPerGenre])
  const globalMax = useMemo(() =>
    Math.max(...statsPerGenre.map(g => g.stats.max)), [statsPerGenre])
  const range = globalMax - globalMin || 1

  const toPercent = (v) => ((v - globalMin) / range) * 100

  const CHART_H = 260

  return (
    <div className={styles.container}>
      {/* Feature selector */}
      <div className={styles.controls}>
        <span className={styles.controlLabel}>FEATURE</span>
        <div className={styles.featureScroll}>
          {featureNames.map(f => (
            <button
              key={f}
              className={`${styles.featureBtn} ${selectedFeature === f ? styles.featureBtnActive : ''}`}
              onClick={() => setSelectedFeature(f)}
            >
              {f.replace(/_/g, ' ')}
            </button>
          ))}
        </div>
      </div>

      {/* Chart area */}
      <div className={styles.chartArea}>
        <div className={styles.yAxisLabels}>
          <span>{globalMax.toFixed(2)}</span>
          <span>{((globalMax + globalMin) / 2).toFixed(2)}</span>
          <span>{globalMin.toFixed(2)}</span>
        </div>

        <div className={styles.chart}>
          {/* Horizontal grid lines */}
          {[0, 25, 50, 75, 100].map(pct => (
            <div
              key={pct}
              className={styles.gridLine}
              style={{ bottom: `${pct}%` }}
            />
          ))}

          {/* Box plots */}
          <div className={styles.boxes}>
            {statsPerGenre.map(({ genre, stats, count }) => {
              const color = getColor(genre)
              const isHovered = hoveredGenre === genre
              const isDimmed = hoveredGenre !== null && !isHovered

              const boxBottom = toPercent(stats.q1)
              const boxTop = toPercent(stats.q3)
              const boxH = boxTop - boxBottom
              const medianPos = toPercent(stats.median)
              const meanPos = toPercent(stats.mean)
              const whiskerBottom = toPercent(stats.min)
              const whiskerTop = toPercent(stats.max)

              return (
                <div
                  key={genre}
                  className={`${styles.boxWrapper} ${isDimmed ? styles.dimmed : ''} ${isHovered ? styles.hovered : ''}`}
                  onMouseEnter={() => setHoveredGenre(genre)}
                  onMouseLeave={() => setHoveredGenre(null)}
                >
                  {/* Whisker line */}
                  <div
                    className={styles.whisker}
                    style={{
                      bottom: `${whiskerBottom}%`,
                      height: `${whiskerTop - whiskerBottom}%`,
                      borderColor: color,
                    }}
                  />
                  {/* Whisker caps */}
                  <div className={styles.whiskerCap} style={{ bottom: `${whiskerBottom}%`, background: color }} />
                  <div className={styles.whiskerCap} style={{ bottom: `${whiskerTop}%`, background: color }} />

                  {/* IQR box */}
                  <div
                    className={styles.box}
                    style={{
                      bottom: `${boxBottom}%`,
                      height: `${Math.max(boxH, 1)}%`,
                      background: `${color}22`,
                      borderColor: color,
                    }}
                  />

                  {/* Median line */}
                  <div
                    className={styles.medianLine}
                    style={{
                      bottom: `${medianPos}%`,
                      background: color,
                    }}
                  />

                  {/* Mean dot */}
                  <div
                    className={styles.meanDot}
                    style={{
                      bottom: `${meanPos}%`,
                      background: color,
                    }}
                  />

                  {/* Label */}
                  <div className={styles.boxLabel} style={{ color: isHovered ? color : undefined }}>
                    {genre}
                  </div>

                  {/* Tooltip */}
                  {isHovered && (
                    <div className={styles.tooltip} style={{ borderColor: color }}>
                      <div className={styles.ttGenre} style={{ color }}>{genre}</div>
                      <div className={styles.ttRow}><span>max</span><span>{stats.max.toFixed(3)}</span></div>
                      <div className={styles.ttRow}><span>Q3</span><span>{stats.q3.toFixed(3)}</span></div>
                      <div className={styles.ttRow} style={{ color }}><span>median</span><span>{stats.median.toFixed(3)}</span></div>
                      <div className={styles.ttRow}><span>Q1</span><span>{stats.q1.toFixed(3)}</span></div>
                      <div className={styles.ttRow}><span>min</span><span>{stats.min.toFixed(3)}</span></div>
                      <div className={styles.ttDivider} />
                      <div className={styles.ttRow}><span>mean</span><span>{stats.mean.toFixed(3)}</span></div>
                      <div className={styles.ttRow}><span>n</span><span>{count}</span></div>
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      </div>

      <div className={styles.legend}>
        <span className={styles.legendItem}>
          <span className={styles.legendBox} /> IQR (Q1–Q3)
        </span>
        <span className={styles.legendItem}>
          <span className={styles.legendLine} /> Median
        </span>
        <span className={styles.legendItem}>
          <span className={styles.legendDot} /> Mean
        </span>
      </div>
    </div>
  )
}