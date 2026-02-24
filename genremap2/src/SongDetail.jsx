// SongDetail.jsx — detail panel shown when a song is selected
import styles from './SongDetail.module.css'

const KEY_FEATURES = [
  { key: 'spectral_centroid_mean', label: 'Spectral Centroid', unit: 'Hz', max: 5000 },
  { key: 'spectral_bandwidth_mean', label: 'Bandwidth', unit: 'Hz', max: 4000 },
  { key: 'zero_crossing_rate_mean', label: 'Zero Crossing Rate', unit: '', max: 0.3 },
  { key: 'tempo', label: 'Tempo', unit: 'BPM', max: 220 },
  { key: 'rms_mean', label: 'RMS Energy', unit: '', max: 0.4 },
  { key: 'rolloff_mean', label: 'Spectral Rolloff', unit: 'Hz', max: 10000 },
  { key: 'mfcc1_mean', label: 'MFCC 1', unit: '', min: -300, max: 0 },
  { key: 'mfcc2_mean', label: 'MFCC 2', unit: '', max: 200 },
]

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

function FeatureBar({ value, min = 0, max, color }) {
  const pct = Math.max(0, Math.min(100, ((value - min) / (max - min)) * 100))
  return (
    <div className={styles.barTrack}>
      <div className={styles.barFill} style={{ width: `${pct}%`, background: color }} />
    </div>
  )
}

function fmt(val) {
  if (val === undefined || val === null) return '—'
  if (Math.abs(val) >= 1000) return val.toFixed(0)
  if (Math.abs(val) >= 10) return val.toFixed(1)
  return val.toFixed(3)
}

export default function SongDetail({ song, onClose }) {
  if (!song) return null
  const color = GENRE_COLORS[song.genre] || GENRE_COLORS['unknown']

  return (
    <div className={styles.panel}>
      <div className={styles.header}>
        <div className={styles.clusterBadge} style={{ borderColor: color, color }}>
          {song.genre}
        </div>
        <div className={styles.playingBadge}>🔊 playing</div>
        <button className={styles.close} onClick={onClose}>✕</button>
      </div>

      <div className={styles.filename}>{song.filename}</div>

      {song.boundary_score !== null && song.boundary_score !== undefined && (
        <div className={styles.boundaryScore}>
          <span className={styles.boundaryLabel}>Boundary score:</span>
          <span className={styles.boundaryValue}>
            {song.boundary_score.toFixed(2)}
          </span>
          <span className={styles.boundaryHint}>
            {song.boundary_score < 1.5 ? '(strong boundary case)' :
             song.boundary_score < 3 ? '(moderate boundary)' :
             '(clear example)'}
          </span>
        </div>
      )}

      <div className={styles.coords}>
        <span className={styles.coordItem}>x: {song.x.toFixed(3)}</span>
        <span className={styles.coordItem}>y: {song.y.toFixed(3)}</span>
      </div>

      <div className={styles.divider} />

      <div className={styles.featuresLabel}>FEATURES</div>
      <div className={styles.features}>
        {KEY_FEATURES.map(({ key, label, unit, min = 0, max }) => {
          const val = song.features?.[key]
          if (val === undefined) return null
          return (
            <div key={key} className={styles.featureRow}>
              <div className={styles.featureMeta}>
                <span className={styles.featureName}>{label}</span>
                <span className={styles.featureVal}>
                  {fmt(val)}{unit ? ` ${unit}` : ''}
                </span>
              </div>
              <FeatureBar value={val} min={min} max={max} color={color} />
            </div>
          )
        })}
      </div>
    </div>
  )
}
