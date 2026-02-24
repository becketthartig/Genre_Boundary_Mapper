// GenreLegend.jsx — sidebar showing genre breakdown
import styles from './ClusterLegend.module.css'

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

export default function GenreLegend({ songs, highlightGenre, onHover }) {
  const genres = [...new Set(songs.map(s => s.genre))].sort()

  return (
    <div className={styles.legend}>
      <div className={styles.header}>
        <span className={styles.label}>GENRES</span>
        <span className={styles.count}>{genres.length}</span>
      </div>

      <div className={styles.list}>
        {genres.map(genre => {
          const color = GENRE_COLORS[genre] || GENRE_COLORS['unknown']
          const count = songs.filter(s => s.genre === genre).length
          const isHighlighted = highlightGenre === genre
          const isDimmed = highlightGenre !== null && !isHighlighted

          return (
            <div
              key={genre}
              className={`${styles.item} ${isDimmed ? styles.dimmed : ''} ${isHighlighted ? styles.highlighted : ''}`}
              onMouseEnter={() => onHover(genre)}
              onMouseLeave={() => onHover(null)}
            >
              <div className={styles.swatch} style={{ background: color }} />
              <div className={styles.info}>
                <span className={styles.clusterName}>{genre}</span>
                <span className={styles.clusterMeta}>{count} songs</span>
              </div>
              <div
                className={styles.bar}
                style={{
                  width: `${Math.round((count / songs.length) * 100)}%`,
                  background: color,
                  opacity: isDimmed ? 0.15 : 0.25,
                }}
              />
            </div>
          )
        })}
      </div>
    </div>
  )
}
