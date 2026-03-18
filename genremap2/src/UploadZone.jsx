// UploadZone.jsx — Drag-and-drop upload with genre selection
import { useState, useCallback } from 'react'
import styles from './UploadZone.module.css'

const GENRES = [
  'blues', 'classical', 'country', 'disco', 'hiphop',
  'jazz', 'metal', 'pop', 'reggae', 'rock'
]

export default function UploadZone({ onUpload, uploading }) {
  const [isDragging, setIsDragging] = useState(false)
  const [selectedGenre, setSelectedGenre] = useState('uploaded')

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setIsDragging(false)

    const file = e.dataTransfer.files[0]
    if (file && file.name.endsWith('.wav')) {
      onUpload(file, selectedGenre)
    }
  }, [onUpload, selectedGenre])

  const handleFileSelect = useCallback((e) => {
    const file = e.target.files[0]
    if (file) {
      onUpload(file, selectedGenre)
    }
  }, [onUpload, selectedGenre])

  return (
    <div className={styles.container}>
      {/* Genre selector */}
      <div className={styles.genreSelector}>
        <label className={styles.genreLabel}>Genre:</label>
        <select
          className={styles.genreSelect}
          value={selectedGenre}
          onChange={(e) => setSelectedGenre(e.target.value)}
          disabled={uploading}
        >
          <option value="uploaded">Unknown / Unlabeled</option>
          {GENRES.map(g => (
            <option key={g} value={g}>
              {g.charAt(0).toUpperCase() + g.slice(1)}
            </option>
          ))}
        </select>
      </div>

      {/* Drop zone */}
      <div
        className={`${styles.dropzone} ${isDragging ? styles.dragging : ''} ${uploading ? styles.uploading : ''}`}
        onDragOver={(e) => {
          e.preventDefault()
          setIsDragging(true)
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
      >
        {uploading ? (
          <div className={styles.spinner} />
        ) : (
          <>
            <div className={styles.icon}>↑</div>
            <div className={styles.text}>
              Drop .wav file here
              <span className={styles.or}>or</span>
            </div>
            <label className={styles.browse}>
              Browse
              <input
                type="file"
                accept=".wav"
                onChange={handleFileSelect}
                style={{ display: 'none' }}
              />
            </label>
          </>
        )}
      </div>

      <div className={styles.hint}>
        {selectedGenre === 'uploaded' 
          ? 'Song will be marked as unlabeled'
          : `Song will be added as ${selectedGenre}`}
      </div>
    </div>
  )
}
