// UploadZone.jsx — drag-and-drop + click file upload
import { useState, useCallback } from 'react'
import styles from './UploadZone.module.css'

export default function UploadZone({ onUpload, uploading }) {
  const [dragging, setDragging] = useState(false)

  const handleFile = useCallback((file) => {
    if (!file) return
    if (!file.name.match(/\.wav$/i)) {
      alert('Only .wav files are supported')
      return
    }
    onUpload(file)
  }, [onUpload])

  const onDrop = useCallback((e) => {
    e.preventDefault()
    setDragging(false)
    handleFile(e.dataTransfer.files[0])
  }, [handleFile])

  const onDragOver = (e) => { e.preventDefault(); setDragging(true) }
  const onDragLeave = () => setDragging(false)

  const onInputChange = (e) => handleFile(e.target.files[0])

  return (
    <div
      className={`${styles.zone} ${dragging ? styles.dragging : ''} ${uploading ? styles.uploading : ''}`}
      onDrop={onDrop}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
    >
      <input
        type="file"
        accept=".wav"
        className={styles.input}
        onChange={onInputChange}
        disabled={uploading}
        id="wav-upload"
      />
      <label htmlFor="wav-upload" className={styles.label}>
        {uploading ? (
          <>
            <div className={styles.spinner} />
            <span className={styles.text}>Processing...</span>
          </>
        ) : (
          <>
            <div className={styles.icon}>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                <polyline points="17 8 12 3 7 8"/>
                <line x1="12" y1="3" x2="12" y2="15"/>
              </svg>
            </div>
            <span className={styles.text}>Upload .wav</span>
            <span className={styles.sub}>drag or click</span>
          </>
        )}
      </label>
    </div>
  )
}
