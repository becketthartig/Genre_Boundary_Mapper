// api.js — Integrated "Superscript"
// All calls go through Vite's proxy: /api → http://localhost:8000

const BASE = '/api'

/**
 * Generic request helper with error handling
 */
async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, options)
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || `HTTP ${res.status}`)
  }
  return res.json()
}

export const api = {
  /** 
   * Fetch 3D UMAP projection data.
   * Requires 'umap-learn' on the backend.
   */
  getUMAP3D: () => request('/umap3d'),

  /** Fetch all songs with PCA coords, genre labels, and features */
  getSongs: () => request('/songs'),

  /** Fetch list of available feature names (active and inactive) */
  getFeatures: () => request('/features'),

  /** 
   * Fetch StandardScaler parameters for normalizing features.
   * Returns { mean: [...], scale: [...], features: [...] }
   * Used to ensure frontend distance calculations match backend.
   */
  getScaler: () => request('/scaler'),

  /**
   * Fetch mean feature values for each genre in normalized space.
   * Returns { blues: {...}, jazz: {...}, ... }
   * Used to show deviation from genre means.
   */
  getGenreMeans: () => request('/genre-means'),

  /** 
   * Upload a new .wav file with genre label.
   * Returns the processed song object with extracted features and projections.
   * 
   * @param {File} file - WAV file to upload
   * @param {string} genre - Genre label (defaults to 'uploaded')
   */
  uploadSong: (file, genre = 'uploaded') => {
    const form = new FormData()
    form.append('file', file)
    form.append('genre', genre)
    return request('/upload', { method: 'POST', body: form })
  },

  /** 
   * Recompute dimensionality reduction with a different subset of features and/or method.
   * Forces the backend to update results.json and pipeline.pkl.
   * 
   * @param {string[]} features - List of feature names to include
   * @param {string} method - 'lda', 'umap-supervised', or 'umap-unsupervised'
   */
  recompute: (features, method = 'umap-supervised') => 
    request('/recompute', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features, method }),
    }),

  /** Basic API health and status check */
  health: () => request('/health'),
}