// api.js — thin wrapper around the FastAPI backend
// All calls go through Vite's proxy: /api → http://localhost:8000

const BASE = '/api'

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, options)
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || `HTTP ${res.status}`)
  }
  return res.json()
}

export const api = {
  /** Fetch all songs with coords + features */
  getSongs: () => request('/songs'),

  /** Fetch available feature names */
  getFeatures: () => request('/features'),

  /** Upload a new .wav file. Returns the new song object. */
  uploadSong: (file) => {
    const form = new FormData()
    form.append('file', file)
    return request('/upload', { method: 'POST', body: form })
  },

  /** Recompute PCA with different feature selection */
  recompute: (features) => 
    request('/recompute', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features }),
    }),

  /** Health check */
  health: () => request('/health'),
}
