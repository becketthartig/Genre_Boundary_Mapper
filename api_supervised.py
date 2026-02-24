"""
api_supervised.py
-----------------
FastAPI backend for supervised genre visualization.

Colors by true genre labels, no clustering.
Supports dynamic feature selection and PCA re-computation.

Endpoints:
  GET  /songs          → all songs with PCA coords and genre labels
  GET  /features       → list of all available feature names
  POST /upload         → process a new .wav file
  POST /recompute      → recompute PCA with different feature selection
  GET  /health         → status check

Run with:
  uvicorn api_supervised:app --reload --port 8000

Install deps:
  pip install fastapi uvicorn python-multipart
"""

import json
import shutil
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from inference_supervised import load_pipeline, process_new_song, append_to_results

# ─── Config ───────────────────────────────────────────────────────────────────

RESULTS_PATH = "results.json"
PIPELINE_PATH = "pipeline.pkl"
CSV_PATH = "Data/features_30_sec.csv"  # needed for recompute
AUDIO_BASE_PATH = "Data/genres_original"  # path to audio files relative to where API runs

# ─── App state ────────────────────────────────────────────────────────────────

app_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load pipeline and results at startup."""
    print("Loading pipeline...")
    app_state["pipeline"] = load_pipeline(PIPELINE_PATH)

    print("Loading results...")
    with open(RESULTS_PATH) as f:
        app_state["songs"] = json.load(f)

    # Load the original CSV for recompute endpoint
    import pandas as pd
    app_state["df"] = pd.read_csv(CSV_PATH)

    print(f"Ready. {len(app_state['songs'])} songs loaded.")
    yield


app = FastAPI(
    title="Genre Visualizer API (Supervised)",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request models ───────────────────────────────────────────────────────────

class RecomputeRequest(BaseModel):
    features: list[str]  # list of feature names to include in PCA


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    pipeline = app_state.get("pipeline")
    songs = app_state.get("songs", [])
    genres = list(set(s["genre"] for s in songs))
    return {
        "status": "ok",
        "mode": "supervised",
        "songs_loaded": len(songs),
        "genres": len(genres),
        "features_in_use": len(pipeline["feat_cols"]) if pipeline else None,
    }


@app.get("/songs")
def get_songs() -> list[dict]:
    """Returns all songs with PCA coords and genre labels."""
    return app_state["songs"]


@app.get("/features")
def get_features() -> dict:
    """
    Returns available features and which ones are currently in use.
    
    Response shape:
      {
        "all": [...],      # all features from the CSV
        "active": [...]    # features currently used in PCA
      }
    """
    df = app_state.get("df")
    pipeline = app_state.get("pipeline")
    
    if df is None or pipeline is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    exclude = {"filename", "length", "label"}
    all_features = [c for c in df.columns if c not in exclude]
    active_features = pipeline["feat_cols"]
    
    return {
        "all": all_features,
        "active": active_features,
    }


@app.get("/songs/{song_id}")
def get_song(song_id: int) -> dict:
    songs = app_state["songs"]
    matches = [s for s in songs if s["id"] == song_id]
    if not matches:
        raise HTTPException(status_code=404, detail=f"Song id={song_id} not found")
    return matches[0]


@app.get("/genre/{genre_name}")
def get_genre(genre_name: str) -> list[dict]:
    """All songs with a given genre label."""
    return [s for s in app_state["songs"] if s["genre"].lower() == genre_name.lower()]


@app.get("/audio/{filename}")
def get_audio(filename: str):
    """
    Serve audio file for a given filename.
    
    Expects filename like 'blues.00008.wav' and constructs path as:
    Data/genres_original/{genre}/{filename}
    
    Returns 404 if file doesn't exist.
    """
    from pathlib import Path
    
    # Extract genre from filename (e.g., 'blues.00008.wav' -> 'blues')
    genre = filename.split('.')[0]
    
    audio_path = Path(AUDIO_BASE_PATH) / genre / filename
    
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail=f"Audio file not found: {audio_path}")
    
    return FileResponse(
        path=str(audio_path),
        media_type="audio/wav",
        headers={
            "Accept-Ranges": "bytes",  # Enable seeking/streaming
            "Cache-Control": "public, max-age=3600",  # Cache for 1 hour
        }
    )


@app.post("/upload")
async def upload_song(file: UploadFile = File(...)) -> dict[str, Any]:
    """
    Upload a .wav file, extract features, project into PCA space.
    Genre is set to "unknown" since we don't have ground truth.
    """
    if not file.filename.endswith((".wav", ".WAV")):
        raise HTTPException(status_code=400, detail="Only .wav files are supported")

    pipeline = app_state.get("pipeline")
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        from extractor import extract_features
        feature_list = extract_features(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {e}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    try:
        result = process_new_song(
            filename=file.filename,
            feature_list=feature_list,
            pipeline=pipeline,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    updated_songs = append_to_results(result, RESULTS_PATH)
    app_state["songs"] = updated_songs

    return result


@app.post("/recompute")
def recompute_pca(req: RecomputeRequest) -> dict:
    """
    Recompute LDA using a different subset of features.
    
    This regenerates results.json and pipeline.pkl without needing
    to re-run the command-line script.
    
    Returns the new feature set and explained variance.
    """
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.neighbors import NearestNeighbors
    import pickle
    
    df = app_state.get("df")
    if df is None:
        raise HTTPException(status_code=503, detail="CSV data not loaded")
    
    # Validate requested features
    exclude = {"filename", "length", "label"}
    all_features = [c for c in df.columns if c not in exclude]
    invalid = [f for f in req.features if f not in all_features]
    if invalid:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid features: {invalid}. Must be one of: {all_features[:10]}..."
        )
    
    if not req.features:
        raise HTTPException(status_code=422, detail="Must specify at least one feature")
    
    # Extract feature matrix
    X = df[req.features].values.astype(np.float32)
    y = df["label"].values
    
    # Normalize and LDA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    n_classes = len(np.unique(y))
    max_components = min(2, n_classes - 1)
    
    lda = LinearDiscriminantAnalysis(n_components=max_components)
    coords_2d = lda.fit_transform(X_scaled, y)
    
    # Pad if only 1 component
    if coords_2d.shape[1] == 1:
        coords_2d = np.hstack([coords_2d, np.zeros((coords_2d.shape[0], 1))])
    
    explained = lda.explained_variance_ratio_.sum() * 100
    
    # Compute boundary scores
    k = 21
    nn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    nn.fit(X_scaled)
    distances, indices = nn.kneighbors(X_scaled)
    
    boundary_scores = np.zeros(len(y))
    within_genre_dists = []
    
    for i in range(len(y)):
        my_label = y[i]
        neighbor_labels = y[indices[i]]
        neighbor_dists = distances[i]
        
        cross_genre_mask = neighbor_labels != my_label
        if cross_genre_mask.any():
            cross_genre_dist = neighbor_dists[cross_genre_mask][0]
        else:
            cross_genre_dist = neighbor_dists[-1]
        
        boundary_scores[i] = cross_genre_dist
        
        within_genre_mask = (neighbor_labels == my_label) & (np.arange(len(neighbor_labels)) > 0)
        if within_genre_mask.any():
            within_genre_dists.append(neighbor_dists[within_genre_mask][0])
    
    median_within = np.median(within_genre_dists)
    boundary_scores = boundary_scores / median_within
    
    # Rebuild songs with new coords and boundary scores
    songs = []
    for i, row in df.reset_index(drop=True).iterrows():
        song = {
            "id": i,
            "filename": row["filename"],
            "genre": row.get("label", "unknown"),
            "x": float(coords_2d[i, 0]),
            "y": float(coords_2d[i, 1]),
            "boundary_score": float(boundary_scores[i]),
            "features": {col: float(row[col]) for col in req.features},
        }
        songs.append(song)
    
    # Save updated results and pipeline
    with open(RESULTS_PATH, "w") as f:
        json.dump(songs, f, indent=2)
    
    pipeline_state = {
        "scaler": scaler,
        "lda": lda,
        "feat_cols": req.features,
    }
    with open(PIPELINE_PATH, "wb") as f:
        pickle.dump(pipeline_state, f)
    
    # Update in-memory state
    app_state["songs"] = songs
    app_state["pipeline"] = pipeline_state
    
    return {
        "features_used": len(req.features),
        "explained_variance": round(explained, 2),
        "ld1_variance": round(lda.explained_variance_ratio_[0] * 100, 2) if max_components >= 1 else 0,
        "ld2_variance": round(lda.explained_variance_ratio_[1] * 100, 2) if max_components >= 2 else 0,
    }


@app.delete("/songs/{song_id}")
def delete_song(song_id: int) -> dict:
    """Remove a song from the dataset."""
    songs = app_state["songs"]
    matches = [s for s in songs if s["id"] == song_id]
    if not matches:
        raise HTTPException(status_code=404, detail=f"Song id={song_id} not found")

    app_state["songs"] = [s for s in songs if s["id"] != song_id]

    with open(RESULTS_PATH, "w") as f:
        json.dump(app_state["songs"], f, indent=2)

    return {"deleted": song_id}
