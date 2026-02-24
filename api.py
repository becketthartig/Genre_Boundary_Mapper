"""
api.py
------
FastAPI backend. Serves song data and handles new uploads.

Endpoints:
  GET  /songs          → full results.json (all songs + coords + features)
  GET  /features       → list of available feature names
  POST /upload         → process a new .wav file, return its result
  GET  /cluster/{id}   → all songs in a given cluster
  GET  /health         → pipeline status check

Run with:
  uvicorn api:app --reload --port 8000

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

from inference import load_pipeline, process_new_song, append_to_results

# ─── Config ───────────────────────────────────────────────────────────────────

RESULTS_PATH = "results.json"
PIPELINE_PATH = "pipeline.pkl"

# ─── App state ────────────────────────────────────────────────────────────────

# Pipeline loaded once at startup, shared across all requests
app_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load pipeline and results at startup."""
    print("Loading pipeline...")
    app_state["pipeline"] = load_pipeline(PIPELINE_PATH)

    print("Loading results...")
    with open(RESULTS_PATH) as f:
        app_state["songs"] = json.load(f)

    print(f"Ready. {len(app_state['songs'])} songs loaded.")
    yield
    # Cleanup on shutdown (nothing needed here)


app = FastAPI(
    title="Genre Classifier API",
    lifespan=lifespan,
)

# Allow the React dev server to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    pipeline = app_state.get("pipeline")
    songs = app_state.get("songs", [])
    return {
        "status": "ok",
        "songs_loaded": len(songs),
        "clusters": pipeline["cluster_model"].n_clusters if pipeline else None,
        "features": len(pipeline["feat_cols"]) if pipeline else None,
    }


@app.get("/songs")
def get_songs() -> list[dict]:
    """
    Returns all songs with their 2D coordinates, cluster assignment,
    and full feature dict. This is the main payload for the frontend scatter plot.
    """
    return app_state["songs"]


@app.get("/features")
def get_features() -> list[str]:
    """
    Returns the ordered list of feature names.
    Used by the frontend to populate the cross-section feature selector.
    """
    pipeline = app_state.get("pipeline")
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")
    return pipeline["feat_cols"]


@app.get("/songs/{song_id}")
def get_song(song_id: int) -> dict:
    songs = app_state["songs"]
    matches = [s for s in songs if s["id"] == song_id]
    if not matches:
        raise HTTPException(status_code=404, detail=f"Song id={song_id} not found")
    return matches[0]


@app.get("/cluster/{cluster_id}")
def get_cluster(cluster_id: int) -> list[dict]:
    """All songs belonging to a given cluster."""
    return [s for s in app_state["songs"] if s["cluster"] == cluster_id]


@app.post("/upload")
async def upload_song(file: UploadFile = File(...)) -> dict[str, Any]:
    """
    Accept a .wav file upload, extract features, assign cluster,
    project to 2D, and append to the dataset.

    Returns the new song object (same shape as /songs entries).
    """
    if not file.filename.endswith((".wav", ".WAV")):
        raise HTTPException(status_code=400, detail="Only .wav files are supported")

    pipeline = app_state.get("pipeline")
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")

    # Save upload to a temp file for librosa to read
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # Import here so api.py doesn't hard-depend on librosa at module level
        from extractor import extract_features
        feature_list = extract_features(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {e}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    # Run through the pipeline
    try:
        result = process_new_song(
            filename=file.filename,
            feature_list=feature_list,
            pipeline=pipeline,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Persist to results.json and update in-memory state
    updated_songs = append_to_results(result, RESULTS_PATH)
    app_state["songs"] = updated_songs

    return result


@app.delete("/songs/{song_id}")
def delete_song(song_id: int) -> dict:
    """Remove a song from the dataset (user-uploaded songs only, typically)."""
    songs = app_state["songs"]
    matches = [s for s in songs if s["id"] == song_id]
    if not matches:
        raise HTTPException(status_code=404, detail=f"Song id={song_id} not found")

    app_state["songs"] = [s for s in songs if s["id"] != song_id]

    with open(RESULTS_PATH, "w") as f:
        json.dump(app_state["songs"], f, indent=2)

    return {"deleted": song_id}
