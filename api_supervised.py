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
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
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
    features: list[str]  # list of feature names to include
    method: str = "umap-supervised"  # lda, umap-supervised, or umap-unsupervised in PCA


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


@app.get("/scaler")
def get_scaler() -> dict:
    """
    Returns StandardScaler parameters (mean and scale) for feature normalization.
    
    Frontend can use these to normalize features before computing distances,
    ensuring distances match the backend's boundary score computation.
    
    Response shape:
      {
        "mean": [...],      # mean for each feature
        "scale": [...],     # scale (std) for each feature  
        "features": [...]   # feature names in order
      }
    """
    pipeline = app_state.get("pipeline")
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")
    
    scaler = pipeline["scaler"]
    
    return {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "features": pipeline["feat_cols"]
    }


@app.get("/genre-means")
def get_genre_means() -> dict:
    """
    Returns the mean feature values for each genre in normalized space.
    
    Used to show how much a song deviates from its genre's typical characteristics.
    
    Response shape:
      {
        "blues": {"tempo": 0.23, "spectral_centroid_mean": -0.45, ...},
        "jazz": {"tempo": 0.67, "spectral_centroid_mean": 1.2, ...},
        ...
      }
    """
    import numpy as np
    import pandas as pd
    
    df = app_state.get("df")
    pipeline = app_state.get("pipeline")
    
    if df is None or pipeline is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    feat_cols = pipeline["feat_cols"]
    scaler = pipeline["scaler"]
    
    # Get normalized features
    X = df[feat_cols].values.astype(np.float32)
    X_scaled = scaler.transform(X)
    
    # Compute mean for each genre
    genre_means = {}
    for genre in df['label'].unique():
        genre_mask = df['label'] == genre
        genre_features_scaled = X_scaled[genre_mask]
        genre_mean = np.mean(genre_features_scaled, axis=0)
        
        genre_means[genre] = {
            feat: float(genre_mean[i]) 
            for i, feat in enumerate(feat_cols)
        }
    
    return genre_means


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
    
    Checks two locations:
    1. Data/uploads/{filename} for uploaded songs
    2. Data/genres_original/{genre}/{filename} for dataset songs
    
    Returns 404 if file doesn't exist in either location.
    """
    from pathlib import Path
    
    # Try uploads first (for user-uploaded songs)
    upload_path = Path("Data/uploads") / filename
    if upload_path.exists():
        return FileResponse(
            path=str(upload_path),
            media_type="audio/wav",
            headers={
                "Accept-Ranges": "bytes",
                "Cache-Control": "public, max-age=3600",
            }
        )
    
    # Fall back to original dataset
    # Extract genre from filename (e.g., 'blues.00008.wav' -> 'blues')
    parts = filename.split('.')
    if len(parts) < 2:
        raise HTTPException(status_code=404, detail=f"Invalid filename format: {filename}")
    
    genre = parts[0]
    audio_path = Path(AUDIO_BASE_PATH) / genre / filename
    
    if not audio_path.exists():
        raise HTTPException(
            status_code=404, 
            detail=f"Audio file not found in uploads or dataset: {filename}"
        )
    
    return FileResponse(
        path=str(audio_path),
        media_type="audio/wav",
        headers={
            "Accept-Ranges": "bytes",
            "Cache-Control": "public, max-age=3600",
        }
    )


@app.post("/upload")
async def upload_song(file: UploadFile = File(...), genre: str = Form("uploaded")) -> dict:
    """
    Upload a new song, extract features, save to dataset, and project into visualization.
    
    Args:
        file: WAV audio file
        genre: Genre label (optional, defaults to "uploaded")
    
    Process:
    1. Save audio file to Data/uploads/
    2. Extract 57 features using standardized extractor
    3. Append to features CSV with genre label
    4. Project into existing dimensionality reduction space
    5. Recompute boundary scores for all songs
    6. Return song object with coordinates and boundary score
    """
    from datetime import datetime
    import numpy as np
    import pandas as pd

    if not file.filename.endswith('.wav'):
        raise HTTPException(status_code=422, detail="Only WAV files supported")
    
    # Validate genre
    genre = genre.strip().lower()
    if not genre:
        genre = "uploaded"
    
    # Import feature extractor
    try:
        from feature_extractor import extract_features, get_feature_names
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="feature_extractor.py not found. Make sure it's in the same directory as the API."
        )
    
    # Save uploaded file
    upload_dir = Path("Data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = Path(file.filename).stem
    safe_name = "".join(c for c in base_name if c.isalnum() or c in ('-', '_'))
    audio_filename = f"{safe_name}_{timestamp}.wav"
    audio_path = upload_dir / audio_filename
    
    # Save audio file
    with open(audio_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    print(f"Saved upload to {audio_path}")
    
    # Extract features
    try:
        feature_list = extract_features(str(audio_path))
        feature_names = get_feature_names()
        
        if len(feature_list) != len(feature_names):
            raise ValueError(f"Feature count mismatch: got {len(feature_list)}, expected {len(feature_names)}")
        
    except Exception as e:
        # Clean up file if extraction fails
        audio_path.unlink()
        raise HTTPException(status_code=422, detail=f"Feature extraction failed: {str(e)}")
    
    # Load pipeline
    pipeline = app_state.get("pipeline")
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")
    
    # Build feature dict
    feature_dict = dict(zip(feature_names, feature_list))
    
    # Get features in pipeline order
    pipeline_features = pipeline["feat_cols"]
    feature_list_filtered = [feature_dict[f] for f in pipeline_features]
    
    # Project into visualization space
    scaler = pipeline["scaler"]
    model = pipeline.get("model") or pipeline.get("lda") or pipeline.get("umap")
    if not model:
        raise HTTPException(status_code=503, detail="Model not found in pipeline")
    
    X = np.array(feature_list_filtered, dtype=np.float32).reshape(1, -1)
    X_scaled = scaler.transform(X)
    coords = model.transform(X_scaled)
    
    if coords.shape[1] == 1:
        x, y = float(coords[0, 0]), 0.0
    else:
        x, y = float(coords[0, 0]), float(coords[0, 1])
    
    # Append to CSV
    csv_path = Path(CSV_PATH)
    df = pd.read_csv(csv_path)
    
    new_row = {
        'filename': audio_filename,
        'length': 30.0,  # Assume 30 seconds
        'label': genre,  # Use provided genre label
    }
    
    # Add all features
    for name, val in zip(feature_names, feature_list):
        new_row[name] = val
    
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    # Save updated CSV (with backup)
    backup_path = csv_path.with_suffix('.csv.backup')
    if csv_path.exists():
        shutil.copy(csv_path, backup_path)
    df.to_csv(csv_path, index=False)
    
    print(f"Appended to CSV: {audio_filename} (genre: {genre})")
    
    # Update in-memory state
    app_state["df"] = df
    
    # Recompute boundary scores for ALL songs (including the new one)
    print("Recomputing boundary scores...")
    X_all = df[pipeline_features].values.astype(np.float32)
    y_all = df['label'].values
    X_all_scaled = scaler.transform(X_all)
    
    from sklearn.neighbors import NearestNeighbors
    k = 21
    nn = NearestNeighbors(n_neighbors=min(k, len(df)), metric='euclidean')
    nn.fit(X_all_scaled)
    distances, indices = nn.kneighbors(X_all_scaled)
    
    boundary_scores = np.zeros(len(df))
    boundary_neighbor_ids = np.full(len(df), -1, dtype=int)  # Store the neighbor ID
    within_genre_dists = []
    
    for i in range(len(df)):
        my_label = y_all[i]
        neighbor_labels = y_all[indices[i]]
        neighbor_dists = distances[i]
        
        cross_genre_mask = neighbor_labels != my_label
        if cross_genre_mask.any():
            cross_genre_dist = neighbor_dists[cross_genre_mask][0]
            # Store the ID of the boundary neighbor
            cross_genre_idx = np.where(cross_genre_mask)[0][0]
            boundary_neighbor_ids[i] = int(indices[i][cross_genre_idx])
        else:
            cross_genre_dist = neighbor_dists[-1] if len(neighbor_dists) > 0 else 1.0
            boundary_neighbor_ids[i] = -1
        
        boundary_scores[i] = cross_genre_dist
        
        within_genre_mask = (neighbor_labels == my_label) & (np.arange(len(neighbor_labels)) > 0)
        if within_genre_mask.any():
            within_genre_dists.append(neighbor_dists[within_genre_mask][0])
    
    # Normalize boundary scores
    if len(within_genre_dists) > 0:
        median_within = np.median(within_genre_dists)
        boundary_scores = boundary_scores / median_within
    
    # Rebuild all songs with updated boundary scores
    songs = []
    for i, row in df.reset_index(drop=True).iterrows():
        song = {
            "id": i,
            "filename": row["filename"],
            "genre": row.get("label", "unknown"),
            "x": float(coords[i, 0]) if i < len(coords) else x,  # Use projected coords for all
            "y": float(coords[i, 1]) if i < len(coords) and coords.shape[1] > 1 else (y if i == len(df) - 1 else 0.0),
            "boundary_score": float(boundary_scores[i]),
            "boundary_neighbor_id": int(boundary_neighbor_ids[i]),
            "features": {f: float(row[f]) for f in pipeline_features},
        }
        songs.append(song)
    
    # Actually, we need to reproject ALL songs, not just use old coords
    # Let me fix this properly:
    coords_all = model.transform(X_all_scaled)
    if coords_all.shape[1] == 1:
        coords_all = np.hstack([coords_all, np.zeros((coords_all.shape[0], 1))])
    
    songs = []
    for i, row in df.reset_index(drop=True).iterrows():
        song = {
            "id": i,
            "filename": row["filename"],
            "genre": row.get("label", "unknown"),
            "x": float(coords_all[i, 0]),
            "y": float(coords_all[i, 1]),
            "boundary_score": float(boundary_scores[i]),
            "boundary_neighbor_id": int(boundary_neighbor_ids[i]),  # Add this
            "features": {f: float(row[f]) for f in pipeline_features},
        }
        songs.append(song)
    
    # Update in-memory songs list
    app_state["songs"] = songs
    
    # Save updated results.json
    with open(RESULTS_PATH, "w") as f:
        json.dump(songs, f, indent=2)
    
    # Return the newly added song
    new_song = songs[-1]
    print(f"Created song object with id={new_song['id']}, boundary_score={new_song['boundary_score']:.2f}")
    
    return new_song


@app.post("/recompute")
def recompute_pca(req: RecomputeRequest) -> dict:
    """
    Recompute dimensionality reduction using a different subset of features and/or method.
    
    This regenerates results.json and pipeline.pkl without needing
    to re-run the command-line script.
    
    Returns the new feature set and method info.
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
    
    # Validate method
    valid_methods = ["lda", "umap-supervised", "umap-unsupervised"]
    if req.method not in valid_methods:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid method: {req.method}. Must be one of: {valid_methods}"
        )
    
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
    
    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Run chosen method
    if req.method == "lda":
        n_classes = len(np.unique(y))
        max_components = min(2, n_classes - 1)
        
        model = LinearDiscriminantAnalysis(n_components=max_components)
        coords_2d = model.fit_transform(X_scaled, y)
        
        if coords_2d.shape[1] == 1:
            coords_2d = np.hstack([coords_2d, np.zeros((coords_2d.shape[0], 1))])
        
        explained = model.explained_variance_ratio_.sum() * 100
        variance_info = {
            "explained_variance": round(explained, 2),
            "ld1_variance": round(model.explained_variance_ratio_[0] * 100, 2) if max_components >= 1 else 0,
            "ld2_variance": round(model.explained_variance_ratio_[1] * 100, 2) if max_components >= 2 else 0,
        }
    
    elif req.method == "umap-supervised":
        try:
            import umap
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="umap-learn not installed. Install with: pip install umap-learn"
            )
        
        # Convert string labels to integers for UMAP
        unique_labels = np.unique(y)
        label_to_int = {label: i for i, label in enumerate(unique_labels)}
        y_int = np.array([label_to_int[label] for label in y])
        
        model = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric='euclidean',
            target_weight=0.5,
            random_state=42,
        )
        coords_2d = model.fit_transform(X_scaled, y=y_int)
        variance_info = {"method": "umap-supervised", "target_weight": 0.5}
    
    elif req.method == "umap-unsupervised":
        try:
            import umap
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="umap-learn not installed. Install with: pip install umap-learn"
            )
        
        model = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric='euclidean',
            random_state=42,
        )
        coords_2d = model.fit_transform(X_scaled)
        variance_info = {"method": "umap-unsupervised"}
    
    # Compute boundary scores
    k = 21
    nn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    nn.fit(X_scaled)
    distances, indices = nn.kneighbors(X_scaled)
    
    boundary_scores = np.zeros(len(y))
    boundary_neighbor_ids = np.full(len(y), -1, dtype=int)
    within_genre_dists = []
    
    for i in range(len(y)):
        my_label = y[i]
        neighbor_labels = y[indices[i]]
        neighbor_dists = distances[i]
        
        cross_genre_mask = neighbor_labels != my_label
        if cross_genre_mask.any():
            cross_genre_dist = neighbor_dists[cross_genre_mask][0]
            # Store the ID of the boundary neighbor
            cross_genre_idx = np.where(cross_genre_mask)[0][0]
            boundary_neighbor_ids[i] = int(indices[i][cross_genre_idx])
        else:
            cross_genre_dist = neighbor_dists[-1]
            boundary_neighbor_ids[i] = -1
        
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
            "boundary_neighbor_id": int(boundary_neighbor_ids[i]),
            "features": {col: float(row[col]) for col in req.features},
        }
        songs.append(song)
    
    # Save updated results and pipeline
    with open(RESULTS_PATH, "w") as f:
        json.dump(songs, f, indent=2)
    
    pipeline_state = {
        "scaler": scaler,
        "model": model,
        "feat_cols": req.features,
        "method": req.method,
    }
    with open(PIPELINE_PATH, "wb") as f:
        pickle.dump(pipeline_state, f)
    
    # Update in-memory state
    app_state["songs"] = songs
    app_state["pipeline"] = pipeline_state
    
    return {
        "method": req.method,
        "features_used": len(req.features),
        **variance_info
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


@app.get("/umap3d")
def get_umap3d() -> list[dict]:
    """
    Compute and return 3D UMAP projection of all songs.
    
    This is separate from the main 2D visualization and uses different
    UMAP parameters. Results are cached after first computation.
    
    Returns:
        List of songs with x3d, y3d, z3d coordinates
    """
    # Return cached result if available
    if "umap3d_cache" in app_state:
        return app_state["umap3d_cache"]

    import numpy as np
    from umap import UMAP

    songs = app_state.get("songs", [])
    pipeline = app_state.get("pipeline")

    if not songs or not pipeline:
        raise HTTPException(status_code=503, detail="Data not loaded")

    feat_cols = pipeline["feat_cols"]

    # Extract features from songs
    X = []
    valid_songs = []
    for s in songs:
        if s.get("features"):
            row = [s["features"].get(f, 0.0) for f in feat_cols]
            X.append(row)
            valid_songs.append(s)

    if not X:
        raise HTTPException(status_code=422, detail="No feature data available")

    X = np.array(X, dtype=np.float32)
    X_scaled = pipeline["scaler"].transform(X)

    # Compute 3D UMAP
    reducer = UMAP(
        n_components=3,
        n_neighbors=15,
        min_dist=0.1,
        random_state=42,
        verbose=False,
    )
    coords = reducer.fit_transform(X_scaled)

    # Build result
    result = []
    for i, s in enumerate(valid_songs):
        result.append({
            "id": s["id"],
            "filename": s["filename"],
            "genre": s["genre"],
            "x3d": float(coords[i, 0]),
            "y3d": float(coords[i, 1]),
            "z3d": float(coords[i, 2]),
        })

    # Cache result
    app_state["umap3d_cache"] = result
    return result


# """
# api_supervised.py
# -----------------
# FastAPI backend for supervised genre visualization.

# Colors by true genre labels, no clustering.
# Supports dynamic feature selection and PCA re-computation.

# Endpoints:
#   GET  /songs          → all songs with PCA coords and genre labels
#   GET  /features       → list of all available feature names
#   POST /upload         → process a new .wav file
#   POST /recompute      → recompute PCA with different feature selection
#   GET  /health         → status check

# Run with:
#   uvicorn api_supervised:app --reload --port 8000

# Install deps:
#   pip install fastapi uvicorn python-multipart
# """

# import json
# import shutil
# import tempfile
# from pathlib import Path
# from contextlib import asynccontextmanager
# from typing import Any

# from fastapi import FastAPI, UploadFile, File, HTTPException, Form
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import FileResponse
# from pydantic import BaseModel

# from inference_supervised import load_pipeline, process_new_song, append_to_results

# # ─── Config ───────────────────────────────────────────────────────────────────

# RESULTS_PATH = "results.json"
# PIPELINE_PATH = "pipeline.pkl"
# CSV_PATH = "Data/features_30_sec.csv"  # needed for recompute
# AUDIO_BASE_PATH = "Data/genres_original"  # path to audio files relative to where API runs

# # ─── App state ────────────────────────────────────────────────────────────────

# app_state: dict = {}


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """Load pipeline and results at startup."""
#     print("Loading pipeline...")
#     app_state["pipeline"] = load_pipeline(PIPELINE_PATH)

#     print("Loading results...")
#     with open(RESULTS_PATH) as f:
#         app_state["songs"] = json.load(f)

#     # Load the original CSV for recompute endpoint
#     import pandas as pd
#     app_state["df"] = pd.read_csv(CSV_PATH)

#     print(f"Ready. {len(app_state['songs'])} songs loaded.")
#     yield


# app = FastAPI(
#     title="Genre Visualizer API (Supervised)",
#     lifespan=lifespan,
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000", "http://localhost:5173"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# # ─── Request models ───────────────────────────────────────────────────────────

# class RecomputeRequest(BaseModel):
#     features: list[str]  # list of feature names to include
#     method: str = "umap-supervised"  # lda, umap-supervised, or umap-unsupervised in PCA


# # ─── Routes ───────────────────────────────────────────────────────────────────

# @app.get("/health")
# def health():
#     pipeline = app_state.get("pipeline")
#     songs = app_state.get("songs", [])
#     genres = list(set(s["genre"] for s in songs))
#     return {
#         "status": "ok",
#         "mode": "supervised",
#         "songs_loaded": len(songs),
#         "genres": len(genres),
#         "features_in_use": len(pipeline["feat_cols"]) if pipeline else None,
#     }


# @app.get("/songs")
# def get_songs() -> list[dict]:
#     """Returns all songs with PCA coords and genre labels."""
#     return app_state["songs"]


# @app.get("/features")
# def get_features() -> dict:
#     """
#     Returns available features and which ones are currently in use.
    
#     Response shape:
#       {
#         "all": [...],      # all features from the CSV
#         "active": [...]    # features currently used in PCA
#       }
#     """
#     df = app_state.get("df")
#     pipeline = app_state.get("pipeline")
    
#     if df is None or pipeline is None:
#         raise HTTPException(status_code=503, detail="Data not loaded")
    
#     exclude = {"filename", "length", "label"}
#     all_features = [c for c in df.columns if c not in exclude]
#     active_features = pipeline["feat_cols"]
    
#     return {
#         "all": all_features,
#         "active": active_features,
#     }


# @app.get("/scaler")
# def get_scaler() -> dict:
#     """
#     Returns StandardScaler parameters (mean and scale) for feature normalization.
    
#     Frontend can use these to normalize features before computing distances,
#     ensuring distances match the backend's boundary score computation.
    
#     Response shape:
#       {
#         "mean": [...],      # mean for each feature
#         "scale": [...],     # scale (std) for each feature  
#         "features": [...]   # feature names in order
#       }
#     """
#     pipeline = app_state.get("pipeline")
    
#     if pipeline is None:
#         raise HTTPException(status_code=503, detail="Pipeline not loaded")
    
#     scaler = pipeline["scaler"]
    
#     return {
#         "mean": scaler.mean_.tolist(),
#         "scale": scaler.scale_.tolist(),
#         "features": pipeline["feat_cols"]
#     }


# @app.get("/songs/{song_id}")
# def get_song(song_id: int) -> dict:
#     songs = app_state["songs"]
#     matches = [s for s in songs if s["id"] == song_id]
#     if not matches:
#         raise HTTPException(status_code=404, detail=f"Song id={song_id} not found")
#     return matches[0]


# @app.get("/genre/{genre_name}")
# def get_genre(genre_name: str) -> list[dict]:
#     """All songs with a given genre label."""
#     return [s for s in app_state["songs"] if s["genre"].lower() == genre_name.lower()]


# @app.get("/audio/{filename}")
# def get_audio(filename: str):
#     """
#     Serve audio file for a given filename.
    
#     Checks two locations:
#     1. Data/uploads/{filename} for uploaded songs
#     2. Data/genres_original/{genre}/{filename} for dataset songs
    
#     Returns 404 if file doesn't exist in either location.
#     """
#     from pathlib import Path
    
#     # Try uploads first (for user-uploaded songs)
#     upload_path = Path("Data/uploads") / filename
#     if upload_path.exists():
#         return FileResponse(
#             path=str(upload_path),
#             media_type="audio/wav",
#             headers={
#                 "Accept-Ranges": "bytes",
#                 "Cache-Control": "public, max-age=3600",
#             }
#         )
    
#     # Fall back to original dataset
#     # Extract genre from filename (e.g., 'blues.00008.wav' -> 'blues')
#     parts = filename.split('.')
#     if len(parts) < 2:
#         raise HTTPException(status_code=404, detail=f"Invalid filename format: {filename}")
    
#     genre = parts[0]
#     audio_path = Path(AUDIO_BASE_PATH) / genre / filename
    
#     if not audio_path.exists():
#         raise HTTPException(
#             status_code=404, 
#             detail=f"Audio file not found in uploads or dataset: {filename}"
#         )
    
#     return FileResponse(
#         path=str(audio_path),
#         media_type="audio/wav",
#         headers={
#             "Accept-Ranges": "bytes",
#             "Cache-Control": "public, max-age=3600",
#         }
#     )


# @app.post("/upload")
# async def upload_song(file: UploadFile = File(...), genre: str = Form("uploaded")) -> dict:
#     """
#     Upload a new song, extract features, save to dataset, and project into visualization.
    
#     Args:
#         file: WAV audio file
#         genre: Genre label (optional, defaults to "uploaded")
    
#     Process:
#     1. Save audio file to Data/uploads/
#     2. Extract 57 features using standardized extractor
#     3. Append to features CSV with genre label
#     4. Project into existing dimensionality reduction space
#     5. Recompute boundary scores for all songs
#     6. Return song object with coordinates and boundary score
#     """
#     if not file.filename.endswith('.wav'):
#         raise HTTPException(status_code=422, detail="Only WAV files supported")
    
#     # Validate genre
#     genre = genre.strip().lower()
#     if not genre:
#         genre = "uploaded"
    
#     # Import feature extractor
#     try:
#         from feature_extractor import extract_features, get_feature_names
#     except ImportError:
#         raise HTTPException(
#             status_code=500,
#             detail="feature_extractor.py not found. Make sure it's in the same directory as the API."
#         )
    
#     # Save uploaded file
#     upload_dir = Path("Data/uploads")
#     upload_dir.mkdir(parents=True, exist_ok=True)
    
#     # Generate unique filename
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     base_name = Path(file.filename).stem
#     safe_name = "".join(c for c in base_name if c.isalnum() or c in ('-', '_'))
#     audio_filename = f"{safe_name}_{timestamp}.wav"
#     audio_path = upload_dir / audio_filename
    
#     # Save audio file
#     with open(audio_path, "wb") as f:
#         content = await file.read()
#         f.write(content)
    
#     print(f"Saved upload to {audio_path}")
    
#     # Extract features
#     try:
#         feature_list = extract_features(str(audio_path))
#         feature_names = get_feature_names()
        
#         if len(feature_list) != len(feature_names):
#             raise ValueError(f"Feature count mismatch: got {len(feature_list)}, expected {len(feature_names)}")
        
#     except Exception as e:
#         # Clean up file if extraction fails
#         audio_path.unlink()
#         raise HTTPException(status_code=422, detail=f"Feature extraction failed: {str(e)}")
    
#     # Load pipeline
#     pipeline = app_state.get("pipeline")
#     if not pipeline:
#         raise HTTPException(status_code=503, detail="Pipeline not loaded")
    
#     # Build feature dict
#     feature_dict = dict(zip(feature_names, feature_list))
    
#     # Get features in pipeline order
#     pipeline_features = pipeline["feat_cols"]
#     feature_list_filtered = [feature_dict[f] for f in pipeline_features]
    
#     # Project into visualization space
#     scaler = pipeline["scaler"]
#     model = pipeline.get("model") or pipeline.get("lda") or pipeline.get("umap")
#     if not model:
#         raise HTTPException(status_code=503, detail="Model not found in pipeline")
    
#     X = np.array(feature_list_filtered, dtype=np.float32).reshape(1, -1)
#     X_scaled = scaler.transform(X)
#     coords = model.transform(X_scaled)
    
#     if coords.shape[1] == 1:
#         x, y = float(coords[0, 0]), 0.0
#     else:
#         x, y = float(coords[0, 0]), float(coords[0, 1])
    
#     # Append to CSV
#     csv_path = Path(CSV_PATH)
#     df = pd.read_csv(csv_path)
    
#     new_row = {
#         'filename': audio_filename,
#         'length': 30.0,  # Assume 30 seconds
#         'label': genre,  # Use provided genre label
#     }
    
#     # Add all features
#     for name, val in zip(feature_names, feature_list):
#         new_row[name] = val
    
#     df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
#     # Save updated CSV (with backup)
#     backup_path = csv_path.with_suffix('.csv.backup')
#     if csv_path.exists():
#         shutil.copy(csv_path, backup_path)
#     df.to_csv(csv_path, index=False)
    
#     print(f"Appended to CSV: {audio_filename} (genre: {genre})")
    
#     # Update in-memory state
#     app_state["df"] = df
    
#     # Recompute boundary scores for ALL songs (including the new one)
#     print("Recomputing boundary scores...")
#     X_all = df[pipeline_features].values.astype(np.float32)
#     y_all = df['label'].values
#     X_all_scaled = scaler.transform(X_all)
    
#     from sklearn.neighbors import NearestNeighbors
#     k = 21
#     nn = NearestNeighbors(n_neighbors=min(k, len(df)), metric='euclidean')
#     nn.fit(X_all_scaled)
#     distances, indices = nn.kneighbors(X_all_scaled)
    
#     boundary_scores = np.zeros(len(df))
#     within_genre_dists = []
    
#     for i in range(len(df)):
#         my_label = y_all[i]
#         neighbor_labels = y_all[indices[i]]
#         neighbor_dists = distances[i]
        
#         cross_genre_mask = neighbor_labels != my_label
#         if cross_genre_mask.any():
#             cross_genre_dist = neighbor_dists[cross_genre_mask][0]
#         else:
#             cross_genre_dist = neighbor_dists[-1] if len(neighbor_dists) > 0 else 1.0
        
#         boundary_scores[i] = cross_genre_dist
        
#         within_genre_mask = (neighbor_labels == my_label) & (np.arange(len(neighbor_labels)) > 0)
#         if within_genre_mask.any():
#             within_genre_dists.append(neighbor_dists[within_genre_mask][0])
    
#     # Normalize boundary scores
#     if len(within_genre_dists) > 0:
#         median_within = np.median(within_genre_dists)
#         boundary_scores = boundary_scores / median_within
    
#     # Rebuild all songs with updated boundary scores
#     songs = []
#     for i, row in df.reset_index(drop=True).iterrows():
#         song = {
#             "id": i,
#             "filename": row["filename"],
#             "genre": row.get("label", "unknown"),
#             "x": float(coords[i, 0]) if i < len(coords) else x,  # Use projected coords for all
#             "y": float(coords[i, 1]) if i < len(coords) and coords.shape[1] > 1 else (y if i == len(df) - 1 else 0.0),
#             "boundary_score": float(boundary_scores[i]),
#             "features": {f: float(row[f]) for f in pipeline_features},
#         }
#         songs.append(song)
    
#     # Actually, we need to reproject ALL songs, not just use old coords
#     # Let me fix this properly:
#     coords_all = model.transform(X_all_scaled)
#     if coords_all.shape[1] == 1:
#         coords_all = np.hstack([coords_all, np.zeros((coords_all.shape[0], 1))])
    
#     songs = []
#     for i, row in df.reset_index(drop=True).iterrows():
#         song = {
#             "id": i,
#             "filename": row["filename"],
#             "genre": row.get("label", "unknown"),
#             "x": float(coords_all[i, 0]),
#             "y": float(coords_all[i, 1]),
#             "boundary_score": float(boundary_scores[i]),
#             "features": {f: float(row[f]) for f in pipeline_features},
#         }
#         songs.append(song)
    
#     # Update in-memory songs list
#     app_state["songs"] = songs
    
#     # Save updated results.json
#     with open(RESULTS_PATH, "w") as f:
#         json.dump(songs, f, indent=2)
    
#     # Return the newly added song
#     new_song = songs[-1]
#     print(f"Created song object with id={new_song['id']}, boundary_score={new_song['boundary_score']:.2f}")
    
#     return new_song


# @app.post("/recompute")
# def recompute_pca(req: RecomputeRequest) -> dict:
#     """
#     Recompute dimensionality reduction using a different subset of features and/or method.
    
#     This regenerates results.json and pipeline.pkl without needing
#     to re-run the command-line script.
    
#     Returns the new feature set and method info.
#     """
#     import numpy as np
#     import pandas as pd
#     from sklearn.preprocessing import StandardScaler
#     from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#     from sklearn.neighbors import NearestNeighbors
#     import pickle
    
#     df = app_state.get("df")
#     if df is None:
#         raise HTTPException(status_code=503, detail="CSV data not loaded")
    
#     # Validate method
#     valid_methods = ["lda", "umap-supervised", "umap-unsupervised"]
#     if req.method not in valid_methods:
#         raise HTTPException(
#             status_code=422,
#             detail=f"Invalid method: {req.method}. Must be one of: {valid_methods}"
#         )
    
#     # Validate requested features
#     exclude = {"filename", "length", "label"}
#     all_features = [c for c in df.columns if c not in exclude]
#     invalid = [f for f in req.features if f not in all_features]
#     if invalid:
#         raise HTTPException(
#             status_code=422,
#             detail=f"Invalid features: {invalid}. Must be one of: {all_features[:10]}..."
#         )
    
#     if not req.features:
#         raise HTTPException(status_code=422, detail="Must specify at least one feature")
    
#     # Extract feature matrix
#     X = df[req.features].values.astype(np.float32)
#     y = df["label"].values
    
#     # Normalize
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
    
#     # Run chosen method
#     if req.method == "lda":
#         n_classes = len(np.unique(y))
#         max_components = min(2, n_classes - 1)
        
#         model = LinearDiscriminantAnalysis(n_components=max_components)
#         coords_2d = model.fit_transform(X_scaled, y)
        
#         if coords_2d.shape[1] == 1:
#             coords_2d = np.hstack([coords_2d, np.zeros((coords_2d.shape[0], 1))])
        
#         explained = model.explained_variance_ratio_.sum() * 100
#         variance_info = {
#             "explained_variance": round(explained, 2),
#             "ld1_variance": round(model.explained_variance_ratio_[0] * 100, 2) if max_components >= 1 else 0,
#             "ld2_variance": round(model.explained_variance_ratio_[1] * 100, 2) if max_components >= 2 else 0,
#         }
    
#     elif req.method == "umap-supervised":
#         try:
#             import umap
#         except ImportError:
#             raise HTTPException(
#                 status_code=500,
#                 detail="umap-learn not installed. Install with: pip install umap-learn"
#             )
        
#         # Convert string labels to integers for UMAP
#         unique_labels = np.unique(y)
#         label_to_int = {label: i for i, label in enumerate(unique_labels)}
#         y_int = np.array([label_to_int[label] for label in y])
        
#         model = umap.UMAP(
#             n_components=2,
#             n_neighbors=15,
#             min_dist=0.1,
#             metric='euclidean',
#             target_weight=0.5,
#             random_state=42,
#         )
#         coords_2d = model.fit_transform(X_scaled, y=y_int)
#         variance_info = {"method": "umap-supervised", "target_weight": 0.5}
    
#     elif req.method == "umap-unsupervised":
#         try:
#             import umap
#         except ImportError:
#             raise HTTPException(
#                 status_code=500,
#                 detail="umap-learn not installed. Install with: pip install umap-learn"
#             )
        
#         model = umap.UMAP(
#             n_components=2,
#             n_neighbors=15,
#             min_dist=0.1,
#             metric='euclidean',
#             random_state=42,
#         )
#         coords_2d = model.fit_transform(X_scaled)
#         variance_info = {"method": "umap-unsupervised"}
    
#     # Compute boundary scores
#     k = 21
#     nn = NearestNeighbors(n_neighbors=k, metric='euclidean')
#     nn.fit(X_scaled)
#     distances, indices = nn.kneighbors(X_scaled)
    
#     boundary_scores = np.zeros(len(y))
#     within_genre_dists = []
    
#     for i in range(len(y)):
#         my_label = y[i]
#         neighbor_labels = y[indices[i]]
#         neighbor_dists = distances[i]
        
#         cross_genre_mask = neighbor_labels != my_label
#         if cross_genre_mask.any():
#             cross_genre_dist = neighbor_dists[cross_genre_mask][0]
#         else:
#             cross_genre_dist = neighbor_dists[-1]
        
#         boundary_scores[i] = cross_genre_dist
        
#         within_genre_mask = (neighbor_labels == my_label) & (np.arange(len(neighbor_labels)) > 0)
#         if within_genre_mask.any():
#             within_genre_dists.append(neighbor_dists[within_genre_mask][0])
    
#     median_within = np.median(within_genre_dists)
#     boundary_scores = boundary_scores / median_within
    
#     # Rebuild songs with new coords and boundary scores
#     songs = []
#     for i, row in df.reset_index(drop=True).iterrows():
#         song = {
#             "id": i,
#             "filename": row["filename"],
#             "genre": row.get("label", "unknown"),
#             "x": float(coords_2d[i, 0]),
#             "y": float(coords_2d[i, 1]),
#             "boundary_score": float(boundary_scores[i]),
#             "features": {col: float(row[col]) for col in req.features},
#         }
#         songs.append(song)
    
#     # Save updated results and pipeline
#     with open(RESULTS_PATH, "w") as f:
#         json.dump(songs, f, indent=2)
    
#     pipeline_state = {
#         "scaler": scaler,
#         "model": model,
#         "feat_cols": req.features,
#         "method": req.method,
#     }
#     with open(PIPELINE_PATH, "wb") as f:
#         pickle.dump(pipeline_state, f)
    
#     # Update in-memory state
#     app_state["songs"] = songs
#     app_state["pipeline"] = pipeline_state
    
#     return {
#         "method": req.method,
#         "features_used": len(req.features),
#         **variance_info
#     }


# @app.delete("/songs/{song_id}")
# def delete_song(song_id: int) -> dict:
#     """Remove a song from the dataset."""
#     songs = app_state["songs"]
#     matches = [s for s in songs if s["id"] == song_id]
#     if not matches:
#         raise HTTPException(status_code=404, detail=f"Song id={song_id} not found")

#     app_state["songs"] = [s for s in songs if s["id"] != song_id]

#     with open(RESULTS_PATH, "w") as f:
#         json.dump(app_state["songs"], f, indent=2)

#     return {"deleted": song_id}


# @app.get("/umap3d")
# def get_umap3d() -> list[dict]:
#     """
#     Compute and return 3D UMAP projection of all songs.
    
#     This is separate from the main 2D visualization and uses different
#     UMAP parameters. Results are cached after first computation.
    
#     Returns:
#         List of songs with x3d, y3d, z3d coordinates
#     """
#     # Return cached result if available
#     if "umap3d_cache" in app_state:
#         return app_state["umap3d_cache"]

#     import numpy as np
#     from umap import UMAP

#     songs = app_state.get("songs", [])
#     pipeline = app_state.get("pipeline")

#     if not songs or not pipeline:
#         raise HTTPException(status_code=503, detail="Data not loaded")

#     feat_cols = pipeline["feat_cols"]

#     # Extract features from songs
#     X = []
#     valid_songs = []
#     for s in songs:
#         if s.get("features"):
#             row = [s["features"].get(f, 0.0) for f in feat_cols]
#             X.append(row)
#             valid_songs.append(s)

#     if not X:
#         raise HTTPException(status_code=422, detail="No feature data available")

#     X = np.array(X, dtype=np.float32)
#     X_scaled = pipeline["scaler"].transform(X)

#     # Compute 3D UMAP
#     reducer = UMAP(
#         n_components=3,
#         n_neighbors=15,
#         min_dist=0.1,
#         random_state=42,
#         verbose=False,
#     )
#     coords = reducer.fit_transform(X_scaled)

#     # Build result
#     result = []
#     for i, s in enumerate(valid_songs):
#         result.append({
#             "id": s["id"],
#             "filename": s["filename"],
#             "genre": s["genre"],
#             "x3d": float(coords[i, 0]),
#             "y3d": float(coords[i, 1]),
#             "z3d": float(coords[i, 2]),
#         })

#     # Cache result
#     app_state["umap3d_cache"] = result
#     return result