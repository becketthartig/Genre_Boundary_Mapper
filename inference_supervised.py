"""
inference_supervised.py
-----------------------
Handles new song uploads in supervised mode with LDA.

New songs are projected into the existing LDA space. Boundary score
cannot be computed without re-fitting the full KNN, so uploads get a null score.

Usage (standalone test):
  python inference_supervised.py --wav my_new_song.wav --pipeline pipeline.pkl

Used by FastAPI:
  from inference_supervised import load_pipeline, process_new_song
"""

import json
import pickle
import argparse
import numpy as np
from pathlib import Path


def load_pipeline(pkl_path: str = "pipeline.pkl") -> dict:
    """
    Load the fitted pipeline from disk.
    
    Returns dict with keys: scaler, lda, feat_cols
    """
    with open(pkl_path, "rb") as f:
        pipeline = pickle.load(f)

    required_keys = {"scaler", "lda", "feat_cols"}
    if not required_keys.issubset(pipeline.keys()):
        raise ValueError(f"pipeline.pkl is missing keys: {required_keys - pipeline.keys()}")

    print(f"Pipeline loaded: {len(pipeline['feat_cols'])} features, "
          f"LDA with {pipeline['lda'].n_components} components")
    return pipeline


def process_new_song(
    filename: str,
    feature_list: list[float],
    pipeline: dict,
) -> dict:
    """
    Process a single new song through the fitted LDA pipeline.

    Args:
        filename     - original filename
        feature_list - output of extractor.extract_features(wav_path)
                       must match the feature columns used in training
        pipeline     - loaded pipeline dict from load_pipeline()

    Returns:
        dict with id, filename, genre (unknown), x, y, boundary_score (null), features
    """
    scaler = pipeline["scaler"]
    lda = pipeline["lda"]
    feat_cols = pipeline["feat_cols"]

    # Validate feature count
    if len(feature_list) != len(feat_cols):
        raise ValueError(
            f"Feature count mismatch: extractor returned {len(feature_list)} features "
            f"but pipeline expects {len(feat_cols)}. "
            f"Make sure extract_features() returns the same features in the same order."
        )

    # Build feature dict
    features = dict(zip(feat_cols, feature_list))

    # Normalize and project
    X = np.array(feature_list, dtype=np.float32).reshape(1, -1)
    X_scaled = scaler.transform(X)
    coords = lda.transform(X_scaled)
    
    # Handle case where LDA only has 1 component
    if coords.shape[1] == 1:
        x, y = float(coords[0, 0]), 0.0
    else:
        x, y = float(coords[0, 0]), float(coords[0, 1])

    return {
        "filename": filename,
        "genre": "unknown",  # no ground truth for uploads
        "x": x,
        "y": y,
        "boundary_score": None,  # can't compute without full KNN refit
        "features": features,
    }


def append_to_results(
    new_song: dict,
    results_path: str = "results.json",
) -> list[dict]:
    """Add a newly processed song to the results JSON and save."""
    with open(results_path, "r") as f:
        songs = json.load(f)

    max_id = max(s["id"] for s in songs) if songs else -1
    new_song["id"] = max_id + 1
    songs.append(new_song)

    with open(results_path, "w") as f:
        json.dump(songs, f, indent=2)

    print(f"Added '{new_song['filename']}' as id={new_song['id']} → genre: {new_song['genre']}")
    return songs


# ─── Standalone test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", required=True, help="Path to .wav file to process")
    parser.add_argument("--pipeline", default="pipeline.pkl")
    parser.add_argument("--results", default="results.json")
    args = parser.parse_args()

    # Import your extractor
    import sys
    sys.path.insert(0, ".")
    from extractor import extract_features

    print(f"Extracting features from {args.wav}...")
    feature_list = extract_features(args.wav)

    pipeline = load_pipeline(args.pipeline)
    result = process_new_song(
        filename=Path(args.wav).name,
        feature_list=feature_list,
        pipeline=pipeline,
    )

    print(f"\nResult:")
    print(f"  Genre:       {result['genre']}")
    print(f"  Position:    ({result['x']:.3f}, {result['y']:.3f})")
    print(f"  Features:    {len(result['features'])} extracted")

    append_to_results(result, args.results)
