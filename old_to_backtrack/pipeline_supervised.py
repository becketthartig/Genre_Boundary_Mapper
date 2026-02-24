"""
pipeline_supervised.py
----------------------
Supervised genre visualization pipeline using PCA.

Colors points by the true genre labels from the CSV.
User can specify which features to include via --features flag.

Outputs:
  - results.json  → per-song data with x, y coords and genre labels
  - pipeline.pkl  → saved PCA + scaler + feature list for new uploads

Usage:
  # Use all features
  python pipeline_supervised.py --csv features.csv

  # Use only spectral features
  python pipeline_supervised.py --csv features.csv --features spectral_centroid_mean spectral_bandwidth_mean rolloff_mean

  # Use MFCCs only
  python pipeline_supervised.py --csv features.csv --features mfcc*
"""

import json
import pickle
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import fnmatch


EXCLUDE_COLS = {"filename", "length", "label"}


def load_data(csv_path: str) -> tuple[pd.DataFrame, list[str]]:
    """Load CSV and return full dataframe + available feature names."""
    df = pd.read_csv(csv_path)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    n_before = len(df)
    df.dropna(inplace=True)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"  Dropped {n_dropped} rows with NaN/inf values")

    all_features = [c for c in df.columns if c not in EXCLUDE_COLS]
    print(f"  Loaded {len(df)} songs")
    return df, all_features


def select_features(all_features: list[str], patterns: list[str] | None) -> list[str]:
    """
    Select features to use based on patterns.
    Supports wildcards, e.g., 'mfcc*' matches all MFCC columns.
    
    If patterns is None, use all features.
    """
    if not patterns:
        return all_features

    selected = []
    for pattern in patterns:
        if '*' in pattern:
            # Wildcard matching
            matches = [f for f in all_features if fnmatch.fnmatch(f, pattern)]
            selected.extend(matches)
        else:
            # Exact match
            if pattern in all_features:
                selected.append(pattern)
            else:
                print(f"  Warning: feature '{pattern}' not found in CSV, skipping")

    # Remove duplicates while preserving order
    selected = list(dict.fromkeys(selected))
    
    if not selected:
        raise ValueError("No features selected! Check your --features patterns.")
    
    return selected


def run_pca(X: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, PCA, StandardScaler]:
    """
    Normalize and run PCA.
    
    Returns:
        coords_2d - (n_samples, 2) array of 2D coordinates
        pca       - fitted PCA model
        scaler    - fitted StandardScaler
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=n_components, random_state=42)
    coords_2d = pca.fit_transform(X_scaled)
    
    explained = pca.explained_variance_ratio_.sum() * 100
    print(f"  PCA explained variance: {explained:.1f}%")
    print(f"  PC1: {pca.explained_variance_ratio_[0]*100:.1f}%")
    print(f"  PC2: {pca.explained_variance_ratio_[1]*100:.1f}%")
    
    return coords_2d, pca, scaler


def build_results(
    df: pd.DataFrame,
    feat_cols: list[str],
    coords_2d: np.ndarray,
) -> list[dict]:
    """
    Build the JSON output for the frontend.
    
    Each song gets:
      - id, filename, genre (the true label)
      - x, y (PCA coordinates)
      - features (dict of all selected features)
    """
    songs = []
    for i, row in df.reset_index(drop=True).iterrows():
        song = {
            "id": i,
            "filename": row["filename"],
            "genre": row.get("label", "unknown"),
            "x": float(coords_2d[i, 0]),
            "y": float(coords_2d[i, 1]),
            "features": {col: float(row[col]) for col in feat_cols},
        }
        songs.append(song)
    return songs


def save_pipeline(
    scaler: StandardScaler,
    pca: PCA,
    feat_cols: list[str],
    output_path: str = "pipeline.pkl",
) -> None:
    """Save the fitted pipeline for processing new uploads."""
    pipeline_state = {
        "scaler": scaler,
        "pca": pca,
        "feat_cols": feat_cols,
    }
    with open(output_path, "wb") as f:
        pickle.dump(pipeline_state, f)
    print(f"  Pipeline saved to {output_path}")


def print_genre_summary(songs: list[dict]):
    """Print per-genre breakdown."""
    from collections import Counter
    genres = [s["genre"] for s in songs]
    counts = Counter(genres)
    
    print(f"\n── Genre breakdown ───────────────────────────────────")
    for genre, count in sorted(counts.items()):
        print(f"  {genre:15s} {count:4d} songs")


def run_pipeline(
    csv_path: str,
    feature_patterns: list[str] | None,
    output_json: str,
    output_pkl: str,
) -> None:
    print("\n── Loading data ──────────────────────────────────────")
    df, all_features = load_data(csv_path)

    print("\n── Selecting features ────────────────────────────────")
    feat_cols = select_features(all_features, feature_patterns)
    print(f"  Using {len(feat_cols)} features:")
    for f in feat_cols[:10]:
        print(f"    - {f}")
    if len(feat_cols) > 10:
        print(f"    ... and {len(feat_cols) - 10} more")

    X = df[feat_cols].values.astype(np.float32)

    print("\n── Running PCA ───────────────────────────────────────")
    coords_2d, pca, scaler = run_pca(X)

    print("\n── Building results ──────────────────────────────────")
    songs = build_results(df, feat_cols, coords_2d)

    with open(output_json, "w") as f:
        json.dump(songs, f, indent=2)
    print(f"  Results written to {output_json} ({len(songs)} songs)")

    print("\n── Saving pipeline ───────────────────────────────────")
    save_pipeline(scaler, pca, feat_cols, output_pkl)

    print_genre_summary(songs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Supervised PCA visualization pipeline")
    parser.add_argument("--csv", default="features.csv", help="Path to features CSV")
    parser.add_argument(
        "--features",
        nargs="+",
        default=None,
        help="Features to include (supports wildcards, e.g., 'mfcc*'). Omit to use all features."
    )
    parser.add_argument("--output-json", default="results.json")
    parser.add_argument("--output-pkl", default="pipeline.pkl")
    args = parser.parse_args()

    run_pipeline(
        csv_path=args.csv,
        feature_patterns=args.features,
        output_json=args.output_json,
        output_pkl=args.output_pkl,
    )
