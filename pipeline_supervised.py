"""
pipeline_supervised.py
----------------------
Supervised genre visualization pipeline using LDA.

Uses Linear Discriminant Analysis instead of PCA to maximize genre separation.
Computes boundary scores based on nearest cross-genre neighbor distance.

Outputs:
  - results.json  → per-song data with x, y coords, genre labels, and boundary_score
  - pipeline.pkl  → saved LDA + scaler + feature list for new uploads

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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NearestNeighbors
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


def run_lda(X: np.ndarray, y: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, LinearDiscriminantAnalysis, StandardScaler]:
    """
    Normalize and run LDA (Linear Discriminant Analysis).
    
    LDA finds the projection that maximizes separation between genres.
    Unlike PCA, it uses the labels to find discriminative directions.
    
    Returns:
        coords_2d - (n_samples, 2) array of 2D coordinates
        lda       - fitted LDA model
        scaler    - fitted StandardScaler
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # LDA can create at most (n_classes - 1) components
    n_classes = len(np.unique(y))
    max_components = min(n_components, n_classes - 1)
    
    if max_components < n_components:
        print(f"  Note: LDA can only create {max_components} components with {n_classes} genres")
    
    lda = LinearDiscriminantAnalysis(n_components=max_components)
    coords_2d = lda.fit_transform(X_scaled, y)
    
    explained = lda.explained_variance_ratio_.sum() * 100
    print(f"  LDA explained variance: {explained:.1f}%")
    if max_components >= 2:
        print(f"  LD1: {lda.explained_variance_ratio_[0]*100:.1f}%")
        print(f"  LD2: {lda.explained_variance_ratio_[1]*100:.1f}%")
    
    # If we only got 1 component (2 genres), pad with zeros for 2D viz
    if coords_2d.shape[1] == 1:
        coords_2d = np.hstack([coords_2d, np.zeros((coords_2d.shape[0], 1))])
    
    return coords_2d, lda, scaler


def compute_boundary_scores(X_scaled: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Compute boundary score for each song based on nearest cross-genre neighbor.
    
    Boundary score = normalized distance to nearest song of a different genre.
    Lower score = closer to another genre = boundary case.
    
    Algorithm:
    1. For each song, find k=20 nearest neighbors in full 57-D space
    2. Find the nearest neighbor with a different label
    3. Normalize by the median within-genre distance
    
    Returns:
        boundary_scores - array of shape (n_samples,)
                         Range: ~0 (strong boundary case) to ~5+ (clear example)
    """
    print(f"\n  Computing boundary scores...")
    
    # Fit KNN on the full scaled feature space
    k = 21  # k+1 because the first neighbor is always itself
    nn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    nn.fit(X_scaled)
    distances, indices = nn.kneighbors(X_scaled)
    
    boundary_scores = np.zeros(len(labels))
    within_genre_dists = []
    
    for i in range(len(labels)):
        my_label = labels[i]
        neighbor_labels = labels[indices[i]]
        neighbor_dists = distances[i]
        
        # Find nearest cross-genre neighbor
        cross_genre_mask = neighbor_labels != my_label
        if cross_genre_mask.any():
            cross_genre_dist = neighbor_dists[cross_genre_mask][0]
        else:
            # Extremely rare case: all k neighbors are same genre
            cross_genre_dist = neighbor_dists[-1]
        
        boundary_scores[i] = cross_genre_dist
        
        # Also track within-genre distances for normalization
        within_genre_mask = (neighbor_labels == my_label) & (np.arange(len(neighbor_labels)) > 0)
        if within_genre_mask.any():
            within_genre_dists.append(neighbor_dists[within_genre_mask][0])
    
    # Normalize by median within-genre distance
    median_within = np.median(within_genre_dists)
    boundary_scores_normalized = boundary_scores / median_within
    
    print(f"  Boundary score stats:")
    print(f"    Min (strongest boundary):  {boundary_scores_normalized.min():.2f}")
    print(f"    Median:                     {np.median(boundary_scores_normalized):.2f}")
    print(f"    Max (clearest example):     {boundary_scores_normalized.max():.2f}")
    
    return boundary_scores_normalized


def build_results(
    df: pd.DataFrame,
    feat_cols: list[str],
    coords_2d: np.ndarray,
    boundary_scores: np.ndarray,
) -> list[dict]:
    """
    Build the JSON output for the frontend.
    
    Each song gets:
      - id, filename, genre (the true label)
      - x, y (LDA coordinates)
      - boundary_score (lower = boundary case)
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
            "boundary_score": float(boundary_scores[i]),
            "features": {col: float(row[col]) for col in feat_cols},
        }
        songs.append(song)
    return songs


def save_pipeline(
    scaler: StandardScaler,
    lda: LinearDiscriminantAnalysis,
    feat_cols: list[str],
    output_path: str = "pipeline.pkl",
) -> None:
    """Save the fitted pipeline for processing new uploads."""
    pipeline_state = {
        "scaler": scaler,
        "lda": lda,
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


def print_boundary_cases(songs: list[dict], top_n: int = 10):
    """Print the strongest boundary cases."""
    sorted_songs = sorted(songs, key=lambda s: s["boundary_score"])
    
    print(f"\n── Top {top_n} boundary cases ────────────────────────")
    print(f"  (closest to other genres)")
    for i, s in enumerate(sorted_songs[:top_n], 1):
        print(f"  {i:2d}. {s['filename']:30s} {s['genre']:10s} score={s['boundary_score']:.2f}")


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
    y = df["label"].values

    print("\n── Running LDA ───────────────────────────────────────")
    coords_2d, lda, scaler = run_lda(X, y)

    print("\n── Computing boundary scores ─────────────────────────")
    X_scaled = scaler.transform(X)
    boundary_scores = compute_boundary_scores(X_scaled, y)

    print("\n── Building results ──────────────────────────────────")
    songs = build_results(df, feat_cols, coords_2d, boundary_scores)

    with open(output_json, "w") as f:
        json.dump(songs, f, indent=2)
    print(f"  Results written to {output_json} ({len(songs)} songs)")

    print("\n── Saving pipeline ───────────────────────────────────")
    save_pipeline(scaler, lda, feat_cols, output_pkl)

    print_genre_summary(songs)
    print_boundary_cases(songs, top_n=15)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Supervised LDA visualization pipeline")
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
