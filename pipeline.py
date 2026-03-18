"""
pipeline.py
-----------
Core ML pipeline for unsupervised genre clustering.

Takes the features CSV, normalizes, clusters with K-Means,
and reduces to 2D with both UMAP and PCA for comparison.

Outputs:
  - results.json  → full per-song data for the frontend
  - pipeline.pkl  → saved pipeline state (scaler + cluster model + reducer)

Usage:
  python pipeline.py --csv features.csv --clusters 10 --reducer umap
"""

import json
import pickle
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# UMAP is strongly preferred over PCA for audio features — clusters are
# more separated and meaningful. Install with: pip install umap-learn
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: umap-learn not installed. Falling back to PCA.")
    print("Install with: pip install umap-learn")


# ─── Feature columns ──────────────────────────────────────────────────────────
# Everything except filename, length, and label.
# Adjust this list if your CSV differs.
EXCLUDE_COLS = {"filename", "length", "label"}


def load_data(csv_path: str) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """
    Load CSV and split into metadata and feature matrix.

    Returns:
        df       - full dataframe (metadata + features)
        X        - raw feature matrix as numpy array
        feat_cols - list of feature column names used
    """
    df = pd.read_csv(csv_path)

    # Drop any rows with NaN or inf values (corrupted extractions)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    n_before = len(df)
    df.dropna(inplace=True)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"  Dropped {n_dropped} rows with NaN/inf values")

    feat_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X = df[feat_cols].values.astype(np.float32)

    print(f"  Loaded {len(df)} songs with {len(feat_cols)} features")
    return df, X, feat_cols


def normalize(X: np.ndarray) -> tuple[np.ndarray, StandardScaler]:
    """
    StandardScaler normalization: zero mean, unit variance per feature.

    This is critical — MFCCs and spectral centroid live on wildly different
    scales, so unnormalized clustering is dominated by high-variance features.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def find_optimal_k(X_scaled: np.ndarray, k_range: range) -> dict:
    """
    Run K-Means for a range of k values and compute:
      - inertia (elbow method)
      - silhouette score (higher = better separated clusters)

    Use this to pick k before committing to a final model.
    Prints a table; look for the 'elbow' in inertia and the peak silhouette.
    """
    print(f"\n  Scanning k = {k_range.start} to {k_range.stop - 1}...")
    results = {}

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertia = km.inertia_
        sil = silhouette_score(X_scaled, labels, sample_size=500, random_state=42)
        results[k] = {"inertia": inertia, "silhouette": sil}
        print(f"    k={k:2d}  inertia={inertia:10.1f}  silhouette={sil:.4f}")

    best_k = max(results, key=lambda k: results[k]["silhouette"])
    print(f"\n  Best k by silhouette: {best_k}")
    return results


def cluster(X_scaled: np.ndarray, n_clusters: int) -> tuple[np.ndarray, KMeans]:
    """Fit final K-Means model and return cluster labels + fitted model."""
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    print(f"  Silhouette score: {sil:.4f}  (range -1 to 1, higher is better)")
    return labels, km


def reduce_2d(
    X_scaled: np.ndarray,
    method: str = "umap",
) -> tuple[np.ndarray, object]:
    """
    Reduce to 2D for visualization.

    UMAP is strongly recommended — it preserves both local and global structure
    much better than PCA for high-dimensional audio features. The parameters
    below are tuned for ~1000 songs; adjust n_neighbors for larger datasets.

    PCA is provided as a fast fallback and for comparison.
    """
    if method == "umap" and UMAP_AVAILABLE:
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=15,      # local neighborhood size; try 10-30
            min_dist=0.1,        # how tightly points cluster visually; try 0.05-0.5
            metric="euclidean",
            random_state=42,
        )
        print(f"  Running UMAP (n_neighbors=15, min_dist=0.1)...")
    else:
        if method == "umap":
            print("  UMAP not available, falling back to PCA")
        reducer = PCA(n_components=2, random_state=42)
        print(f"  Running PCA...")

    coords_2d = reducer.fit_transform(X_scaled)

    if isinstance(reducer, PCA):
        explained = reducer.explained_variance_ratio_.sum() * 100
        print(f"  PCA explained variance: {explained:.1f}%")

    return coords_2d, reducer


def build_results(
    df: pd.DataFrame,
    feat_cols: list[str],
    labels: np.ndarray,
    coords_2d: np.ndarray,
) -> list[dict]:
    """
    Assemble the per-song result objects that the frontend will consume.

    Each song becomes a JSON object with:
      - id, filename, true_label (ground truth, hidden from clustering)
      - cluster (what the algorithm assigned)
      - x, y (2D coordinates for the scatter plot)
      - features (full feature dict for the cross-section views)
    """
    songs = []
    for i, row in df.reset_index(drop=True).iterrows():
        song = {
            "id": i,
            "filename": row["filename"],
            "true_label": row.get("label", "unknown"),  # kept for eval, not shown
            "cluster": int(labels[i]),
            "x": float(coords_2d[i, 0]),
            "y": float(coords_2d[i, 1]),
            # Key features surfaced for the cross-section visualization
            "features": {col: float(row[col]) for col in feat_cols},
        }
        songs.append(song)
    return songs


def save_pipeline(
    scaler: StandardScaler,
    cluster_model: KMeans,
    reducer: object,
    feat_cols: list[str],
    output_path: str = "pipeline.pkl",
) -> None:
    """
    Pickle the fitted pipeline so new uploaded songs can be processed
    without re-fitting. The FastAPI backend will load this on startup.

    The pipeline contains:
      scaler       - to normalize new song features identically
      cluster_model - to assign new songs to nearest centroid
      reducer      - to project new songs into the same 2D space
      feat_cols    - ordered list of features (must match extractor output)
    """
    pipeline_state = {
        "scaler": scaler,
        "cluster_model": cluster_model,
        "reducer": reducer,
        "feat_cols": feat_cols,
    }
    with open(output_path, "wb") as f:
        pickle.dump(pipeline_state, f)
    print(f"  Pipeline saved to {output_path}")


def run_pipeline(
    csv_path: str,
    n_clusters: int | None,
    reducer_method: str,
    scan_k: bool,
    output_json: str,
    output_pkl: str,
) -> None:
    print("\n── Loading data ──────────────────────────────────────")
    df, X, feat_cols = load_data(csv_path)

    print("\n── Normalizing features ──────────────────────────────")
    X_scaled, scaler = normalize(X)

    if scan_k:
        print("\n── Scanning for optimal k ────────────────────────────")
        find_optimal_k(X_scaled, k_range=range(5, 16))
        print("\nRe-run with --clusters <k> to fit the final model.")
        return

    k = n_clusters or 10  # default to 10 (one per GTZAN genre)
    print(f"\n── Clustering (k={k}) ────────────────────────────────")
    labels, cluster_model = cluster(X_scaled, k)

    print(f"\n── Reducing to 2D ({reducer_method}) ─────────────────")
    coords_2d, reducer = reduce_2d(X_scaled, method=reducer_method)

    print(f"\n── Assembling results ────────────────────────────────")
    songs = build_results(df, feat_cols, labels, coords_2d)

    with open(output_json, "w") as f:
        json.dump(songs, f, indent=2)
    print(f"  Results written to {output_json} ({len(songs)} songs)")

    print(f"\n── Saving pipeline state ─────────────────────────────")
    save_pipeline(scaler, cluster_model, reducer, feat_cols, output_pkl)

    # Quick cluster summary
    print(f"\n── Cluster summary ───────────────────────────────────")
    label_series = pd.Series(labels)
    for cluster_id, count in label_series.value_counts().sort_index().items():
        # If we have ground truth, show dominant genre per cluster
        cluster_songs = df[labels == cluster_id]
        if "label" in df.columns:
            dominant = cluster_songs["label"].mode()[0]
            print(f"  Cluster {cluster_id:2d}: {count:4d} songs  (dominant genre: {dominant})")
        else:
            print(f"  Cluster {cluster_id:2d}: {count:4d} songs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genre clustering pipeline")
    parser.add_argument("--csv", default="features.csv", help="Path to features CSV")
    parser.add_argument("--clusters", type=int, default=None,
                        help="Number of clusters (omit to scan for best k)")
    parser.add_argument("--reducer", choices=["umap", "pca"], default="umap",
                        help="Dimensionality reduction method")
    parser.add_argument("--scan-k", action="store_true",
                        help="Scan k=5..15 and print silhouette scores, then exit")
    parser.add_argument("--output-json", default="results.json")
    parser.add_argument("--output-pkl", default="pipeline.pkl")
    args = parser.parse_args()

    run_pipeline(
        csv_path=args.csv,
        n_clusters=args.clusters,
        reducer_method=args.reducer,
        scan_k=args.scan_k,
        output_json=args.output_json,
        output_pkl=args.output_pkl,
    )
