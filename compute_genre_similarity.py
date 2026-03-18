"""
compute_genre_similarity.py
---------------------------
Computes pairwise genre similarity using multiple methods:
  1. Wasserstein distance between feature distributions
  2. Boundary case overlap (which genres' boundary cases are closest to each other)
  3. Feature centroid correlation

Outputs a genre-genre similarity matrix to genre_similarity.json

Usage:
  python compute_genre_similarity.py --csv Data/features_30_sec.csv
"""

import json
import argparse
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler


EXCLUDE_COLS = {"filename", "length", "label"}


def load_data(csv_path: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Load CSV and return dataframe, feature names, and genre list."""
    df = pd.read_csv(csv_path)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    genres = sorted(df['label'].unique())
    
    print(f"Loaded {len(df)} songs, {len(feature_cols)} features, {len(genres)} genres")
    return df, feature_cols, genres


def compute_wasserstein_similarity(df: pd.DataFrame, feature_cols: list[str], genres: list[str]) -> np.ndarray:
    """
    Compute genre similarity using Wasserstein distance between feature distributions.
    
    For each pair of genres, compute the average Wasserstein distance across all features.
    Lower distance = higher similarity.
    
    Returns normalized similarity matrix (0 = dissimilar, 1 = identical)
    """
    print("\n── Computing Wasserstein distances ──────────────────")
    
    n = len(genres)
    distance_matrix = np.zeros((n, n))
    
    for i, g1 in enumerate(genres):
        for j, g2 in enumerate(genres):
            if i == j:
                distance_matrix[i, j] = 0.0  # Same genre
                continue
            
            g1_songs = df[df['label'] == g1]
            g2_songs = df[df['label'] == g2]
            
            # Compute Wasserstein distance for each feature
            distances = []
            for feat in feature_cols:
                v1 = g1_songs[feat].values
                v2 = g2_songs[feat].values
                
                # Normalize to [0, 1] for comparable distances
                all_vals = np.concatenate([v1, v2])
                v1_norm = (v1 - all_vals.min()) / (all_vals.max() - all_vals.min() + 1e-10)
                v2_norm = (v2 - all_vals.min()) / (all_vals.max() - all_vals.min() + 1e-10)
                
                dist = wasserstein_distance(v1_norm, v2_norm)
                distances.append(dist)
            
            # Average distance across all features
            distance_matrix[i, j] = np.mean(distances)
            
        print(f"  {g1}: computed distances to all genres")
    
    # Convert distance to similarity: similarity = 1 / (1 + distance)
    # This maps distance=0 -> similarity=1, distance=inf -> similarity=0
    similarity_matrix = 1.0 / (1.0 + distance_matrix)
    
    # Normalize so diagonal is exactly 1.0
    np.fill_diagonal(similarity_matrix, 1.0)
    
    return similarity_matrix


def compute_boundary_overlap(df: pd.DataFrame, feature_cols: list[str], genres: list[str], threshold: float = 1.5) -> np.ndarray:
    """
    Compute genre similarity based on boundary case overlap.
    
    For each genre's boundary cases (boundary_score < threshold), count which
    other genres they are closest to in feature space.
    
    Returns matrix where entry [i, j] = proportion of genre i's boundary cases
    that have genre j as their nearest cross-genre neighbor.
    """
    print("\n── Computing boundary overlap ───────────────────────")
    
    n = len(genres)
    genre_to_idx = {g: i for i, g in enumerate(genres)}
    overlap_matrix = np.zeros((n, n))
    
    # Standardize features for distance computation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols].values)
    
    # Check if boundary_score exists
    has_boundary_score = 'boundary_score' in df.columns
    
    if not has_boundary_score:
        print("  Warning: No boundary_score column found in CSV")
        print("  Using all songs instead of just boundary cases")
        boundary_songs = df.index.tolist()
    else:
        # Get boundary case indices
        boundary_songs = df[
            (df['boundary_score'].notna()) & 
            (df['boundary_score'] < threshold)
        ].index.tolist()
        print(f"  Found {len(boundary_songs)} boundary cases (score < {threshold})")
    
    if len(boundary_songs) == 0:
        print("  No boundary cases found, returning uniform matrix")
        return np.ones((n, n)) / n
    
    # For each boundary case, find nearest cross-genre neighbor
    for idx in boundary_songs:
        my_genre = df.loc[idx, 'label']
        my_features = X_scaled[idx]
        
        # Find closest song from each other genre
        min_dist_per_genre = {}
        for other_idx in df.index:
            if other_idx == idx:
                continue
            
            other_genre = df.loc[other_idx, 'label']
            if other_genre == my_genre:
                continue
            
            other_features = X_scaled[other_idx]
            dist = euclidean(my_features, other_features)
            
            if other_genre not in min_dist_per_genre or dist < min_dist_per_genre[other_genre]:
                min_dist_per_genre[other_genre] = dist
        
        # Find the genre with minimum distance
        if min_dist_per_genre:
            closest_genre = min(min_dist_per_genre, key=min_dist_per_genre.get)
            overlap_matrix[genre_to_idx[my_genre], genre_to_idx[closest_genre]] += 1
    
    # Normalize rows to get proportions
    row_sums = overlap_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    overlap_matrix = overlap_matrix / row_sums
    
    # Set diagonal to 1.0 (genre is most similar to itself)
    np.fill_diagonal(overlap_matrix, 1.0)
    
    print(f"  Computed overlap for {len(boundary_songs)} boundary cases")
    
    return overlap_matrix


def compute_centroid_correlation(df: pd.DataFrame, feature_cols: list[str], genres: list[str]) -> np.ndarray:
    """
    Compute genre similarity using correlation between mean feature vectors.
    
    For each genre, compute the centroid (mean feature vector), then compute
    Pearson correlation between all pairs of centroids.
    
    Returns correlation matrix (-1 to 1)
    """
    print("\n── Computing centroid correlations ──────────────────")
    
    n = len(genres)
    centroids = {}
    
    for genre in genres:
        genre_songs = df[df['label'] == genre]
        centroids[genre] = genre_songs[feature_cols].mean().values
        print(f"  {genre}: centroid from {len(genre_songs)} songs")
    
    # Compute pairwise correlations
    corr_matrix = np.zeros((n, n))
    for i, g1 in enumerate(genres):
        for j, g2 in enumerate(genres):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                # Pearson correlation
                c1 = centroids[g1]
                c2 = centroids[g2]
                corr = np.corrcoef(c1, c2)[0, 1]
                corr_matrix[i, j] = corr
    
    return corr_matrix


def combine_similarities(wass_sim: np.ndarray, boundary_sim: np.ndarray, centroid_corr: np.ndarray) -> np.ndarray:
    """
    Combine three similarity measures into a single consensus score.
    
    Uses weighted average:
    - Wasserstein: 50% (most comprehensive)
    - Boundary overlap: 30% (reveals confusion)
    - Centroid correlation: 20% (simple baseline)
    """
    # Normalize centroid correlation from [-1, 1] to [0, 1]
    centroid_sim = (centroid_corr + 1) / 2
    
    # Weighted combination
    combined = (
        0.5 * wass_sim + 
        0.3 * boundary_sim + 
        0.2 * centroid_sim
    )
    
    return combined


def print_top_similarities(similarity_matrix: np.ndarray, genres: list[str], top_n: int = 15):
    """Print the most similar genre pairs."""
    print(f"\n── Top {top_n} most similar genre pairs ────────────")
    
    pairs = []
    n = len(genres)
    for i in range(n):
        for j in range(i + 1, n):  # Upper triangle only
            pairs.append((genres[i], genres[j], similarity_matrix[i, j]))
    
    pairs.sort(key=lambda x: x[2], reverse=True)
    
    for i, (g1, g2, sim) in enumerate(pairs[:top_n], 1):
        print(f"  {i:2d}. {g1:12s} ↔ {g2:12s}  similarity={sim:.4f}")


def save_results(
    genres: list[str],
    wasserstein: np.ndarray,
    boundary: np.ndarray,
    centroid: np.ndarray,
    combined: np.ndarray,
    output_path: str = "genre_similarity.json"
):
    """Save all similarity matrices to JSON."""
    
    def matrix_to_dict(matrix: np.ndarray) -> dict:
        """Convert numpy matrix to nested dict for JSON."""
        return {
            g1: {g2: float(matrix[i, j]) for j, g2 in enumerate(genres)}
            for i, g1 in enumerate(genres)
        }
    
    results = {
        "genres": genres,
        "wasserstein_similarity": matrix_to_dict(wasserstein),
        "boundary_overlap": matrix_to_dict(boundary),
        "centroid_correlation": matrix_to_dict(centroid),
        "combined_similarity": matrix_to_dict(combined),
        "metadata": {
            "description": "Genre-genre similarity matrices",
            "methods": {
                "wasserstein_similarity": "Avg Wasserstein distance between feature distributions (0=dissimilar, 1=identical)",
                "boundary_overlap": "Proportion of boundary cases with nearest neighbor in other genre (0=no overlap, 1=full overlap)",
                "centroid_correlation": "Pearson correlation between mean feature vectors (-1 to 1)",
                "combined_similarity": "Weighted average: 50% Wasserstein + 30% boundary + 20% centroid"
            }
        }
    }
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compute genre-genre similarity matrix")
    parser.add_argument("--csv", default="features.csv", help="Path to features CSV")
    parser.add_argument("--output", default="genre_similarity.json", help="Output JSON path")
    parser.add_argument("--boundary-threshold", type=float, default=1.5, 
                       help="Boundary score threshold for overlap analysis (default: 1.5)")
    args = parser.parse_args()
    
    # Load data
    df, feature_cols, genres = load_data(args.csv)
    
    # Compute three types of similarity
    wasserstein_sim = compute_wasserstein_similarity(df, feature_cols, genres)
    boundary_sim = compute_boundary_overlap(df, feature_cols, genres, args.boundary_threshold)
    centroid_corr = compute_centroid_correlation(df, feature_cols, genres)
    
    # Combine into consensus score
    print("\n── Combining similarity measures ────────────────────")
    combined_sim = combine_similarities(wasserstein_sim, boundary_sim, centroid_corr)
    
    # Print insights
    print_top_similarities(combined_sim, genres, top_n=15)
    
    # Save results
    save_results(genres, wasserstein_sim, boundary_sim, centroid_corr, combined_sim, args.output)
    
    print(f"\n✓ Genre similarity analysis complete")


if __name__ == "__main__":
    main()
