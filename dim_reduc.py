# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler

# # 1. Load the dataset
# # Assuming your file is named 'features_30_sec.csv'
# df = pd.read_csv('Data/features_30_sec.csv')

# # 2. Separate features from metadata/labels
# # We drop 'filename' and 'length' (not musical features) 
# # and 'label' (we want to use it for coloring later)
# X = df.drop(columns=['filename', 'length', 'label'])
# y = df['label']

# # 3. Scale the features (Critical step!)
# # This puts all features (Tempo, Brightness, Timbre) on a level playing field
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # 4. Perform PCA to reduce to 2 Dimensions
# pca = PCA(n_components=2)
# pca_result = pca.fit_transform(X_scaled)

# # Create a new dataframe for plotting
# pca_df = pd.DataFrame(data=pca_result, columns=['Component 1', 'Component 2'])
# pca_df['Genre'] = y

# # 5. Plot the results
# plt.figure(figsize=(12, 8))
# sns.scatterplot(
#     x='Component 1', 
#     y='Component 2', 
#     hue='Genre', 
#     data=pca_df, 
#     palette='viridis', 
#     alpha=0.7, 
#     s=60
# )

# plt.title('2D Projection of Music Genres (PCA)', fontsize=16)
# plt.xlabel('Component 1 (Capture of Timbre & Energy)', fontsize=12)
# plt.ylabel('Component 2 (Capture of Brightness & Harmony)', fontsize=12)
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.tight_layout()

# # Save and show
# plt.savefig('genre_clusters_2d.png')
# plt.show()

# # Optional: See which features impact the clusters the most
# loadings = pd.DataFrame(
#     pca.components_.T, 
#     columns=['PC1', 'PC2'], 
#     index=X.columns
# )
# print("Top features impacting Component 1 (Horizontal axis):")
# print(loadings['PC1'].abs().sort_values(ascending=False).head(5))




import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Load the dataset
df = pd.read_csv('Data/features_30_sec.csv')

# 2. Define the feature subset
# We keep the core spectral/rhythmic features and ONLY the first 13 MFCCs
core_features = [
    'chroma_stft_mean', 'chroma_stft_var', 'rms_mean', 'rms_var',
    'spectral_centroid_mean', 'spectral_centroid_var', 
    'spectral_bandwidth_mean', 'spectral_bandwidth_var',
    'rolloff_mean', 'rolloff_var', 'zero_crossing_rate_mean', 'zero_crossing_rate_var',
    'harmony_mean', 'harmony_var', 'perceptr_mean', 'perceptr_var', 'tempo'
]

# Add only MFCC 1 through 13 (mean and var)
mfcc_cols = []
for i in range(1, 14):
    mfcc_cols.append(f'mfcc{i}_mean')
    mfcc_cols.append(f'mfcc{i}_var')

final_feature_list = core_features + mfcc_cols
X = df[final_feature_list]

# 3. Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. K-Means Clustering (10 classes)
kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

# 5. PCA to 2 Dimensions
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
pca_df['Cluster'] = df['cluster'].astype(str)
pca_df['Genre'] = df['label']

# 6. Plotting
plt.figure(figsize=(12, 8))
sns.scatterplot(
    x='PC1', y='PC2', hue='Cluster', style='Cluster',
    data=pca_df, palette='tab10', s=60, alpha=0.7
)

plt.title('2D PCA Clustering (First 13 MFCCs Only)', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# 7. Cross-tabulation to see improvements
print("Cluster vs Genre Accuracy:")
print(pd.crosstab(df['cluster'], df['label']))




# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler

# # 1. Load the dataset
# df = pd.read_csv('Data/features_30_sec.csv')

# # 2. Extract Features & Cluster (consistent with your previous logic)
# core_features = ['chroma_stft_mean', 'rms_mean', 'spectral_centroid_mean', 'tempo']
# mfcc_cols = [f'mfcc{i}_mean' for i in range(1, 14)]
# X = df[core_features + mfcc_cols]

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
# df['cluster'] = kmeans.fit_predict(X_scaled)

# # 3. Calculate Rolling Local Density
# # Define the window size (total 100Hz = +/- 50Hz)
# window_hz = 50 

# def get_local_density(brightness, all_brightness, window):
#     # Count how many points are within the window
#     return np.sum(np.abs(all_brightness - brightness) <= window)

# # Apply density calculation to every point
# df['density_count'] = df['spectral_centroid_mean'].apply(
#     lambda x: get_local_density(x, df['spectral_centroid_mean'], window_hz)
# )

# # 4. Generate Rolling Jitter
# # The 'height' of the jitter is proportional to the local density count
# # We use a random uniform distribution between (-density, +density)
# df['rolling_jitter'] = df.apply(
#     lambda row: np.random.uniform(-row['density_count'], row['density_count']), 
#     axis=1
# )

# # 5. Plot
# plt.figure(figsize=(15, 7))

# sns.scatterplot(
#     data=df,
#     x='spectral_centroid_mean',
#     y='rolling_jitter',
#     hue='cluster',
#     palette='tab10',
#     alpha=0.6,
#     s=50,
#     edgecolor='none'
# )

# # 6. Formatting
# plt.title('Brightness Inspection: Rolling Window Density Scatter', fontsize=16)
# plt.xlabel('Brightness (Spectral Centroid in Hz)', fontsize=12)
# plt.ylabel('Relative Local Density (Point-wise Jitter)', fontsize=12)

# # Optional: Horizontal line at 0 for visual reference
# plt.axhline(0, color='black', linewidth=0.8, alpha=0.3)

# plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.grid(axis='x', linestyle='--', alpha=0.3)
# plt.tight_layout()

# plt.show()