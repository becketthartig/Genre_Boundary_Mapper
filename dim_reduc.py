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

# 1. Load the GTZAN feature file
df = pd.read_csv('Data/features_30_sec.csv')

# 2. Prepare the data (removing non-musical metadata)
X = df.drop(columns=['filename', 'length', 'label'])

# 3. Standardize the features
# This ensures that 'Tempo' and 'Spectral Centroid' have equal weight
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Apply K-Means Clustering (10 classes)
# We set n_clusters=10 to see how well the math aligns with the 10 genres
kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_transform(X_scaled).argmin(axis=1) # Get cluster labels

# 5. Dimensionality Reduction to 2D for plotting
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(data=pca_result, columns=['Component 1', 'Component 2'])
pca_df['Cluster'] = df['cluster'].astype(str) # Convert to string for discrete coloring
pca_df['Actual_Genre'] = df['label'] # Keep the real label for comparison

# 6. Visualize the Unsupervised Clusters
plt.figure(figsize=(12, 8))
sns.scatterplot(
    x='Component 1', 
    y='Component 2', 
    hue='Cluster', 
    data=pca_df, 
    palette='tab10', 
    alpha=0.8, 
    s=70,
    style='Cluster'
)

plt.title('Unsupervised Music Clustering (K-Means on PCA)', fontsize=16)
plt.xlabel('Component 1 (Timbre & Energy)', fontsize=12)
plt.ylabel('Component 2 (Brightness & Harmony)', fontsize=12)
plt.legend(title='Machine-Defined Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()

plt.savefig('unsupervised_music_clusters.png')
plt.show()

# 7. Analysis: See which genres ended up in which cluster
cluster_composition = pd.crosstab(df['cluster'], df['label'])
print("Cluster Composition (Rows are Clusters, Columns are Genres):")
print(cluster_composition)