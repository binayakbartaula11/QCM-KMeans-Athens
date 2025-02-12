# K-means Clustering of Quantum Circuits (IBMQ Athens)

## Project Overview

Quantum computing holds immense promise but is hampered by performance variability due to noise, errors, and the fragile nature of quantum states. This project leverages **K-means clustering** to analyze and classify quantum circuits from the **IBMQ Athens** dataset based on key performance metrics. By grouping circuits with similar characteristics, we can better understand and potentially mitigate the challenges associated with quantum circuit execution.

The IBMQ Athens dataset offers rich information about circuit behavior on IBM's Athens processor. Our objective is to apply unsupervised learning techniques to uncover hidden patterns in the data, ultimately guiding improvements in quantum circuit design and performance.

### Key Performance Metrics

| **Feature**                     | **Description**                                                                                                                                      |
|---------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Quantum Gate Connectivity**   | The arrangement and interaction of quantum gates, where higher connectivity can increase complexity—and noise—in the circuit.                         |
| **Error Rates**                 | The frequency of errors occurring during gate operations, qubit initialization, and measurement.                                                    |
| **Circuit Depth**               | The number of sequential layers of gates; deeper circuits may face more noise while shallower ones might lack computational complexity.             |
| **Coherence Times**             | The duration a qubit can maintain its quantum state before decoherence, directly impacting circuit reliability and performance.                       |

________________________________________

## Clustering Methodology

This project uses the K-means algorithm to partition the quantum circuits into distinct clusters. The clustering process helps to:
- **Identify Reliable Groups:** Spot clusters of circuits that perform well.
- **Detect Outliers:** Highlight circuits with suboptimal performance.
- **Inform Design Choices:** Provide insights that can lead to more robust quantum circuit designs.

### Steps in the Clustering Process

1. **Preprocessing:** Scale and normalize the selected performance features.
2. **Clustering:** Apply the K-means algorithm to segment the dataset into clusters.
3. **Evaluation:** Use the Elbow Method and Silhouette Score to determine the optimal number of clusters.

#### Model Evaluation Metrics

| **Metric**             | **Purpose**                                                                                 |
|------------------------|---------------------------------------------------------------------------------------------|
| **Silhouette Score**   | Measures the compactness and separation of the clusters.                                   |
| **Elbow Method**       | Helps pinpoint the ideal cluster count by analyzing the reduction in inertia.               |

________________________________________

## How to Run the Project

Follow these steps to replicate the clustering analysis:

### Step 1: Import the Required Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
```
### Step 2: Load the Dataset
```python
df = pd.read_csv('/content/IBMQAthens.csv')
```
### Step 3: Inspect the Data
```python
# Preview the data and check for missing values
print(df.head())
print(df.info())
print(df.describe())
```
### Step 4: Clean the Data
```python
# Remove or fill missing data
df = df.dropna()  # Alternatively, use df.fillna() with mean/median imputation
```
### Step 5: Select the Features
```python
# Select features corresponding to the connectivity metrics
features = df[['cx_0_1', 'cx_0_2', 'cx_0_3', 'cx_0_4', 'cx_1_0', 'cx_1_2', 
               'cx_1_3', 'cx_1_4', 'cx_2_0', 'cx_2_1', 'cx_2_3', 'cx_2_4', 
               'cx_3_0', 'cx_3_1', 'cx_3_2', 'cx_3_4', 'cx_4_0', 'cx_4_1', 
               'cx_4_2', 'cx_4_3']]
```
### Step 6: Normalize the Features
```python
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
```
#### Step 7: Determine the Optimal Number of Clusters
```python
inertia = []
sil_scores = []

for k in range(1, 11):  # Evaluating cluster counts from 1 to 10
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(features_scaled)
    inertia.append(kmeans.inertia_)
    if k > 1:
        score = silhouette_score(features_scaled, kmeans.labels_)
        sil_scores.append(score)
```
### Step 8: Visualize the Evaluation Metrics
```python
# Elbow Method Plot
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o', linestyle='-', color='blue')
plt.title('Elbow Method for Optimal Cluster Count')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

# Silhouette Score Plot
plt.figure(figsize=(8, 6))
plt.plot(range(2, 11), sil_scores, marker='o', linestyle='-', color='green')
plt.title('Silhouette Score vs. Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()
```
### Step 9: Apply K-means Clustering
```python
# Choose an optimal number of clusters based on the plots (here, using 3 as an example)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(features_scaled)
df['cluster'] = clusters
```
### Step 10: Visualize the Clusters with PCA
```python
pca = PCA(n_components=2)
pca_features = pca.fit_transform(features_scaled)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(pca_features[:, 0], pca_features[:, 1], c=df['cluster'], cmap='viridis', alpha=0.7)
plt.title("Visualization of K-means Clusters on IBMQ Athens Data")
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(scatter, label='Cluster Label')
plt.grid(True)
plt.show()
```
### Step 11: Analyze Cluster Distribution and Characteristics
```python
# Print cluster distribution
print(df['cluster'].value_counts())

# Examine the centroids
centroids = kmeans.cluster_centers_
print(centroids)

# Evaluate clustering quality
sil_score = silhouette_score(features_scaled, clusters)
print(f"Silhouette Score: {sil_score:.4f}")

# Get summary statistics for each cluster
cluster_summary = {}
for cluster_num in range(3):  # Update range if using a different number of clusters
    cluster_summary[cluster_num] = df[df['cluster'] == cluster_num].describe()
    print(f"Cluster {cluster_num} Summary:")
    print(cluster_summary[cluster_num])
```
### Step 12: Optional Fine-Tuning and Validation
```python
# Fine-tune by increasing the number of initialization runs
kmeans = KMeans(n_clusters=3, n_init=20, random_state=42)
clusters = kmeans.fit_predict(features_scaled)
df['cluster'] = clusters

# Validate consistency between different K-means runs using Adjusted Rand Index (ARI)
from sklearn.metrics import adjusted_rand_score

kmeans_run_1 = KMeans(n_clusters=3, random_state=42)
clusters_run_1 = kmeans_run_1.fit_predict(features_scaled)

kmeans_run_2 = KMeans(n_clusters=3, random_state=43)
clusters_run_2 = kmeans_run_2.fit_predict(features_scaled)

ari = adjusted_rand_score(clusters_run_1, clusters_run_2)
print(f"Adjusted Rand Index (ARI) between two K-means runs: {ari:.4f}")
```
________________________________________
## Conclusion
By applying K-means clustering to the IBMQ Athens quantum circuits dataset, this project uncovers meaningful patterns and clusters
based on performance indicators. This analysis not only highlights the challenges inherent in quantum circuit execution—such as noise
and error propagation—but also points toward potential pathways for optimizing quantum circuit design.
________________________________________
## Future Directions
- Enhance Feature Engineering: Integrate additional circuit parameters to develop a more detailed performance profile.
- Explore Alternative Algorithms: Investigate other clustering methods like DBSCAN or hierarchical clustering to compare results.
- Extend the Dataset: Validate findings using other quantum circuit datasets to ensure robust, generalizable insights.
- Real-Time Analysis: Develop real-time monitoring tools to dynamically cluster and evaluate circuits as new data becomes available.
________________________________________
## Source Code
For the complete source code, please visit the GitHub repository:
[QCM-KMeans-Athens](https://github.com/binayakbartaula11/QCM-KMeans-Athens/blob/main/IBMQ_Athens_QuantumKmeans_Clustering.ipynb)
________________________________________
## Dataset
The IBMQ Athens dataset used in this project is available here:
[IBMQ Athens Dataset](https://data.mendeley.com/datasets/pmycgb2bt7/1)
