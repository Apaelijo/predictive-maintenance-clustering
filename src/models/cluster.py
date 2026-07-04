import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage
import joblib
import logging

logging.basicConfig(level=logging.INFO)

def find_optimal_k(X, max_k=10):
    """Elbow Method + Silhouette for K-Means"""
    inertias = []
    sil_scores = []
    
    for k in range(2, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        inertias.append(kmeans.inertia_)
        sil_scores.append(silhouette_score(X, labels))
    
    # Plot Elbow
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(2, max_k+1), inertias, 'bo-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia (WCSS)')
    plt.title('Elbow Method')
    
    plt.subplot(1, 2, 2)
    plt.plot(range(2, max_k+1), sil_scores, 'ro-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis')
    plt.show()
    
    return inertias, sil_scores

def kmeans_clustering(X, n_clusters=4):
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    sil = silhouette_score(X, labels)
    db = davies_bouldin_score(X, labels)
    logging.info(f"K-Means (k={n_clusters}) - Silhouette: {sil:.4f}, DB: {db:.4f}")
    return model, labels, sil, db

def dbscan_clustering(X, eps=0.5, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    # Ignore noise points (-1) for metrics if many
    core_mask = labels != -1
    if len(set(labels[core_mask])) > 1:
        sil = silhouette_score(X[core_mask], labels[core_mask])
        db = davies_bouldin_score(X[core_mask], labels[core_mask])
    else:
        sil = db = np.nan
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    logging.info(f"DBSCAN (eps={eps}, min_samples={min_samples}) - Clusters: {n_clusters}, Silhouette: {sil:.4f}")
    return model, labels, sil, db

def hierarchical_clustering(X, n_clusters=4):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = model.fit_predict(X)
    sil = silhouette_score(X, labels)
    db = davies_bouldin_score(X, labels)
    logging.info(f"Hierarchical (n={n_clusters}) - Silhouette: {sil:.4f}, DB: {db:.4f}")
    return model, labels, sil, db

def plot_dendrogram(X, method='ward'):
    Z = linkage(X, method)
    plt.figure(figsize=(12, 8))
    dendrogram(Z, truncate_mode='lastp', p=12)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample index or Cluster size')
    plt.ylabel('Distance')
    plt.show()