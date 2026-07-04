from src.features.build_features import create_full_pipeline, engineer_features
from src.data.preprocess import load_and_clean_data
from src.models.cluster import (find_optimal_k, kmeans_clustering, 
                               dbscan_clustering, hierarchical_clustering)

def main():
    # Load & preprocess
    data, _ = load_and_clean_data()
    data_eng = engineer_features(data)
    pipeline = create_full_pipeline(n_components=8)  # Adjust based on variance
    X = pipeline.fit_transform(data_eng)
    
    # K-Means tuning
    print("=== K-Means Tuning ===")
    find_optimal_k(X, max_k=8)
    kmeans_model, kmeans_labels, k_sil, k_db = kmeans_clustering(X, n_clusters=4)  # Choose from elbow/sil
    
    # DBSCAN (tune eps via k-distance plot - separate cell recommended)
    print("=== DBSCAN ===")
    db_model, db_labels, db_sil, db_db = dbscan_clustering(X, eps=0.8, min_samples=10)
    
    # Hierarchical
    print("=== Hierarchical ===")
    plot_dendrogram(X[:1000])  # subsample for speed
    hier_model, hier_labels, h_sil, h_db = hierarchical_clustering(X, n_clusters=4)
    
    # Comparison table
    results = pd.DataFrame({
        'Model': ['K-Means (k=4)', 'DBSCAN', 'Hierarchical (k=4)'],
        'Silhouette': [k_sil, db_sil, h_sil],
        'Davies-Bouldin': [k_db, db_db, h_db]
    })
    print(results)
    
    # Save champion (example: K-Means - adjust based on metrics)
    joblib.dump(kmeans_model, 'models/champion_kmeans.pkl')
    joblib.dump(pipeline, 'models/preprocessing_pipeline.pkl')
    print("Champion model and pipeline saved to models/")

if __name__ == "__main__":
    main()