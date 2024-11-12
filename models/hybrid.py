# Data handling and numerical operations
import numpy as np
import pandas as pd

# Scikit-learn models and utilities
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.neural_network import MLPRegressor

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

class BestOfAllAnomalyDetectionEnsemble:

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.autoencoder = None
        self.pca = None
        self.kmeans = None
        self.dbscan = None

    def preprocess_data(self, data):
        # Scale data to standardize
        return self.scaler.fit_transform(data)

    def find_optimal_k(self, data, max_k=10):
        # Optimal k based on silhouette score for KMeans
        silhouette_scores = []
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state)
            labels = kmeans.fit_predict(data)
            silhouette_scores.append(silhouette_score(data, labels))
        optimal_k = np.argmax(silhouette_scores) + 2
        return optimal_k

    def find_optimal_eps(self, data, n_neighbors=5):
        # Calculate optimal epsilon for DBSCAN using nearest neighbor distances
        nearest_neighbors = NearestNeighbors(n_neighbors=n_neighbors)
        distances, _ = nearest_neighbors.fit(data).kneighbors(data)
        distances = np.sort(distances[:, n_neighbors - 1])
        return np.median(distances)

    def train_autoencoder(self, data, epochs=100, hidden_layer_size=32):
        input_dim = data.shape[1]
        self.autoencoder = MLPRegressor(
            hidden_layer_sizes=(hidden_layer_size, int(hidden_layer_size / 2), hidden_layer_size),
            activation='relu', solver='adam', max_iter=epochs, random_state=self.random_state
        )
        self.autoencoder.fit(data, data)
        # Calculate reconstruction error
        reconstruction_error = np.mean(np.power(data - self.autoencoder.predict(data), 2), axis=1)
        return reconstruction_error

    def perform_pca(self, data, n_components=2):
        # PCA for dimensionality reduction and visualization
        self.pca = PCA(n_components=n_components, random_state=self.random_state)
        return self.pca.fit_transform(data)

    def build_ensemble(self, data, optimal_k, eps):
        # Initialize and fit KMeans
        self.kmeans = KMeans(n_clusters=optimal_k, random_state=self.random_state)
        kmeans_labels = self.kmeans.fit_predict(data)

        # Initialize and fit DBSCAN
        self.dbscan = DBSCAN(eps=eps, min_samples=5)
        dbscan_labels = self.dbscan.fit_predict(data)

        # Nearest Neighbors-based anomaly detection (percentile threshold)
        nearest_neighbors = NearestNeighbors(n_neighbors=5)
        distances, _ = nearest_neighbors.fit(data).kneighbors(data)
        nn_threshold = np.percentile(distances[:, -1], 95)
        nn_anomalies = (distances[:, -1] > nn_threshold).astype(int)

        return kmeans_labels, dbscan_labels, nn_anomalies

    def combine_anomaly_scores(self, kmeans_labels, dbscan_labels, nn_anomalies, autoencoder_scores):
        # Convert DBSCAN and KMeans labels to anomaly scores
        dbscan_anomalies = (dbscan_labels == -1).astype(int)
        kmeans_anomalies = (kmeans_labels != 0).astype(int)

        # Scale autoencoder scores to [0, 1] range for combination
        autoencoder_scaled = (autoencoder_scores - autoencoder_scores.min()) / (autoencoder_scores.max() - autoencoder_scores.min())

        # Ensemble voting: weighted sum of anomaly indicators
        combined_scores = (kmeans_anomalies + dbscan_anomalies + nn_anomalies + autoencoder_scaled) / 4
        combined_anomalies = np.where(combined_scores > 0.5, 1, 0)  # Adaptive threshold

        return combined_anomalies, combined_scores

    def detect_anomalies(self, data):
        # Full pipeline execution
        scaled_data = self.preprocess_data(data)

        # Adaptive KMeans and DBSCAN parameters
        optimal_k = self.find_optimal_k(scaled_data)
        eps = self.find_optimal_eps(scaled_data)

        # Build ensemble with KMeans, DBSCAN, and Nearest Neighbors
        kmeans_labels, dbscan_labels, nn_anomalies = self.build_ensemble(scaled_data, optimal_k, eps)

        # Autoencoder-based anomaly detection
        autoencoder_scores = self.train_autoencoder(scaled_data)

        # Combine anomaly scores
        combined_anomalies, combined_scores = self.combine_anomaly_scores(kmeans_labels, dbscan_labels, nn_anomalies, autoencoder_scores)

        return combined_anomalies, combined_scores

    def visualize_anomalies(self, data, combined_anomalies):
        # PCA for 2D visualization of anomalies
        pca_data = self.perform_pca(data)
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=combined_anomalies, palette={0: 'blue', 1: 'red'})
        plt.title("Anomaly Visualization using PCA")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.show()

