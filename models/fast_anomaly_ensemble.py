
# Standard Library Imports
import calendar
import warnings
import hashlib
import os

# Third-Party Imports
import numpy as np
import pandas as pd
import statsmodels.api as sm
import shap
from scipy.stats import norm, gamma
from joblib import Parallel, delayed

# Plotting
import plotly.graph_objs as go
import plotly.io as pio
pio.renderers.default = 'iframe'

# Intel Optimization for Scikit-Learn
from sklearnex import patch_sklearn
patch_sklearn()  # Apply Intel optimizations

# Scikit-Learn Imports
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold
)
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import (
    IsolationForest, RandomForestClassifier, RandomForestRegressor
)
from sklearn.metrics import (
    classification_report, roc_auc_score, make_scorer,
    confusion_matrix, accuracy_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
    silhouette_score
)
from sklearn.feature_selection import (
    SelectKBest, chi2, f_classif, mutual_info_classif, RFECV
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics.pairwise import cosine_distances, pairwise_distances
class FastAnomalyEnsemble:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.pca_data = None
        self.tsne_data = None

    def find_optimal_k(self, scaled_ml_input_data, max_k=7):
        inertia = []
        silhouette_scores = []

        def compute_kmeans(k):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10, n_jobs=-1)
            kmeans.fit(scaled_ml_input_data)
            return kmeans.inertia_, silhouette_score(scaled_ml_input_data, kmeans.labels_) if len(set(kmeans.labels_)) > 1 else -1

        results = Parallel(n_jobs=-1)(delayed(compute_kmeans)(k) for k in range(2, max_k + 1))
        inertia, silhouette_scores = zip(*results)
        
        elbow_trace = go.Scatter(x=list(range(2, max_k + 1)), y=inertia, mode='lines+markers', name='Inertia')
        silhouette_trace = go.Scatter(x=list(range(2, max_k + 1)), y=silhouette_scores, mode='lines+markers', name='Silhouette Score')
        
        optimal_k = np.argmax(silhouette_scores) + 2
        return optimal_k, elbow_trace, silhouette_trace

    def find_optimal_eps(self, scaled_ml_input_data, n_neighbors=5):
        nearest_neighbors = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
        neighbors_fit = nearest_neighbors.fit(scaled_ml_input_data)
        distances, indices = neighbors_fit.kneighbors(scaled_ml_input_data)
        distances = np.sort(distances[:, n_neighbors - 1], axis=0)
        
        eps_trace = go.Scatter(x=list(range(len(distances))), y=distances, mode='lines', name='k-NN Distance')
        eps_value = np.median(distances)
        
        return eps_value, eps_trace

    def build_ensemble(self, scaled_ml_input_data, optimal_k, n_neighbors=5, nn_percentile=95):
        kmeans = KMeans(n_clusters=optimal_k, random_state=self.random_state, n_init=10, n_jobs=-1)
        eps_value, _ = self.find_optimal_eps(scaled_ml_input_data, n_neighbors=n_neighbors)
        dbscan = DBSCAN(eps=eps_value, metric='euclidean', n_jobs=-1)
        nearest_neighbors = NearestNeighbors(n_neighbors=n_neighbors, algorithm='kd_tree', metric='euclidean', n_jobs=-1)

        try:
            kmeans_labels = kmeans.fit_predict(scaled_ml_input_data)
            dbscan_labels = dbscan.fit_predict(scaled_ml_input_data)
            nearest_neighbors.fit(scaled_ml_input_data)
            distances, indices = nearest_neighbors.kneighbors(scaled_ml_input_data)
            nn_anomaly = np.where(distances[:, -1] > np.percentile(distances[:, -1], nn_percentile), 1, 0)
        except Exception as e:
            print(f"Error in model fitting: {e}")
            return None, None, None

        return kmeans_labels, dbscan_labels, nn_anomaly

    def combine_results(self, kmeans_labels, dbscan_labels, nn_anomaly):
        dbscan_anomaly = np.where(dbscan_labels == -1, 1, 0)
        votes = (kmeans_labels != 0).astype(int) + dbscan_anomaly + nn_anomaly
        combined_anomalies = np.where(votes >= 2, 1, 0)
        
        return combined_anomalies

    def perform_pca(self, scaled_ml_input_data, n_components=2):
        if self.pca_data is None:
            pca = PCA(n_components=n_components)
            self.pca_data = pca.fit_transform(scaled_ml_input_data)
        return self.pca_data

    def perform_tsne(self, scaled_ml_input_data, n_components=3):
        if self.tsne_data is None:
            tsne = TSNE(n_components=n_components, random_state=self.random_state, n_jobs=-1)
            self.tsne_data = tsne.fit_transform(scaled_ml_input_data)
        return self.tsne_data

