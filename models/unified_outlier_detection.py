# Intel Optimization for Scikit-Learn
from sklearnex import patch_sklearn
patch_sklearn()  # Apply Intel optimizations

# Core Libraries
import numpy as np
import pandas as pd
from scipy.stats import norm, gamma

# Scikit-Learn Imports
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_distances, pairwise_distances



class UnifiedOutlierDetection:
    def __init__(self, random_state=42):
        self.random_state = random_state

    # Step 1: Implement the various outlier detection models

    def knn_outlier_detection(self, data, k=5):
        """ kNN-based outlier detection based on the distance to the kth nearest neighbor. """
        knn = NearestNeighbors(n_neighbors=k)
        knn.fit(data)
        distances, _ = knn.kneighbors(data)
        knn_scores = distances[:, -1]  # Use the distance to the k-th nearest neighbor
        return knn_scores

    def db_outlier_detection(self, data, D=1.0):
        """ Density-based outlier detection (DB-Outlier). Measures the distance of a data point to the rest of the data. """
        distances = pairwise_distances(data)
        db_scores = np.mean(distances > D, axis=1)  # Fraction of points farther than distance D
        return db_scores

    def abod_outlier_detection(self, data):
        """ Angle-Based Outlier Detection (ABOD). Uses the variance in angles between data points. """
        distances = cosine_distances(data)
        angles = np.var(distances, axis=1)  # Variance in angles
        return angles

    def ldof_outlier_detection(self, data, k=5):
        """ Local Distance-Based Outlier Factor (LDOF). Measures the distance of a point to its neighbors and compares it with its neighbors. """
        knn = NearestNeighbors(n_neighbors=k)
        knn.fit(data)
        distances, indices = knn.kneighbors(data)
        # Compute the mean distance to neighbors for each point
        mean_distances = np.mean(distances, axis=1)
        # For each point, compute the mean distance to the neighbors of its neighbors
        neighbor_mean_distances = np.mean(distances, axis=1)  # This computes the mean distances of the neighbors
        # Calculate LDOF by dividing the mean distance to neighbors by the mean of the neighbors' mean distances
        ldof_scores = mean_distances / (neighbor_mean_distances + 1e-6)  # Adding epsilon to avoid division by zero
        return ldof_scores

    def loci_outlier_detection(self, data, epsilon=1.0):
        """ Local Correlation Integral (LOCI). Multi-granularity-based local outlier detection. """
        distances = pairwise_distances(data)
        loci_scores = np.mean(distances < epsilon, axis=1)  # Fraction of points within epsilon distance
        return loci_scores

    # Step 2: Regularization of outlier scores (baseline regularization)

    def regularize_scores(self, scores):
        """ Regularize scores by subtracting the baseline and ensuring all values are >= 0. """
        baseline = np.median(scores)  # Use the median as the baseline
        regularized_scores = np.maximum(0, scores - baseline)
        return regularized_scores

    # Step 3: Normalization of outlier scores using Gaussian scaling

    def gaussian_scaling(self, scores):
        """ Applies Gaussian scaling to the regularized outlier scores. Converts scores to a [0, 1] range. """
        mu, sigma = np.mean(scores), np.std(scores)
        if sigma == 0:
            sigma = 1e-6  # Prevent division by zero
        scaled_scores = norm.cdf(scores, mu, sigma)  # Use the CDF of the normal distribution
        return scaled_scores

    def gamma_scaling(self, scores):
        """ Applies Gamma scaling to the outlier scores. Useful for some types of distributions. """
        shape, loc, scale = gamma.fit(scores)
        scaled_scores = gamma.cdf(scores, shape, loc, scale)
        return scaled_scores

    # Step 4: Logarithmic inversion for contrast enhancement in ABOD

    def logarithmic_inversion(self, scores):
        """ Applies logarithmic inversion for better contrast in low-variance scores. """
        max_score = np.max(scores)
        if max_score == 0:
            max_score = 1e-6  # Prevent division by zero in log
        return -np.log(scores / max_score + 1e-6)

    # Step 5: Unification and Combination of Outlier Scores

    def unify_and_combine_scores(self, data):
        """
        Runs multiple outlier detection methods, regularizes and normalizes the scores,
        then combines them into a single unified score.
        """
        # Step 1: Get outlier scores from different models
        knn_scores = self.knn_outlier_detection(data)
        db_scores = self.db_outlier_detection(data)
        abod_scores = self.abod_outlier_detection(data)
        ldof_scores = self.ldof_outlier_detection(data)
        loci_scores = self.loci_outlier_detection(data)

        # Step 2: Regularize the scores
        knn_scores = self.regularize_scores(knn_scores)
        db_scores = self.regularize_scores(db_scores)
        abod_scores = self.logarithmic_inversion(abod_scores)
        ldof_scores = self.regularize_scores(ldof_scores)
        loci_scores = self.regularize_scores(loci_scores)

        # Step 3: Normalize the scores using Gaussian scaling
        knn_scores = self.gaussian_scaling(knn_scores)
        db_scores = self.gaussian_scaling(db_scores)
        abod_scores = self.gaussian_scaling(abod_scores)
        ldof_scores = self.gaussian_scaling(ldof_scores)
        loci_scores = self.gaussian_scaling(loci_scores)

        # Step 4: Combine the normalized scores (simple mean for this example)
        combined_scores = (knn_scores + db_scores + abod_scores + ldof_scores + loci_scores) / 5
        # rank_based_combination_scores = OutlierScoreCombiner.rank_based_combination(knn_scores, db_scores, abod_scores, ldof_scores, loci_scores, combined_scores)

        return  pd.DataFrame({
            # 'combined_scores':combined_scores,
            'knn_scores_norm': knn_scores,
            'db_scores_norm': db_scores,
            'abod_scores_norm': abod_scores,
            'ldof_scores_norm': ldof_scores,
            'loci_scores_norm': loci_scores
            })

    def run_outlier_detection(self, data):
        """
        Entry function to apply the unified outlier detection.
        """
        unified_scores = self.unify_and_combine_scores(data)
        return unified_scores

    def assign_outlier_labels(self, scores, threshold=0.75):
        """
        Assigns anomaly labels based on a threshold value.
        - Label `1` for outliers
        - Label `0` for normal data
        """
        return np.where(scores > threshold, 1, 0)
