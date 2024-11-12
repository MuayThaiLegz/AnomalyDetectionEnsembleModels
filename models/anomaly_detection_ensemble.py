# Core Libraries
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# Scikit-Learn Imports
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.cluster import KMeans, DBSCAN

# Keras/TensorFlow for Deep Learning Autoencoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input


class AnomalyDetectionEnsemble:
    def __init__(self, random_state=42, cv=4, anomaly_label=1, normal_label=0, outlier_label=-1):
        self.random_state = random_state
        self.cv = cv
        self.anomaly_label = anomaly_label
        self.normal_label = normal_label
        self.outlier_label = outlier_label

    def sklearn_models(self, raw_input_data, scaled_input_data):
        """
        Applies multiple scikit-learn anomaly detection models to the scaled input data and updates
        the raw_input_data DataFrame with anomaly labels and scores.

        :param raw_input_data: Original input DataFrame to be updated with anomaly information.
        :param scaled_input_data: Scaled numerical data for model input.
        :return: Tuple of (updated scaled_input_data, updated raw_input_data).
        """
        # Initialize anomaly detection algorithms with set random_state for reproducibility
        """
        I feel the need to to hyperperameter tunning on the these
        """
        anomaly_algorithms = {
            'EllipticEnvelope': EllipticEnvelope(support_fraction=0.45, random_state=self.random_state),
            'IsolationForest': IsolationForest(random_state=self.random_state),
            'OneClassSVM': OneClassSVM(kernel='rbf'),
            'LocalOutlierFactor': LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True),
        }

        for name, algorithm in anomaly_algorithms.items():
            # Fit the model and predict anomalies
            algorithm.fit(scaled_input_data)
            predictions = algorithm.predict(scaled_input_data)
            scores = algorithm.decision_function(scaled_input_data)

            # Standardize the scores for consistency
            scaler = StandardScaler()
            scores_scaled = scaler.fit_transform(scores.reshape(-1, 1)).flatten()

            # Update the raw_input_data DataFrame
            anomaly_label_column = f"{name}_Anomaly"
            anomaly_score_column = f"{name}_Anomaly_Score"

            raw_input_data[anomaly_label_column] = np.where(
                predictions == self.outlier_label,
                self.anomaly_label,
                self.normal_label
            )
            raw_input_data[anomaly_score_column] = scores_scaled

        return scaled_input_data, raw_input_data

    def train_autoencoder(self, data, epochs=70, batch_size=16, test_size=0.2):
        """
        Trains an autoencoder on the given data and returns the normalized reconstruction errors
        and the anomaly threshold.

        :param data: Scaled numerical input data.
        :param epochs: Number of training epochs.
        :param batch_size: Training batch size.
        :param test_size: Fraction of data to use for validation.
        :return: Tuple of (normalized reconstruction errors, anomaly threshold).
        """
        train_data, test_data = train_test_split(data, test_size=test_size, random_state=self.random_state)
        input_dim = train_data.shape[1]
        encoding_dim = int(input_dim / 2)

        # Define the autoencoder model
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(encoding_dim, activation='relu')(input_layer)
        decoded = Dense(input_dim, activation='linear')(encoded)

        autoencoder = Model(inputs=input_layer, outputs=decoded)
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')

        # Train the autoencoder
        autoencoder.fit(
            train_data, train_data,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(test_data, test_data),
            verbose=0
        )

        # Compute reconstruction errors
        data_predictions = autoencoder.predict(data)
        reconstruction_error = np.mean(np.power(data - data_predictions, 2), axis=1)

        # Normalize the reconstruction error
        normalized_error = (reconstruction_error - reconstruction_error.min()) / (reconstruction_error.max() - reconstruction_error.min())

        # Determine the anomaly threshold
        threshold = np.percentile(normalized_error, 90)

        return normalized_error, threshold

    def run_autoencoder(self, scaled_input_data, raw_input_data):
        """
        Applies the trained autoencoder to detect anomalies and updates the raw_input_data DataFrame.

        :param scaled_input_data: Scaled numerical input data.
        :param raw_input_data: DataFrame to be updated with autoencoder anomaly information.
        :return: Tuple of (scaled_input_data, updated raw_input_data).
        """
        normalized_error, threshold = self.train_autoencoder(scaled_input_data)

        # Update the raw_input_data DataFrame
        raw_input_data['Autoencoder_Anomaly'] = np.where(
            normalized_error > threshold,
            self.anomaly_label,
            self.normal_label
        )
        raw_input_data['Autoencoder_Anomaly_Score'] = normalized_error

        return scaled_input_data, raw_input_data

    def run_anomaly_detection_models(self, raw_data, scaled_data):
        """
        Runs all anomaly detection models and returns the updated DataFrames.

        :param raw_data: Original input DataFrame.
        :param scaled_data: Scaled numerical data.
        :return: Tuple of (updated scaled_data, updated raw_data).
        """
        scaled_data, raw_data = self.sklearn_models(raw_data, scaled_data)
        scaled_data, raw_data = self.run_autoencoder(scaled_data, raw_data)

        return scaled_data, raw_data