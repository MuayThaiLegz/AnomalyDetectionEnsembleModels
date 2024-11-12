# AnomalyDetectionEnsembleModels

### A Comprehensive Ensemble-Based Approach to Anomaly Detection

This repository showcases a series of ensemble models for detecting anomalies across diverse datasets, designed to leverage the strengths of multiple anomaly detection techniques. It serves as a demonstration of advanced ensemble methods, adaptive tuning, and regularization techniques, providing a robust foundation for academic research, data analysis, and machine learning applications.

---

## Overview

Anomaly detection is critical in various domains, including finance, healthcare, manufacturing, and cybersecurity, where identifying outliers can highlight issues ranging from fraudulent activities to system malfunctions. This project provides a unified approach to anomaly detection, combining classical, machine learning, and deep learning methods within ensemble frameworks to improve anomaly detection accuracy and adaptability across different data distributions.

### Key Features:
- **Unified Outlier Detection Model**: Combines multiple classical and machine learning outlier detection techniques, applying regularization and normalization to create a unified outlier score.
- **Anomaly Detection Ensemble**: Integrates traditional models (e.g., Isolation Forest, One-Class SVM) with an autoencoder for non-linear anomaly detection, achieving high accuracy across non-linear datasets.
- **Fast Anomaly Ensemble**: Utilizes adaptive parameter tuning, clustering, and density-based methods for rapid and scalable anomaly detection.

---
## Models and Methodology

1. **Unified Outlier Detection**: 
   - A comprehensive approach combining distance-based, density-based, and angle-based outlier detection methods.
   - Applies regularization and Gaussian scaling to normalize outlier scores across methods.

2. **Anomaly Detection Ensemble**: 
   - Employs a mix of machine learning models with autoencoders to identify complex, non-linear anomalies.
   - Hyperparameter tuning is recommended to maximize model performance across datasets with diverse characteristics.

3. **Fast Anomaly Ensemble**:
   - Built for scalability, this model uses adaptive clustering (e.g., K-means and DBSCAN) with optimized parameters for rapid deployment.
   - Suitable for applications requiring high computational efficiency and adaptive performance.

4. **Voted Anomaly Labels for Classifier Training**:
   - After deriving the anomaly labels through an ensemble voting mechanism across these models, the final “voted” anomaly label will be used as ground truth to train a powerful classifier.
   - This classifier can be trained using supervised learning to enhance future anomaly detection accuracy, benefiting from the ensemble models’ insights.
   - By combining unsupervised anomaly detection with supervised learning, the model can generalize more effectively in complex environments.

---

## Notebooks on Kaggle

Each notebook provides an interactive walkthrough, detailing the steps involved in data preprocessing, model selection, parameter tuning, and evaluation:

- [Unified Outlier Detection Model on Kaggle](https://www.kaggle.com/code/javierbarrnun/unifyingoutlierscores) - Demonstrates the Unified Outlier Detection model, focusing on combining traditional outlier detection techniques.
- [Anomaly Detection Ensemble on Kaggle](https://www.kaggle.com/code/javierbarrnun/anomalydetectionensemblesystem) - Walkthrough for the Anomaly Detection Ensemble model, illustrating the integration of classical and deep learning approaches.
- [Fast Anomaly Ensemble on Kaggle](https://www.kaggle.com/code/javierbarrnun/fastirvisual) - Guide for the Fast Anomaly Ensemble model, with an emphasis on adaptive tuning and computational efficiency.

---
