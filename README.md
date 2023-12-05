# Spectral Densities for Dimensionality Reduction

## Overview
This project explores the use of higher-order spectral densities (HOSD) for dimensionality reduction of audio signals, focusing on musical instrument sounds. We have developed a machine learning pipeline using Power Spectral Density (PSD), Skewness Spectral Density (SSD), and Kurtosis Spectral Density (KSD) for unique acoustic signatures identification.

## Data
The dataset comprises high-quality recordings of various musical instruments, specifically focusing on the middle C note (261.35 Hz). It includes 12 different instruments recorded using one or up to 4 different microphones.

## Methodology
- **Feature Extraction**: Extract key spectral features such as PSD, SSD, and KSD from audio samples.
- **Machine Learning Models**: Use Ensemble learning, Support Vector Classifier (SVC), KNearestNeighbours, and Decision trees for classification.
- **Evaluation Metrics**: Utilize confusion matrix, log-loss, and accuracy metrics for model evaluation.

## Applications
- Musical instrument classification.
- Potential industrial fault detection through vibrational data analysis.

## Future Work
- Refinement of models based on feature importance analysis.
- Application of model explainability techniques like SHAP and LIME.
- Noise reduction and signal enhancement for improved data quality.


