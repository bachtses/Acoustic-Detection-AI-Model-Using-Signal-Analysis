# Acoustic Detection AI Model Using Signal Analysis

## Overview

This repository contains the full machine learning pipeline for acoustic-based drone detection, including data collection, preprocessing, model training, evaluation, and deployment optimization.

The goal is to develop a robust and lightweight model capable of distinguishing drone sounds from environmental noise in real-world conditions, optimized for edge deployment.


## Problem Statement

Detect the presence of UAVs using their acoustic signatures in noisy outdoor environments.

Challenges:
- Background noise (wind, traffic, people)
- Variability in drone distance and orientation
- Real-time constraints for edge devices


## Dataset

Custom data capture pipeline (~12,000+ samples)
Classes:
- Drone
- Background (noise, speech, cars, etc.)
Recorded using:
- ReSpeaker 4-Mic Array
- Real-world outdoor recordings


## Data Preprocessing Pipeline

- Audio loading & normalization
- Resampling to 16 kHz
- Fixed-length segmentation (2 sec)

  
## Feature extraction

- STFT
- Log-Mel Spectrogram

  
## Data augmentation

- Gaussian noise
- Time shifting
- Volume scaling


## Model Development

- Input: Log-Mel Spectrogram
- Architecture: 2D CNN
- Output: Binary classification (Drone / No Drone)


## Model Training

- Train/Test split: 80/20
- Balanced dataset
- Deterministic training setup (fixed seeds)


## Metrics

- Accuracy
- Precision
- Recall
- Confusion Matrix


## Evaluation

- Offline testing on unseen data
- Real-time evaluation using live audio stream
- Distance-based detection experiments
- Model comparison across architectures


## Deployment

- Model converted to TensorFlow Lite for edge inference:


## Includes scripts for

- Dataset validation (data_checker)
- Visualization (data_visualization)
- Distance-based evaluation
- Model comparison (live & offline)


## Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Librosa
- PyAudio
- Matplotlib


## Key Outcomes

- Robust acoustic detection in noisy environments
- Lightweight model suitable for edge devices
- Real-time inference capability
- End-to-end ML pipeline from raw audio to deployment

  
