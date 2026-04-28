# Acoustic Detection AI Model Using Signal Analysis

## Overview

This repository contains the full machine learning pipeline for acoustic-based drone detection, including data collection, preprocessing, model training, evaluation, and deployment optimization.

The goal is to develop a robust and lightweight model capable of distinguishing drone sounds from environmental noise in real-world conditions, optimized for edge deployment.

<img width="984" height="222" alt="image" src="https://github.com/user-attachments/assets/03971e92-ab6a-4b8d-be19-be5d8acbaa46" />

## Run
git clone https://github.com/bachtses/Acoustic-Detection-AI-Model-Using-Signal-Analysis.git
cd Acoustic-Detection-AI-Model-Using-Signal-Analysis
sudo python3 test_model.py (in Raspberry-Pi 5) 

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
  
<img width="953" height="299" alt="image" src="https://github.com/user-attachments/assets/b28e2482-9e57-4038-b9cc-55de065b8d1c" />

<img width="973" height="547" alt="image" src="https://github.com/user-attachments/assets/0f3f340e-ac43-445a-9c28-67634a5af68c" />


## Data Preprocessing Pipeline

- Audio loading & normalization
- Resampling to 16 kHz
- Fixed-length segmentation (2 sec)

  
## Feature extraction

- STFT
- Log-Mel Spectrogram

<img width="976" height="492" alt="image" src="https://github.com/user-attachments/assets/3c382205-fa5a-4741-a1d8-0c572453817e" />

  
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

<img width="927" height="437" alt="image" src="https://github.com/user-attachments/assets/1eb3f9c8-3186-426d-985e-e3e7f60007b6" />

<img width="793" height="353" alt="image" src="https://github.com/user-attachments/assets/09135720-ccc9-48a8-b662-d9bd29820a97" />


## Evaluation

- Offline testing on unseen data
- Real-time evaluation using live audio stream
- Distance-based detection experiments
- Model comparison across architectures

<img width="967" height="619" alt="image" src="https://github.com/user-attachments/assets/08a63ca6-539b-498e-b4fd-0d0cc0485e1b" />


## Deployment

- Model converted to TensorFlow Lite for edge inference on Raspberry Pi 5

<img width="279" height="439" alt="image" src="https://github.com/user-attachments/assets/e0576881-576e-4abb-8480-0882664c0632" />


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

  
