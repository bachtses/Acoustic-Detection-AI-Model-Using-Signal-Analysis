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


## Project Structure

.
├── data_capture/
├── dataset/
├── models/
│
├── data_capture_respeaker.py
├── data_split.py
├── train_model.py
├── test_model.py
├── converter_tflite.py
├── test_model_tflite.py
├── model_inference.py

