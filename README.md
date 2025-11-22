# Confusion Detection from EEG, Video & Demographic
Dataset: [https://www.kaggle.com/datasets/wanghaohan/confused-eeg](https://www.kaggle.com/datasets/wanghaohan/confused-eeg)

This project builds a multimodal classifier that detects learner confusion using:

* EEG time-series
* Video frames (ResNet-50 embeddings)
* Demographic info

A TCN encoder processes EEG signals, and a learned attention module fuses EEG, video, and demographic features. Evaluation uses Leave-One-Subject-Out (LOSO) cross-validation.

## Features

* EEG resampling → 120×C + per-trial normalization
* Optional augmentation: noise, channel dropout, time masking
* Video frame sampling + ResNet-50 mean-pooled embeddings
* One-hot demographic vectors
* Attention-based fusion model
* XGBoost ensemble
* Metrics: F1, AUC, accuracy, precision, recall + threshold tuning

All results saved to `./artifacts/`, including:

* ROC/PR curves
* Confusion matrices
* Per-fold metrics
* Attention weights

## Results 

Some overall aggregated results include: 
* F1 Score - 0.89
* AUC - 0.86
* Accuracy - 0.88
* Precision - 0.84
* Recall - 0.98

