# Voice Health Analysis - Exploratory Data Analysis

Exploratory data analysis of voice/audio features to identify characteristics that distinguish healthy from unhealthy voices.

## Project Overview

This project analyzes audio features extracted from vowel "A" sound recordings to explore differences between healthy and unhealthy individuals. The analysis focuses on spectral features (bandwidth, centroid, rolloff), energy metrics (RMSE), and Mel-frequency cepstral coefficients (MFCCs), with particular attention to gender-specific patterns.

## Dataset

- **Source**: `VowelA_High_latest.csv`
- **Size**: 2,037 recordings
- **Features**: 
  - Audio metrics: chroma_stft, rmse, spectral_centroid, spectral_bandwidth, rolloff, zero_crossing_rate
  - MFCCs: 20 coefficients (mfcc1-mfcc20)
  - Demographics: age, gender
  - Target: health_status (healthy/unhealthy)

## Key Findings

### Demographics
- **Age distribution**: Mean 40.7 years (±18.0), with significant difference between groups
  - Healthy median: 22 years
  - Unhealthy median: 52 years
  - ⚠️ **Age is a major confounder** (30-year difference)
- **Gender**: 56.8% female, 43.2% male
- **Class imbalance**: 66.2% unhealthy, 33.8% healthy
