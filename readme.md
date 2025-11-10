# Voice Health Analysis

Machine learning project analyzing voice/audio features to predict health status and explore patterns that distinguish healthy from unhealthy voices.

## Project Overview

This project performs exploratory data analysis (EDA) and predictive modeling on audio features extracted from vowel "A" sound recordings. The analysis identifies key differences between healthy and unhealthy individuals and builds classification models to predict health status based on voice characteristics.

**Key Focus Areas:**
- Spectral features (bandwidth, centroid, rolloff)
- Energy metrics (RMSE, zero crossing rate)
- Mel-frequency cepstral coefficients (MFCCs 1-20)
- Gender-specific patterns
- Age effects on voice characteristics

## Dataset

- **Source**: `VowelA_High_latest.csv`
- **Size**: 2,037 voice recordings
- **Features**: 27 audio features + 2 demographic variables
  - **Audio features**: chroma_stft, rmse, spectral_centroid, spectral_bandwidth, rolloff, zero_crossing_rate, mfcc1-mfcc20
  - **Demographics**: age, gender
  - **Target**: health_status (healthy/unhealthy)

## Key Findings

### Demographics
- **Age distribution**: Mean 40.7 years (±18.0)
  - Healthy median: 22 years
  - Unhealthy median: 52 years
  - ⚠️ **Critical finding**: 30-year age difference between groups indicates age is a major confounding variable
- **Gender**: 56.8% female, 43.2% male
- **Class imbalance**: 66.2% unhealthy, 33.8% healthy

### Voice Characteristics

**Spectral Centroid** (voice brightness/perceived pitch):
- Unhealthy individuals show ~100-150 Hz lower spectral centroid
- Both genders shift toward darker, less bright voices when unhealthy
- Female voices drop more dramatically: ~1250 Hz → ~1100 Hz
- Male voices: ~1050 Hz → ~950 Hz
- Convergence effect: Unhealthy voices become less gender-distinct

**Spectral Bandwidth** (frequency spread/irregularity):
- Unhealthy voices have more scattered, irregular frequency patterns
- Healthy voices show concentrated, stable distributions
- **Strong gender effect**: Males show much more variability when unhealthy
- Females remain relatively similar whether healthy or unhealthy

**Rolloff** (high-frequency cutoff):
- Healthy individuals: Tight, consistent rolloff distributions
- Unhealthy males: Extreme variability (1000-8000 Hz range!)
- Unhealthy females: Minimal changes compared to healthy
- High rolloff in unhealthy voices suggests breathiness and vocal instability

**RMSE** (voice energy/loudness):
- Range: 0.0-0.4 (normalized amplitude)
- Differences between healthy/unhealthy suggest vocal fatigue or reduced breath support

**Zero Crossing Rate** (breathiness/noise):
- Higher rates indicate more noise, fricative sounds, or breathiness
- Useful indicator of vocal quality degradation

**MFCCs** (spectral shape):
- Low to moderate correlations (mostly < 0.4) = good feature diversity
- No multicollinearity issues for modeling
- Each MFCC captures complementary voice information

### Clinical Interpretation

**Healthy voices:**
- Concentrated, stable frequency patterns
- Consistent spectral characteristics
- Brighter, clearer sound quality
- Low variability within groups

**Unhealthy voices:**
- Scattered, irregular frequencies
- High variability (especially in males)
- Darker, less clear sound
- Loss of spectral clarity and brightness
- Indicators of vocal fatigue, reduced breath support, and irregular voice production

## Machine Learning Results

### Models Tested
- **Logistic Regression** (baseline)
- **Random Forest** (ensemble method)
- **Support Vector Machine** (SVM)
- **Gradient Boosting** (advanced ensemble)
- **Neural Network** (deep learning)
- **K-Nearest Neighbors** (KNN)

### Performance
Models evaluated using 5-fold cross-validation with accuracy as the primary metric. The best performing model is automatically selected and evaluated on the held-out test set.

**Key Considerations:**
- Models handle class imbalance using `class_weight='balanced'`
- Features are standardized using StandardScaler
- Gender is one-hot encoded
- 80/20 train-test split with stratification

## Repository Structure

```
health-analysis/
├── data/
│   ├── VowelA_High_latest.csv       # Original dataset
│   └── cleaned_health_data.csv      # Processed data (standardized labels)
├── plots/                            # Generated visualizations (300 DPI PNG)
├── health_analysis_eda.ipynb        # Exploratory data analysis
├── health_analysis_prediction.ipynb # Machine learning models
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Setup & Installation

### Requirements
```bash
pip install -r requirements.txt
```

### Dependencies
- numpy==2.3.4
- pandas==2.3.3
- matplotlib==3.10.7
- seaborn==0.13.2
- scikit-learn==1.7.2

## Usage

### 1. Exploratory Data Analysis
```bash
jupyter notebook health_analysis_eda.ipynb
```

**What it includes:**
- Data cleaning and preprocessing
- Demographics visualizations (age, gender, health status)
- Audio feature distributions
- Gender-stratified analysis
- MFCC correlation matrix
- All plots auto-saved to `plots/` folder at 300 DPI