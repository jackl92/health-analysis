# 🎤 Voice Health Analysis

> Machine learning analysis of voice/audio features to predict health status from speech patterns

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.2-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 📊 Project Overview

Exploratory data analysis and predictive modeling on voice recordings to identify acoustic patterns distinguishing healthy from unhealthy individuals. Analysis of **2,037 voice samples** across **27 audio features** including spectral characteristics, energy metrics, and MFCCs.

### 🎯 Key Objectives
- Identify voice characteristics that differ between healthy/unhealthy individuals
- Build ML models to classify health status from voice features
- Analyze gender-specific patterns in voice health indicators

---

## 📁 Dataset

| Attribute | Details |
|-----------|---------|
| **Source** | VowelA_High_latest.csv |
| **Samples** | 2,037 voice recordings |
| **Features** | 27 audio + 2 demographic |
| **Target** | health_status (healthy/unhealthy) |
| **Split** | 66.2% unhealthy, 33.8% healthy |

**Audio Features:** spectral_centroid, spectral_bandwidth, rolloff, rmse, zero_crossing_rate, chroma_stft, mfcc1-20  
**Demographics:** age, gender

---

## 🔍 Key Findings

### 📈 Demographics
- **Age Gap**: Unhealthy group is 30 years older (median: 52 vs 22) ⚠️ *Major confounder*
- **Gender**: 56.8% female, 43.2% male
- **Imbalance**: 2:1 unhealthy to healthy ratio

### 🎵 Voice Patterns

| Feature | Healthy | Unhealthy | Insight |
|---------|---------|-----------|---------|
| **Spectral Centroid** | Higher (~1200 Hz) | Lower (~1000 Hz) | Darker, less bright voices |
| **Spectral Bandwidth** | Concentrated | Scattered | More irregular frequencies |
| **Rolloff** | Consistent | Variable (1000-8000 Hz) | Breathiness & instability |
| **RMSE** | Higher | Lower | Reduced vocal energy |

### 💡 Clinical Insights
- **Healthy**: Stable, bright, clear voice quality with consistent patterns
- **Unhealthy**: Darker tone, irregular frequencies, vocal fatigue, reduced breath support
- **Gender Effect**: Males show dramatic changes; females remain relatively stable

---

## 🤖 Machine Learning

### Models Evaluated
```
✓ Logistic Regression    (baseline)
✓ Random Forest          (ensemble)
✓ SVM                    (kernel-based)
✓ Gradient Boosting      (advanced ensemble)
✓ Neural Network         (deep learning)
✓ K-Nearest Neighbors    (instance-based)
```

### Performance
- **Best Model**: 85% accuracy
- **Evaluation**: 5-fold cross-validation
- **Metric**: Accuracy (primary)
- **Handling**: Class imbalance via `class_weight='balanced'`
- **Preprocessing**: StandardScaler + one-hot encoding

---

## 🗂️ Repository Structure

```
health-analysis/
├── 📂 data/
│   ├── VowelA_High_latest.csv           # Original dataset
│   └── cleaned_health_data.csv          # Preprocessed data
├── 📂 plots/                             # Auto-saved visualizations (300 DPI)
├── 📂 presentations/
│   └── health_analysis.key              # Keynote presentation
    └── health_analysis.pptx             # Powerpoint presentation
├── 📓 health_analysis_eda.ipynb         # Exploratory analysis
├── 📓 health_analysis_prediction.ipynb  # ML models
├── 📄 health_analysis.pdf               # powerpoint pdf
└── 📄 README.md                         # This file
├── 📄 requirements.txt                  # Dependencies
```

---

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run Analysis
```bash
# EDA notebook
jupyter notebook health_analysis_eda.ipynb

# ML modeling
jupyter notebook health_analysis_prediction.ipynb
```

---

## 📊 Visualizations

### Color Palette
- 🟢 **Healthy**: `#66c2a5` (teal)
- 🟠 **Unhealthy**: `#fc8d62` (orange)  
- 🔵 **Male**: `#8da0cb` (blue)
- 🌸 **Female**: `#e78ac3` (rose)

All plots auto-saved to `plots/` at 300 DPI for publication quality.

---

## ⚠️ Important Notes

> **Age Confounder**: 30-year age gap between groups means many "health" differences may actually be age effects. Future work should control for age.

> **Gender Differences**: Males show dramatic voice changes when unhealthy; females don't. Consider gender-specific models.

> **Class Imbalance**: 2:1 ratio requires careful evaluation beyond accuracy (use precision, recall, F1).

---

## 🎯 Key Takeaways

1. ✅ Voice characteristics **significantly differ** between healthy/unhealthy groups
2. ⚠️ **Age is the primary driver** - 30-year gap is critical confounder
3. 🚻 **Gender matters** - males show stronger health-related changes
4. 🤖 **ML models achieve reasonable accuracy**, but likely rely heavily on age
5. 🎵 **Multiple audio features** provide complementary health information

---

## 🔮 Future Work

- [ ] Age-matched group comparisons
- [ ] Gender-stratified models
- [ ] Feature selection & dimensionality reduction
- [ ] Deep learning (CNN/LSTM) for temporal patterns
- [ ] External dataset validation
- [ ] Clinical expert collaboration

---

## 📦 Dependencies

```
numpy==2.3.4
pandas==2.3.3
matplotlib==3.10.7
seaborn==0.13.2
scikit-learn==1.7.2
```

---