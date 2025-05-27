# Chagas Disease Detection

A machine learning pipeline for early detection of Chagas cardiomyopathy using ECG and HRV features.

## Table of Contents

* [Introduction](#introduction)
* [Dataset](#dataset)
* [Preprocessing](#preprocessing)
* [Feature Selection](#feature-selection)
* [Modeling](#modeling)
* [Evaluation](#evaluation)
* [Usage](#usage)
* [Future Work](#future-work)
* [References](#references)

## Introduction

Chagas disease, also known as American trypanosomiasis, is a tropical parasitic disease caused by *Trypanosoma cruzi*. Spread primarily by "kissing bugs" (Triatominae), it can lead to chronic cardiomyopathy decades after infection, with up to 45% of untreated individuals developing heart disease 10–30 years later.

Inspired by Dr. Manu Prakash’s low-cost Foldscope innovations, this project applies machine learning to predict early signs of Chagas-related heart damage from ECG and HRV-derived features.

## Dataset

We use the [Kaggle ECG Chagas Balanced dataset](https://www.kaggle.com/datasets/matteofasuloo/code15-ecg-chagas-balanced/data), which contains:

* ECG feature values (e.g., PR segment, QT interval)
* HRV metrics (e.g., SDNN, CVNN)
* Demographics (e.g., age)
* Balanced labels for Chagas-positive and healthy controls

## Preprocessing

* **Outlier Filtering:** Remove physiologically implausible values based on clinical boundaries (e.g., PR segment > 300ms, QT interval outside 300–500ms).
* **Missing Data:** Impute or drop missing entries.
* **Normalization:** Scale features to zero mean and unit variance.

## Feature Selection

Top features ranked by information gain:

```
1. PR_segment_mean
2. age
3. ST_segment_max
4. ST_segment_mean
5. P_wave_duration_mean
6. QT_interval_mean
7. PR_interval_mean
8. QRS_duration_mean
9. HRV_SDNN
10. HRV_CVNN
11. P_wave_duration_std
12. QT_interval_std
13. HRV_SDSD
14. PR_segment_std
15. HRV_HTI
```

## Modeling

We implemented two supervised learning algorithms:

* **Random Forest**: Ensemble of decision trees to capture non-linear patterns.
* **XGBoost**: Gradient boosting with regularization for improved generalization.

Hyperparameters were tuned via cross-validation.

## Evaluation

Performance metrics include Accuracy, Recall, Precision, F1 Score, and AUC-ROC, evaluated via stratified k-fold cross-validation.

````bash
# Model evaluation results
Accuracy:  0.72
Recall:    0.72
Precision: 0.71
F1 Score:  0.72
AUC-ROC:   0.78
```bash
# Example evaluation output
Accuracy: 0.87
Recall:    0.85
Precision: 0.88
AUC-ROC:   0.91
````

## Usage

1. Clone the repository:

   ```bash
   ```

git clone [https://github.com/yourusername/chagas-detection.git](https://github.com/yourusername/chagas-detection.git)
cd chagas-detection

````
2. Install dependencies:
   ```bash
pip install -r requirements.txt
````

3. Run preprocessing and model training:

   ```bash
   ```

python src/preprocess.py
python src/train.py

```

## Future Work
- Integrate explainability frameworks (SHAP, LIME) for model transparency.
- Validate on external cohorts and larger real-world datasets.
- Develop a lightweight web app for clinician-friendly interaction.

## References
- Wikipedia: [Chagas disease](https://en.wikipedia.org/wiki/Chagas_disease)
- EKG norms: Medscape, ECGWaves
- Machine learning: Chen & Guestrin (2016) on XGBoost

---

*Project by [Your Name](https://github.com/yourusername), UIUC CS 233*

```
