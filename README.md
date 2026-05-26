# CodeAlpha ML Internship Projects

Repository: `CodeAlpha_task2`

## Task 2 — Disease Prediction from Medical Data

**Objective:** Predict the likelihood of disease from structured patient data.

### How to Run
```bash
pip install -r requirements.txt
python task2_disease_prediction.py
```

### Datasets
| Dataset | Samples | Features | Task |
|---|---|---|---|
| Heart Disease (Cleveland) | 303 | 13 | Binary classification |
| Diabetes (Pima Indians) | 768 | 8 | Binary classification |
| Breast Cancer (Wisconsin) | 569 | 30 | Binary classification |

### Models Compared
| Model | Type | Notes |
|---|---|---|
| Logistic Regression | Linear | Interpretable baseline |
| SVM (RBF kernel) | Kernel | Strong on small datasets |
| Random Forest | Ensemble | Handles non-linearity, provides feature importances |
| Gradient Boosting | Ensemble | High accuracy, slower |
| XGBoost | Ensemble | Often best overall |
| KNN | Instance-based | Simple, sensitive to scale |

### Pipeline Design
```
Raw Data → Median Imputation → StandardScaling → Model
```
Imputation handles the "zeros as missing values" issue common in medical datasets (e.g., blood pressure = 0 is impossible).

### Evaluation Metrics
- **Cross-validation accuracy** (5-fold stratified)
- **Test accuracy** (80/20 split)
- **AUC-ROC** — primary metric for medical tasks (handles class imbalance)
- **Confusion Matrix** — per-class TP/TN/FP/FN breakdown
- **Feature Importance** — which symptoms/tests matter most

### Output Files (per dataset)
- `task2_eda_*.png` — Class & feature distributions
- `task2_correlation_*.png` — Feature correlation heatmap
- `task2_model_comparison_*.png` — Side-by-side model comparison
- `task2_roc_*.png` — Overlay ROC curves
- `task2_cm_*.png` — Confusion matrix (best model)
- `task2_feature_importance_*.png` — Top predictive features

---

## Project Structure
```
CodeAlpha_MachineLearning/
├── task1_handwritten_recognition.py
├── task2_disease_prediction.py
├── requirements.txt
├── README.md
└── outputs/
    ├── task1_*.png
    ├── task2_*.png
    └── handwriting_cnn_model.keras
```

---

## Tech Stack
- Python 3.9+
- TensorFlow / Keras (Task 1)
- Scikit-learn (Task 2)
- XGBoost (Task 2, optional)
- NumPy, Pandas, Matplotlib, Seaborn

---

*CodeAlpha ML Internship | Tasks completed: Task 1 (Compulsory) + Task 2*
