# CodeAlpha ML Internship Projects

Repository: `CodeAlpha_MachineLearning`

---

## Task 1 — Handwritten Character Recognition

**Objective:** Identify handwritten digits (0–9) and optionally letters (a–z) using CNNs.

### How to Run
```bash
pip install -r requirements.txt
python task1_handwritten_recognition.py
```

### Architecture — CNN
```
Input (28×28×1)
    ↓
Conv2D(32) → BN → Conv2D(32) → BN → MaxPool → Dropout(0.25)
    ↓
Conv2D(64) → BN → Conv2D(64) → BN → MaxPool → Dropout(0.25)
    ↓
Flatten → Dense(256) → BN → Dropout(0.5)
    ↓
Dense(10, softmax)     ← 10 digits / 26 letters
```

### Key Techniques
| Technique | Why |
|---|---|
| Batch Normalization | Stabilizes training, faster convergence |
| Dropout (0.25 / 0.5) | Prevents overfitting |
| Data Augmentation | Simulates real-world handwriting variation |
| EarlyStopping + ReduceLR | Avoids over-training |

### Expected Results
| Dataset | Accuracy |
|---|---|
| MNIST (digits) | ~99.3% |
| EMNIST (letters) | ~90–93% |

### Output Files
- `whatsapp images are all the outputs` 

### Switch to EMNIST Letters
```python
# In task1_handwritten_recognition.py, line ~35:
USE_EMNIST = True   # Change False → True
```

---

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
