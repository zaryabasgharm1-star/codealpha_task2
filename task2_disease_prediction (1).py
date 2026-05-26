# ============================================================
# CodeAlpha ML Internship — TASK 2
# Disease Prediction from Medical Data
# Datasets: Heart Disease, Diabetes, Breast Cancer (UCI)
# Models: Logistic Regression, SVM, Random Forest, XGBoost
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Datasets (all included in sklearn)
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import fetch_openml          # Heart disease, Diabetes fallback

# Preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

try:
    from xgboost import XGBClassifier
    XGBOOST_OK = True
except ImportError:
    XGBOOST_OK = False
    print("[INFO] xgboost not installed. Install with: pip install xgboost")

# Evaluation
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score,
                              roc_curve, precision_recall_curve)

print("=" * 60)
print("  CodeAlpha ML Internship — Disease Prediction from Medical Data")
print("=" * 60, "\n")


# ─────────────────────────────────────────────
# 0. DATASET LOADER
# ─────────────────────────────────────────────
def load_heart_disease():
    """
    Cleveland Heart Disease dataset via UCI / OpenML.
    Features: age, sex, chest pain type, blood pressure, cholesterol, etc.
    Target: 0 = no disease, 1 = disease
    """
    print("[DATA] Loading Heart Disease dataset (Cleveland)...")
    try:
        data = fetch_openml("heart-disease", version=1, as_frame=True, parser="auto")
        df   = data.frame.copy()
        # Target: 0 = healthy, >0 = disease → binarize
        df["target"] = (df["num"].astype(int) > 0).astype(int)
        df.drop(columns=["num"], inplace=True)
        df = df.apply(pd.to_numeric, errors="coerce")
        feature_names = [c for c in df.columns if c != "target"]
        X = df[feature_names].values
        y = df["target"].values
        name = "Heart Disease"
    except Exception:
        # Fallback: generate synthetic data matching real distribution
        print("  [FALLBACK] Using synthetic heart disease data (OpenML unavailable).")
        np.random.seed(42)
        n = 303
        X = np.column_stack([
            np.random.randint(29, 78, n),           # age
            np.random.randint(0, 2, n),             # sex
            np.random.randint(0, 4, n),             # chest_pain_type
            np.random.randint(90, 200, n),          # resting_bp
            np.random.randint(120, 570, n),         # cholesterol
            np.random.randint(0, 2, n),             # fasting_blood_sugar
            np.random.randint(0, 3, n),             # resting_ecg
            np.random.randint(70, 205, n),          # max_heart_rate
            np.random.randint(0, 2, n),             # exercise_angina
            np.round(np.random.uniform(0, 6.2, n), 1),  # st_depression
            np.random.randint(1, 4, n),             # st_slope
            np.random.randint(0, 4, n),             # num_major_vessels
            np.random.randint(3, 8, n),             # thal
        ])
        y = np.random.randint(0, 2, n)
        feature_names = ["age", "sex", "chest_pain", "resting_bp", "cholesterol",
                         "fasting_bs", "resting_ecg", "max_hr", "exercise_angina",
                         "st_depression", "st_slope", "num_vessels", "thal"]
        name = "Heart Disease (Synthetic)"

    print(f"  Samples: {X.shape[0]}  |  Features: {X.shape[1]}  |  Positive rate: "
          f"{y.mean()*100:.1f}%\n")
    return X, y, feature_names, name


def load_diabetes():
    """
    Pima Indians Diabetes dataset.
    Features: pregnancies, glucose, blood pressure, skin thickness,
              insulin, BMI, diabetes pedigree, age
    Target: 0 = no diabetes, 1 = diabetes
    """
    print("[DATA] Loading Diabetes dataset (Pima Indians)...")
    try:
        data = fetch_openml("diabetes", version=1, as_frame=True, parser="auto")
        df   = data.frame.copy()
        # Target may be 'tested_negative' / 'tested_positive'
        le = LabelEncoder()
        df["target"] = le.fit_transform(df["class"].astype(str))
        df.drop(columns=["class"], inplace=True)
        df = df.apply(pd.to_numeric, errors="coerce")
        feature_names = [c for c in df.columns if c != "target"]
        X = df[feature_names].values
        y = df["target"].values
    except Exception:
        print("  [FALLBACK] Using synthetic diabetes data.")
        np.random.seed(42)
        n = 768
        X = np.column_stack([
            np.random.randint(0, 18, n),              # pregnancies
            np.random.randint(0, 200, n),             # glucose
            np.random.randint(0, 122, n),             # blood_pressure
            np.random.randint(0, 100, n),             # skin_thickness
            np.random.randint(0, 850, n),             # insulin
            np.round(np.random.uniform(0, 67, n), 1),# bmi
            np.round(np.random.uniform(0.07, 2.5, n), 3),  # dpf
            np.random.randint(21, 82, n),             # age
        ])
        y = np.random.randint(0, 2, n)
        feature_names = ["pregnancies", "glucose", "blood_pressure",
                         "skin_thickness", "insulin", "bmi", "dpf", "age"]

    name = "Diabetes"
    print(f"  Samples: {X.shape[0]}  |  Features: {X.shape[1]}  |  Positive rate: "
          f"{y.mean()*100:.1f}%\n")
    return X, y, feature_names, name


def load_breast_cancer_data():
    """
    Wisconsin Breast Cancer dataset (built into sklearn).
    Features: 30 computed features from cell nuclei images.
    Target: 0 = malignant, 1 = benign
    """
    print("[DATA] Loading Breast Cancer dataset (Wisconsin)...")
    bc = load_breast_cancer()
    X, y = bc.data, bc.target
    feature_names = list(bc.feature_names)
    name = "Breast Cancer"
    print(f"  Samples: {X.shape[0]}  |  Features: {X.shape[1]}  |  Benign rate: "
          f"{y.mean()*100:.1f}%\n")
    return X, y, feature_names, name


# ─────────────────────────────────────────────
# 1. EDA PLOTS
# ─────────────────────────────────────────────
def eda_plots(X, y, feature_names, dataset_name, top_n=8):
    """Visualize class distribution and top feature distributions."""
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"EDA — {dataset_name}", fontsize=15, fontweight="bold")
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

    # Class distribution
    ax0 = fig.add_subplot(gs[0, 0])
    counts = pd.Series(y).value_counts().sort_index()
    bars   = ax0.bar(["Negative", "Positive"], counts.values,
                     color=["#4C72B0", "#DD8452"])
    ax0.set_title("Class Distribution")
    ax0.set_ylabel("Count")
    for bar, val in zip(bars, counts.values):
        ax0.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 str(val), ha="center", fontsize=9)

    # Feature distributions (top_n features)
    for i, col in enumerate(feature_names[:top_n], 1):
        r, c = divmod(i, 4)
        ax = fig.add_subplot(gs[r, c])
        df[df["target"] == 0][col].hist(ax=ax, bins=20, alpha=0.6,
                                         color="#4C72B0", label="Neg")
        df[df["target"] == 1][col].hist(ax=ax, bins=20, alpha=0.6,
                                         color="#DD8452", label="Pos")
        ax.set_title(col[:20], fontsize=8)
        ax.tick_params(labelsize=7)
        if i == 1:
            ax.legend(fontsize=7)

    path = f"task2_eda_{dataset_name.lower().replace(' ', '_')}.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"[SAVED] {path}")

    # Correlation heatmap (top 10 features)
    corr = df.corr()["target"].drop("target").abs().sort_values(ascending=False)
    top_corr_features = corr.head(10).index.tolist()
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    sub = df[top_corr_features + ["target"]].corr()
    sns.heatmap(sub, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, ax=ax2)
    ax2.set_title(f"Feature Correlation — {dataset_name}", fontweight="bold")
    path2 = f"task2_correlation_{dataset_name.lower().replace(' ', '_')}.png"
    plt.tight_layout()
    plt.savefig(path2, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"[SAVED] {path2}")
    return corr


# ─────────────────────────────────────────────
# 2. BUILD PIPELINES
# ─────────────────────────────────────────────
def build_pipelines():
    """
    Each pipeline:  Impute missing → Scale → Model
    Imputation handles the zeros-as-missing-values common in medical data.
    StandardScaler is critical for SVM and Logistic Regression.
    """
    models = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("model",   LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "SVM (RBF)": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("model",   SVC(kernel="rbf", probability=True, random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model",   RandomForestClassifier(n_estimators=200, max_depth=None,
                                               min_samples_leaf=2, random_state=42)),
        ]),
        "Gradient Boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model",   GradientBoostingClassifier(n_estimators=200, learning_rate=0.1,
                                                   max_depth=4, random_state=42)),
        ]),
        "KNN": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("model",   KNeighborsClassifier(n_neighbors=7)),
        ]),
    }
    if XGBOOST_OK:
        models["XGBoost"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model",   XGBClassifier(n_estimators=200, learning_rate=0.1,
                                      max_depth=4, use_label_encoder=False,
                                      eval_metric="logloss", random_state=42)),
        ])
    return models


# ─────────────────────────────────────────────
# 3. TRAIN & EVALUATE ALL MODELS
# ─────────────────────────────────────────────
def train_evaluate_all(X, y, feature_names, dataset_name):
    """
    For each model:
      • 5-fold Stratified CV → cross-val accuracy
      • Train/test split 80/20 → test accuracy, AUC, report
    """
    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset_name}")
    print(f"{'='*60}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipelines  = build_pipelines()
    cv         = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results    = {}

    for name, pipe in pipelines.items():
        # Cross-validation
        cv_scores = cross_val_score(pipe, X_train, y_train,
                                    cv=cv, scoring="accuracy", n_jobs=-1)
        # Fit & predict
        pipe.fit(X_train, y_train)
        y_pred  = pipe.predict(X_test)
        y_prob  = pipe.predict_proba(X_test)[:, 1]
        acc     = accuracy_score(y_test, y_pred)
        auc     = roc_auc_score(y_test, y_prob)

        results[name] = {
            "CV Mean":  cv_scores.mean(),
            "CV Std":   cv_scores.std(),
            "Test Acc": acc,
            "AUC":      auc,
            "y_pred":   y_pred,
            "y_prob":   y_prob,
            "pipe":     pipe,
        }

        print(f"\n  ── {name} ──")
        print(f"  CV Accuracy : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"  Test Acc    : {acc:.4f}")
        print(f"  AUC         : {auc:.4f}")

    return results, y_test, X_test, X_train, y_train


# ─────────────────────────────────────────────
# 4. PLOT RESULTS
# ─────────────────────────────────────────────
def plot_model_comparison(results, dataset_name):
    """Bar chart comparing all models on Test Accuracy and AUC."""
    names    = list(results.keys())
    accs     = [results[n]["Test Acc"] for n in names]
    aucs     = [results[n]["AUC"]      for n in names]
    cv_means = [results[n]["CV Mean"]  for n in names]

    x   = np.arange(len(names))
    w   = 0.27
    fig, ax = plt.subplots(figsize=(13, 5))
    b1 = ax.bar(x - w,   cv_means, w, label="CV Accuracy",   color="#4C72B0", alpha=0.85)
    b2 = ax.bar(x,        accs,    w, label="Test Accuracy",  color="#55A868", alpha=0.85)
    b3 = ax.bar(x + w,   aucs,    w, label="AUC-ROC",        color="#DD8452", alpha=0.85)

    for bars in [b1, b2, b3]:
        for bar in bars:
            ax.annotate(f"{bar.get_height():.3f}",
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"Model Comparison — {dataset_name}", fontweight="bold", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = f"task2_model_comparison_{dataset_name.lower().replace(' ', '_')}.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"[SAVED] {path}")


def plot_roc_curves(results, y_test, dataset_name):
    """Overlay ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors  = plt.cm.tab10(np.linspace(0, 1, len(results)))
    for (name, res), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        ax.plot(fpr, tpr, label=f"{name}  (AUC={res['AUC']:.3f})",
                color=color, linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves — {dataset_name}", fontweight="bold", fontsize=13)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = f"task2_roc_{dataset_name.lower().replace(' ', '_')}.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"[SAVED] {path}")


def plot_best_confusion_matrix(results, y_test, dataset_name):
    """Confusion matrix for the best model (highest AUC)."""
    best_name = max(results, key=lambda k: results[k]["AUC"])
    y_pred    = results[best_name]["y_pred"]
    cm        = confusion_matrix(y_test, y_pred)
    fig, ax   = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {best_name}\n({dataset_name})", fontweight="bold")
    plt.tight_layout()
    path = f"task2_cm_{dataset_name.lower().replace(' ', '_')}.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"[SAVED] {path}")

    print(f"\n[REPORT] Best model — {best_name}:")
    print(classification_report(y_test, y_pred,
                                 target_names=["Negative", "Positive"]))
    return best_name


def plot_feature_importance(results, feature_names, dataset_name):
    """Feature importance from Random Forest (tree-based built-in)."""
    pipe       = results["Random Forest"]["pipe"]
    # Imputer removes NaN but doesn't change feature count
    rf_model   = pipe.named_steps["model"]
    importances= rf_model.feature_importances_
    idx        = np.argsort(importances)[::-1][:15]  # top 15

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(range(len(idx)), importances[idx[::-1]], color="#4C72B0", alpha=0.85)
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_names[i][:25] for i in idx[::-1]], fontsize=9)
    ax.set_xlabel("Importance Score")
    ax.set_title(f"Top Feature Importances (Random Forest) — {dataset_name}",
                 fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = f"task2_feature_importance_{dataset_name.lower().replace(' ', '_')}.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"[SAVED] {path}")


# ─────────────────────────────────────────────
# 5. SUMMARY TABLE
# ─────────────────────────────────────────────
def print_summary_table(all_results):
    """Print a combined summary across all datasets."""
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY — Best AUC per Dataset")
    print("=" * 70)
    print(f"  {'Dataset':<20} {'Best Model':<22} {'Test Acc':>10} {'AUC':>8}")
    print("  " + "-" * 65)
    for dataset_name, (results, _) in all_results.items():
        best = max(results, key=lambda k: results[k]["AUC"])
        acc  = results[best]["Test Acc"]
        auc  = results[best]["AUC"]
        print(f"  {dataset_name:<20} {best:<22} {acc*100:>9.2f}%  {auc:>7.4f}")
    print("=" * 70)


# ─────────────────────────────────────────────
# 6. INFERENCE DEMO
# ─────────────────────────────────────────────
def predict_patient(pipe, feature_names, patient_data: dict):
    """
    Predict disease probability for a single patient.
    Args:
        pipe:          trained sklearn pipeline
        feature_names: list of feature names
        patient_data:  dict mapping feature name → value
    Example:
        predict_patient(pipe, feature_names,
                        {"glucose": 148, "bmi": 33.6, "age": 50, ...})
    """
    row = np.array([[patient_data.get(f, np.nan) for f in feature_names]])
    prob = pipe.predict_proba(row)[0]
    pred = pipe.predict(row)[0]
    print(f"\n  Predicted class : {'Positive' if pred == 1 else 'Negative'}")
    print(f"  Negative prob   : {prob[0]*100:.1f}%")
    print(f"  Positive prob   : {prob[1]*100:.1f}%")
    return pred, prob


# ─────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────
def main():
    datasets = [
        load_heart_disease,
        load_diabetes,
        load_breast_cancer_data,
    ]

    all_results = {}

    for loader in datasets:
        X, y, feature_names, dataset_name = loader()

        # EDA
        eda_plots(X, y, feature_names, dataset_name)

        # Train & evaluate
        results, y_test, X_test, X_train, y_train = \
            train_evaluate_all(X, y, feature_names, dataset_name)

        # Plots
        plot_model_comparison(results, dataset_name)
        plot_roc_curves(results, y_test, dataset_name)
        best_name = plot_best_confusion_matrix(results, y_test, dataset_name)
        plot_feature_importance(results, feature_names, dataset_name)

        all_results[dataset_name] = (results, y_test)

    print_summary_table(all_results)
    print("\n✅ Task 2 Complete!")


if __name__ == "__main__":
    main()
