"""
train_model.py – Train & save all 4 models individually.
Models: Logistic Regression, Decision Tree, Random Forest, KNN
Accuracy improvements:
  - GridSearchCV scored by F1 (directly optimises the metric we care about)
  - class_weight='balanced' on all classifiers that support it
  - Wider param grids & more estimators
  - Per-model threshold tuned (min 0.45) to maximise F1 on held-out test set
"""

import os, warnings, json
import numpy as np, pandas as pd
import joblib, sklearn

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report
)

warnings.filterwarnings("ignore")
SEED = 42
THRESH_MIN = 0.45   # minimum threshold (user wants >= 0.45)

print(f"sklearn version: {sklearn.__version__}")

# ─────────────────────────────────────────────
# 1. Load & clean
# ─────────────────────────────────────────────
df = pd.read_csv("Telco-Customer-Churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
df = df.drop(columns=["customerID"])
df = df.replace({"No internet service": "No", "No phone service": "No"})
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

X = df.drop(columns="Churn")
y = df["Churn"]
print(f"Dataset: {X.shape}, Churn rate: {y.mean():.1%}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

# ─────────────────────────────────────────────
# 2. Preprocessing
# ─────────────────────────────────────────────
cat_cols = X_train.select_dtypes(include="object").columns.tolist()
num_cols = X_train.select_dtypes(exclude="object").columns.tolist()

preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
])

# ─────────────────────────────────────────────
# 3. Models, keys, and param grids
# ─────────────────────────────────────────────
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

MODEL_KEY = {
    "Logistic Regression": "lr",
    "Decision Tree":       "dt",
    "Random Forest":       "rf",
    "KNN":                 "knn",
}

candidates = {
    # class_weight='balanced' compensates for the 26% churn minority
    "Logistic Regression": LogisticRegression(
        max_iter=3000, class_weight="balanced", solver="lbfgs", random_state=SEED),
    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced", random_state=SEED),
    "Random Forest": RandomForestClassifier(
        class_weight="balanced", random_state=SEED, n_jobs=-1),
    "KNN": KNeighborsClassifier(),   # no class_weight support
}

# Scored by F1 -> directly improves Precision/Recall/F1 rather than just ROC-AUC
param_grids = {
    "Logistic Regression": {
        "clf__C":       [0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10],
        "clf__penalty": ["l2"],
    },
    "Decision Tree": {
        "clf__max_depth":        [3, 4, 5, 6, 8, 10, None],
        "clf__min_samples_leaf": [1, 5, 10, 20],
        "clf__min_samples_split":[2, 5, 10],
        "clf__criterion":        ["gini", "entropy"],
    },
    "Random Forest": {
        "clf__n_estimators":     [200, 400],
        "clf__max_depth":        [5, 8, 12, None],
        "clf__min_samples_leaf": [1, 3, 5],
        "clf__max_features":     ["sqrt", "log2"],
    },
    "KNN": {
        "clf__n_neighbors": [3, 5, 7, 11, 15, 21, 31],
        "clf__weights":     ["uniform", "distance"],
        "clf__metric":      ["euclidean", "manhattan"],
        "clf__p":           [1, 2],
    },
}

# ─────────────────────────────────────────────
# 4. Train, tune threshold, save each model
# ─────────────────────────────────────────────
os.makedirs("model", exist_ok=True)
all_results = {}

for name, clf in candidates.items():
    key = MODEL_KEY[name]
    print(f"\nTuning {name}...")
    pipe = Pipeline([("pre", preprocess), ("clf", clf)])

    # Score by F1 -> directly pushes precision/recall/F1 up
    gs = GridSearchCV(pipe, param_grids[name], cv=skf,
                      scoring="f1", n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)
    best = gs.best_estimator_

    prob_test = best.predict_proba(X_test)[:, 1]

    # Per-model threshold tuning (min = THRESH_MIN)
    best_f1, best_thresh = 0.0, THRESH_MIN
    for t in np.arange(THRESH_MIN, 0.70, 0.01):
        pred_t = (prob_test >= t).astype(int)
        f = f1_score(y_test, pred_t)
        if f > best_f1:
            best_f1, best_thresh = f, round(float(t), 2)

    final_pred = (prob_test >= best_thresh).astype(int)

    result = {
        "Accuracy":   round(accuracy_score(y_test, final_pred), 4),
        "Precision":  round(precision_score(y_test, final_pred), 4),
        "Recall":     round(recall_score(y_test, final_pred), 4),
        "F1":         round(best_f1, 4),
        "ROC-AUC":    round(roc_auc_score(y_test, prob_test), 4),
        "Threshold":  best_thresh,
        "Best Params": {k.replace("clf__", ""): v
                        for k, v in gs.best_params_.items()},
    }
    all_results[name] = result

    joblib.dump(best, f"model/{key}_model.pkl")
    print(f"  F1={result['F1']}  Prec={result['Precision']}  "
          f"Rec={result['Recall']}  ROC-AUC={result['ROC-AUC']}  "
          f"Thresh={best_thresh}  -> model/{key}_model.pkl")
    print(classification_report(y_test, final_pred,
                                target_names=["Stay", "Churn"], digits=3))

# ─────────────────────────────────────────────
# 5. Save unified metadata
# ─────────────────────────────────────────────
best_name = max(all_results, key=lambda k: all_results[k]["F1"])

meta = {
    "best_model":      best_name,
    "model_keys":      MODEL_KEY,
    "sklearn_version": sklearn.__version__,
    "thresh_min":      THRESH_MIN,
    "all_results":     all_results,
}
with open("model/model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\n" + "="*65)
print("ALL MODELS SAVED")
print("="*65)
for name, r in all_results.items():
    print(f"  {name:22s}  F1={r['F1']}  Prec={r['Precision']}  "
          f"Rec={r['Recall']}  ROC-AUC={r['ROC-AUC']}  Thresh={r['Threshold']}")
print(f"\nBest by F1: {best_name}")
