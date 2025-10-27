import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.inspection import permutation_importance

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import joblib

# Paths
ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
REPORTS.mkdir(exist_ok=True, parents=True)
MODELS.mkdir(exist_ok=True, parents=True)

# 1) Load data
iris = load_iris(as_frame=True)
df = iris.frame.copy()
df["target"] = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

X = df[feature_names].values
y = df["target"].values

# 2) Class balance
counts = pd.Series(y).value_counts().sort_index()
counts.index = target_names
counts.to_frame("count").to_csv(REPORTS / "class_balance.csv")

# 3) EDA: pairplot & correlations
sns.pairplot(df, vars=feature_names, hue="target", corner=True)
plt.tight_layout()
plt.savefig(REPORTS / "pairplot.png", dpi=160)
plt.close()

corr = df[feature_names].corr()
plt.figure()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation matrix")
plt.tight_layout()
plt.savefig(REPORTS / "correlation_matrix.png", dpi=160)
plt.close()

# 4) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5) Pipelines + grids
pipelines = {
    "KNN": Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier())]),
    "SVM": Pipeline([("scaler", StandardScaler()), ("clf", SVC(probability=True))]),
    "RF": Pipeline([("clf", RandomForestClassifier(random_state=42))]),
}

param_grids = {
    "KNN": {
        "clf__n_neighbors": [3, 5, 7, 11],
        "clf__weights": ["uniform", "distance"],
        "clf__p": [1, 2],
    },
    "SVM": {
        "clf__C": [0.1, 1, 10, 50],
        "clf__kernel": ["rbf", "linear"],
        "clf__gamma": ["scale", "auto"],
    },
    "RF": {
        "clf__n_estimators": [100, 200, 400],
        "clf__max_depth": [None, 3, 5, 8],
        "clf__min_samples_split": [2, 4],
    },
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []
best_models = {}

for name, pipe in pipelines.items():
    grid = GridSearchCV(pipe, param_grids[name], cv=cv, n_jobs=-1, scoring="f1_macro")
    grid.fit(X_train, y_train)
    best_models[name] = grid.best_estimator_
    results.append({
        "model": name,
        "best_params": grid.best_params_,
        "cv_best_score_f1_macro": grid.best_score_,
    })

# 6) Evaluate on test set
metrics = {}
for name, model in best_models.items():
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    # Save confusion matrix plot
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    plt.savefig(REPORTS / f"confusion_{name}.png", dpi=160)
    plt.close()

    # ROC-AUC (OvR)
    y_test_bin = label_binarize(y_test, classes=[0,1,2])
    # decision function may not exist; fall back to predict_proba
    if hasattr(model[-1], "decision_function"):
        scores = model.decision_function(X_test)
    else:
        scores = model.predict_proba(X_test)
    roc_auc_ovr = roc_auc_score(y_test_bin, scores, multi_class="ovr")

    metrics[name] = {
        "classification_report": report,
        "roc_auc_ovr": roc_auc_ovr,
    }

# Pick best by macro F1 on test
best_name = max(metrics.keys(), key=lambda n: metrics[n]["classification_report"]["macro avg"]["f1-score"])
best_model = best_models[best_name]

# 7) Permutation importance on test for interpretability
r = permutation_importance(best_model, X_test, y_test, n_repeats=20, random_state=42, n_jobs=-1)
importances = pd.Series(r.importances_mean, index=feature_names).sort_values(ascending=False)
plt.figure()
importances.plot(kind="bar")
plt.title(f"Permutation Importance — {best_name}")
plt.ylabel("Mean decrease in score")
plt.tight_layout()
plt.savefig(REPORTS / f"feature_importance_{best_name}.png", dpi=160)
plt.close()

# 8) Save artifacts
joblib.dump(best_model, MODELS / "iris_tabular.pkl")

with open(REPORTS / "model_selection.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

with open(REPORTS / "test_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print(f"[OK] Best model: {best_name}")
print(f"Saved: models/iris_tabular.pkl and reports/*")