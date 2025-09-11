# =========================================
# Clasificación de costo de campeones TFT
# RandomForestClassifier + comparación
# entre entrenamiento normal y validación cruzada
# Métrica: Log Loss (cross entropy)
# =========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    log_loss
)

# =========================
# Funciones auxiliares
# =========================
def plot_confusion(cm, labels, title):
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.title(title)
    plt.tight_layout()

def plot_cv_errors(train_losses, val_losses):
    plt.figure(figsize=(7,5))
    plt.plot(range(1, len(train_losses)+1), train_losses, marker='o', label='Log Loss Train (fold)')
    plt.plot(range(1, len(val_losses)+1), val_losses, marker='o', label='Log Loss Validación (fold)')
    plt.xlabel('Fold')
    plt.ylabel('Log Loss')
    plt.title('Log Loss por fold (CV)')
    plt.legend()
    plt.tight_layout()

def plot_cv_vs_holdout(val_losses, holdout_train_loss, holdout_test_loss, cv_train_logloss):
    plt.figure(figsize=(7,5))
    plt.plot(range(1, len(val_losses)+1), val_losses, marker='o', label='Log Loss Validación (CV)')
    plt.plot(range(1, len(cv_train_logloss)+1), cv_train_logloss, marker='o', label='Log Loss Train (CV)')
    plt.hlines(holdout_test_loss, 1, len(val_losses), colors='r', linestyles='--', label='Log Loss Test Hold-out')
    plt.hlines(holdout_train_loss, 1, len(val_losses), colors='g', linestyles='--', label='Log Loss Train Hold-out')
    plt.xlabel('Fold')
    plt.ylabel('Log Loss')
    plt.title('Comparación Log Loss CV vs Hold-out')
    plt.legend()
    plt.tight_layout()

# =========================
# 1. Cargar datos
# =========================
df = pd.read_csv("./Datasets/TFT_set_14_y_15.csv")
X = df.drop(columns=["cost"])
y = df["cost"]
labels_sorted = sorted(y.unique())

# =========================
# 2. Hold-out split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# 3. Entrenamiento normal (Hold-out)
# =========================
clf_holdout = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
clf_holdout.fit(X_train, y_train)

y_pred_test = clf_holdout.predict(X_test)
y_pred_train = clf_holdout.predict(X_train)
proba_train = clf_holdout.predict_proba(X_train)
proba_test = clf_holdout.predict_proba(X_test)

train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)
train_logloss = log_loss(y_train, proba_train, labels=labels_sorted)
test_logloss = log_loss(y_test, proba_test, labels=labels_sorted)

print("=== Entrenamiento Hold-out ===")
print(f'Accuracy Train: {train_accuracy:.4f}  LogLoss: {train_logloss:.4f}')
print(f'Accuracy Test : {test_accuracy:.4f}  LogLoss: {test_logloss:.4f}')
print("\nReporte de Clasificación (Hold-out):")
print(classification_report(y_test, y_pred_test))
cm_holdout = confusion_matrix(y_test, y_pred_test, labels=labels_sorted)

# =========================
# 4. Validación cruzada (StratifiedKFold) usando Log Loss
# =========================
k = 5
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

cv_train_logloss = []
cv_val_logloss = []
cv_all_true = []
cv_all_pred = []

fold = 1
for train_idx, val_idx in skf.split(X, y):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    clf_cv = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1

    )
    clf_cv.fit(X_tr, y_tr)

    y_tr_pred = clf_cv.predict(X_tr)
    y_val_pred = clf_cv.predict(X_val)
    proba_tr = clf_cv.predict_proba(X_tr)
    proba_val = clf_cv.predict_proba(X_val)

    tr_logloss = log_loss(y_tr, proba_tr, labels=labels_sorted)
    val_logloss = log_loss(y_val, proba_val, labels=labels_sorted)
    cv_train_logloss.append(tr_logloss)
    cv_val_logloss.append(val_logloss)

    cv_all_true.extend(y_val.tolist())
    cv_all_pred.extend(y_val_pred.tolist())

    tr_acc = accuracy_score(y_tr, y_tr_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f'Fold {fold}: Train acc={tr_acc:.4f} LogLoss={tr_logloss:.4f} | Val acc={val_acc:.4f} LogLoss={val_logloss:.4f}')
    fold += 1

cm_cv = confusion_matrix(cv_all_true, cv_all_pred, labels=labels_sorted)

print("\nReporte de Clasificación (CV agregada):")
print(classification_report(
    cv_all_true,
    cv_all_pred,
    labels=labels_sorted,
    target_names=[str(l) for l in labels_sorted],
    zero_division=0
))

# =========================
# 5. Gráficos (Log Loss y predicciones)
# =========================
plot_cv_vs_holdout(cv_val_logloss, train_logloss, test_logloss, cv_train_logloss)

plot_confusion(cm_holdout, labels_sorted, 'Matriz Confusión Hold-out')
plot_confusion(cm_cv, labels_sorted, 'Matriz Confusión CV (agregada)')

# =========================
# 6. Importancia de características
# =========================
importances = pd.Series(clf_holdout.feature_importances_, index=X.columns)
# print("\nImportancia de características (Hold-out model):")
# print(importances.sort_values(ascending=False))

plt.figure(figsize=(8,5))
importances.sort_values(ascending=False).head(15).plot(kind='bar')
plt.title('Top 15 Importancias de Características')
plt.ylabel('Importancia')
plt.tight_layout()

plt.show()
