# =========================================
# Clasificación de costo de campeones TFT
# Usando RandomForestClassifier
# =========================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. Cargar datos
df = pd.read_csv("./Datasets/TFT_Champion_Transformed.csv")

# 2. Separar variables independientes (X) y dependiente (y)
X = df.drop(columns=["cost"])
y = df["cost"]

# 3. Dividir en train y test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Crear y entrenar modelo
clf = RandomForestClassifier(
    n_estimators=200,   # número de árboles
    max_depth=None,     # profundidad ilimitada (ajústalo si sobreajusta)
    random_state=42
)
clf.fit(X_train, y_train)

# 5. Predicciones
y_pred = clf.predict(X_test)

# 6. Evaluación
print("Reporte de Clasificación:")
print(classification_report(y_test, y_pred))

print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred))

# 7. Importancia de características
importances = pd.Series(clf.feature_importances_, index=X.columns)
print("\nImportancia de características:")
print(importances.sort_values(ascending=False))
