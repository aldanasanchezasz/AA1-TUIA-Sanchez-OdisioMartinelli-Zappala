"""Compara las predicciones del container con la variable real de test_completo.csv."""
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)

AQUI = Path(__file__).resolve().parent
real = pd.read_csv(AQUI / "test_completo.csv")[["Date", "Location", "RainTomorrow"]]
pred = pd.read_csv(AQUI / "datos" / "predicciones.csv")

# Las filas están en el mismo orden; se verifica antes de comparar
assert len(real) == len(pred), "Distinta cantidad de filas"
assert (real["Date"].values == pred["Date"].values).all() and \
       (real["Location"].values == pred["Location"].values).all(), "Las filas no coinciden"

df = real.merge(pred.drop(columns=["Date", "Location"]), left_index=True, right_index=True)
print("Estados del container:\n" + df["estado"].value_counts().to_string() + "\n")

pred_ok = df[df["RainTomorrow_pred"] != "NO_PREDICHO"]
print(f"Filas comparadas: {len(pred_ok)} de {len(df)}\n")
y, yp = pred_ok["RainTomorrow"], pred_ok["RainTomorrow_pred"]

print(f"Accuracy : {accuracy_score(y, yp):.4f}")
print(f"Precision: {precision_score(y, yp, pos_label='Yes'):.4f}")
print(f"Recall   : {recall_score(y, yp, pos_label='Yes'):.4f}")
print(f"F1       : {f1_score(y, yp, pos_label='Yes'):.4f}")
print(f"ROC AUC  : {roc_auc_score(y == 'Yes', pred_ok['prob_lluvia']):.4f}\n")
print("Matriz de confusión (filas = real, columnas = predicho; orden No, Yes):")
print(confusion_matrix(y, yp, labels=["No", "Yes"]), "\n")
print(classification_report(y, yp, digits=3))

df.to_csv(AQUI / "comparacion.csv", index=False)
print("Detalle fila por fila guardado en comparacion.csv")
