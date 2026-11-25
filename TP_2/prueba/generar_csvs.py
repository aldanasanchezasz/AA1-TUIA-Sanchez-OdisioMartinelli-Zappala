"""Genera los CSV de la prueba de punta a punta del container.

Reproduce, sobre el CSV original, las mismas filas y el mismo split que el notebook
(TP-clasificacion-AA1.ipynb, celdas 53, 59 y 61) y guarda:

    test_completo.csv     -> filas de TEST con TODAS las columnas (incluida RainTomorrow)
    datos/input.csv       -> las mismas filas, sin RainTomorrow ni RainfallTomorrow (entrada del container)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

AQUI = Path(__file__).resolve().parent
CSV_ORIGINAL = AQUI.parent / "weatherAUS_2026C2.csv"

df = pd.read_csv(CSV_ORIGINAL)
original = df.copy()

# Celda 53: se eliminan las filas con RainToday y Rainfall ambos faltantes
# (RainToday se imputa con Rainfall si este último está disponible; no cambia las filas)
df = df.drop(index=df.index[df["RainToday"].isna() & df["Rainfall"].isna()])
# Celda 59: se eliminan las filas sin variable objetivo
df = df.dropna(subset=["RainTomorrow", "RainfallTomorrow"])

# Celda 61: split 70/15/15 estratificado (mismos parámetros que el notebook)
y = df["RainTomorrow"]
_, X_temp, _, y_temp = train_test_split(df, y, test_size=0.3, random_state=42, stratify=y)
_, X_test, _, _ = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

test_completo = original.loc[X_test.index].reset_index(drop=True)
test_completo.to_csv(AQUI / "test_completo.csv", index=False)

(test_completo
    .drop(columns=["Unnamed: 0", "RainTomorrow", "RainfallTomorrow"], errors="ignore")
    .to_csv(AQUI / "datos" / "input.csv", index=False))

print(f"Filas de test: {len(test_completo)}")
print(test_completo["RainTomorrow"].value_counts(normalize=True).round(3).to_string())
print("Generados: test_completo.csv y datos/input.csv")
