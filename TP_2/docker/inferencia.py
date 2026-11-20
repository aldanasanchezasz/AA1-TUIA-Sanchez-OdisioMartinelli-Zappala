"""Inferencia: ¿lloverá mañana? (RainTomorrow) con la red neuronal optimizada del TP.

Uso (dentro del container, ver readme.md):
    python inferencia.py [--input /data/input.csv] [--output /data/predicciones.csv]

Lee un CSV con las mismas columnas que weatherAUS, repite la limpieza del notebook usando
los objetos ajustados con TRAIN (no se reajusta nada) y escribe un CSV con las predicciones.

Cada fila queda en uno de estos estados, según el % de variables predictoras faltantes:
    OK              -> predice normalmente.
    ADVERTENCIA     -> predice, pero avisa que faltan muchos datos y puede equivocarse.
    SIN_PREDICCION  -> no predice: falta demasiada información.
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # menos ruido de TensorFlow en consola

import joblib
import numpy as np
import pandas as pd

ART = Path(__file__).resolve().parent / "artefactos"

# Variables predictoras tal como llegan en el CSV (se usan para medir el % de faltantes)
PREDICTORAS = [
    "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine", "WindGustDir", "WindGustSpeed",
    "WindDir9am", "WindDir3pm", "WindSpeed9am", "WindSpeed3pm", "Humidity9am", "Humidity3pm",
    "Pressure9am", "Pressure3pm", "Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm", "RainToday"]

# Misma codificación cíclica que en el notebook
PUNTOS_CARDINALES = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                     "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
GRADOS_VIENTO = {d: i * 360 / 16 for i, d in enumerate(PUNTOS_CARDINALES)}
COLUMNAS_VIENTO = ["WindGustDir", "WindDir9am", "WindDir3pm"]

# Umbrales por defecto de % de faltantes (se pueden cambiar por argumento o variable de entorno)
UMBRAL_ADVERTENCIA = 0.30   # a partir de acá se predice pero con advertencia
UMBRAL_MAXIMO = 0.50        # a partir de acá no se predice


def parsear_argumentos():
    p = argparse.ArgumentParser(description="Predicción de lluvia para mañana (RainTomorrow).")
    p.add_argument("--input", default=os.environ.get("INPUT_CSV", "/data/input.csv"))
    p.add_argument("--output", default=os.environ.get("OUTPUT_CSV", "/data/predicciones.csv"))
    p.add_argument("--umbral-advertencia", type=float,
                   default=float(os.environ.get("UMBRAL_ADVERTENCIA", UMBRAL_ADVERTENCIA)),
                   help="fracción de variables faltantes desde la cual se predice con advertencia")
    p.add_argument("--umbral-maximo", type=float,
                   default=float(os.environ.get("UMBRAL_MAXIMO", UMBRAL_MAXIMO)),
                   help="fracción de variables faltantes desde la cual NO se predice")
    return p.parse_args()


def cargar_artefactos():
    from tensorflow import keras  # import tardío: es lo más lento de cargar
    return {
        "config": json.load(open(ART / "config.json", encoding="utf-8")),
        "coords": pd.read_csv(ART / "coords.csv"),
        "imputer_mediana": joblib.load(ART / "imputer_mediana.pkl"),
        "scaler_knn": joblib.load(ART / "scaler_knn.pkl"),
        "knn_imputer": joblib.load(ART / "knn_imputer.pkl"),
        "scaler_nn": joblib.load(ART / "scaler_nn.pkl"),
        "red": keras.models.load_model(ART / "red.keras"),
    }


def evaluar_faltantes(df, umbral_adv, umbral_max):
    """Devuelve (fracción de faltantes, estado, aviso) para cada fila del CSV original."""
    frac = df[PREDICTORAS].isna().mean(axis=1)
    estado = np.where(frac >= umbral_max, "SIN_PREDICCION",
                      np.where(frac >= umbral_adv, "ADVERTENCIA", "OK"))
    n_faltan = df[PREDICTORAS].isna().sum(axis=1)
    aviso = np.full(len(df), "", dtype=object)
    for i in np.where(estado == "SIN_PREDICCION")[0]:
        aviso[i] = (f"No se pudo predecir: faltan {n_faltan.iloc[i]} de {len(PREDICTORAS)} variables "
                    f"({frac.iloc[i]:.0%}) y no hay información suficiente.")
    for i in np.where(estado == "ADVERTENCIA")[0]:
        aviso[i] = (f"Predicción poco confiable: faltan {n_faltan.iloc[i]} de {len(PREDICTORAS)} "
                    f"variables ({frac.iloc[i]:.0%}); los datos faltantes se estimaron y el resultado "
                    f"puede ser erróneo.")
    return frac, estado, aviso


def preprocesar(df, art):
    """Repite la limpieza del notebook sobre las filas de df y devuelve la matriz lista para la red."""
    cfg = art["config"]
    X = df.copy()
    avisos_locacion = pd.Series("", index=X.index, dtype=object)

    # 1) Coordenadas en lugar de Location
    X = X.merge(art["coords"], on="Location", how="left").set_index(df.index)
    desconocida = X["Latitud"].isna()
    avisos_locacion[desconocida] = "Location desconocida: se estimaron Latitud y Longitud."
    X = X.drop(columns="Location")

    # 2) Fecha -> año, día y mes cíclico (enero = 0°, ..., diciembre = 330°)
    fecha = pd.to_datetime(X["Date"], errors="coerce")
    X["Year"] = fecha.dt.year
    X["Day"] = fecha.dt.day
    grados_mes = (fecha.dt.month - 1) * 30.0
    X["Month_sin"] = np.sin(np.deg2rad(grados_mes)).round(4)
    X["Month_cos"] = np.cos(np.deg2rad(grados_mes)).round(4)
    X = X.drop(columns="Date")

    # 3) Dirección del viento -> seno y coseno (NaN se propaga y se imputa más adelante)
    for col in COLUMNAS_VIENTO:
        grados = X[col].map(GRADOS_VIENTO)
        X[f"{col}_sin"] = np.sin(np.deg2rad(grados)).round(4)
        X[f"{col}_cos"] = np.cos(np.deg2rad(grados)).round(4)
    X = X.drop(columns=COLUMNAS_VIENTO)

    # 4) RainToday: si falta se deduce de Rainfall (>1 -> Yes). Si faltan ambos queda NaN y se imputa.
    rt = X["RainToday"].astype(object)  # object: si la columna viene vacía pandas la lee como float
    falta = rt.isna() & X["Rainfall"].notna()
    rt[falta] = np.where(X.loc[falta, "Rainfall"] > 1, "Yes", "No")
    X["RainToday_Yes"] = np.where(rt.isna(), np.nan, (rt == "Yes").astype(float))
    X = X.drop(columns="RainToday")

    # 5) Imputación: mediana para 4 columnas, KNN (sobre datos estandarizados) para el resto
    cols_med, cols_knn = cfg["cols_mediana"], cfg["cols_knn"]
    X[cols_med] = art["imputer_mediana"].transform(X[cols_med])
    knn = art["knn_imputer"].transform(art["scaler_knn"].transform(X[cols_knn]))
    X[cols_knn] = art["scaler_knn"].inverse_transform(knn)

    # 6) Multicolinealidad: mismas columnas eliminadas, mismo orden, y escalado de la red
    X = X.drop(columns=cfg["cols_vif_eliminar"])[cfg["columnas_finales"]]
    return art["scaler_nn"].transform(X), avisos_locacion


def predecir(X, art):
    """Probabilidad de lluvia (clase 'Yes') para cada fila de X."""
    salida = art["red"].predict(X, verbose=0)
    return salida[:, 1] if salida.ndim == 2 and salida.shape[1] == 2 else np.ravel(salida)


def main():
    args = parsear_argumentos()
    if not 0 <= args.umbral_advertencia <= args.umbral_maximo <= 1:
        raise SystemExit("Se requiere 0 <= umbral-advertencia <= umbral-maximo <= 1.")

    df = pd.read_csv(args.input)
    df = df.drop(columns=["Unnamed: 0", "RainTomorrow", "RainfallTomorrow"], errors="ignore")
    faltantes = [c for c in ["Date", "Location"] + PREDICTORAS if c not in df.columns]
    if faltantes:
        print(f"AVISO: faltan columnas en el CSV ({', '.join(faltantes)}); se consideran sin datos.")
        for c in faltantes:
            df[c] = np.nan
    df = df.reset_index(drop=True)

    frac, estado, aviso = evaluar_faltantes(df, args.umbral_advertencia, args.umbral_maximo)
    art = cargar_artefactos()
    umbral = art["config"]["umbral"]

    prob = np.full(len(df), np.nan)
    aptas = estado != "SIN_PREDICCION"
    if aptas.any():
        X, aviso_loc = preprocesar(df[aptas], art)
        prob[aptas] = predecir(X, art)
        for i, texto in zip(np.where(aptas)[0], aviso_loc):
            if texto:
                aviso[i] = (aviso[i] + " " + texto).strip()
                if estado[i] == "OK":
                    estado[i] = "ADVERTENCIA"

    pred = np.where(np.isnan(prob), "NO_PREDICHO", np.where(prob >= umbral, "Yes", "No"))
    salida = pd.DataFrame({
        "Date": df["Date"], "Location": df["Location"],
        "prob_lluvia": np.round(prob, 4), "RainTomorrow_pred": pred,
        "estado": estado, "pct_variables_faltantes": (frac * 100).round(1), "aviso": aviso})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    salida.to_csv(args.output, index=False)

    # Resumen por consola
    print(f"Filas procesadas: {len(df)} | umbral de decisión: {umbral:.2f}")
    print(salida["estado"].value_counts().to_string())
    for _, f in salida[salida["aviso"] != ""].iterrows():
        print(f"[{f['estado']}] {f['Date']} {f['Location']}: {f['aviso']}")
    print(f"Predicciones guardadas en {args.output}")


if __name__ == "__main__":
    main()
