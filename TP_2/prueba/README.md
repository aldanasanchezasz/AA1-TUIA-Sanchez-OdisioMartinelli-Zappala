# Prueba de punta a punta del container

Verifica que el container (`../docker`) reproduce el desempeño del notebook: se le pasa el conjunto de **test sin la variable a predecir** y se compara lo que predice con los valores reales.

## Contenido

| Archivo | Descripción |
|---|---|
| `generar_csvs.py` | Reproduce el split del notebook (70/15/15, `random_state=42`, estratificado) y genera los dos CSV |
| `test_completo.csv` | Filas de test con **todas** las columnas, incluida `RainTomorrow` (21.113 filas) |
| `datos/input.csv` | Las mismas filas, sin `RainTomorrow` ni `RainfallTomorrow`: es la entrada del container |
| `datos/predicciones.csv` | Salida del container (se genera al correrlo) |
| `comparar.py` | Compara `predicciones.csv` con `test_completo.csv` y calcula las métricas |
| `comparacion.csv` | Detalle fila por fila: valor real, probabilidad y predicción (se genera al comparar) |

## Pasos

Desde esta carpeta (`prueba/`):

```bash
# 1) (Opcional) regenerar los CSV de test. Requiere ../weatherAUS_2026C2.csv y scikit-learn
python generar_csvs.py

# 2) Construir la imagen (una sola vez, desde ../docker)
docker build -t lluvia-aus ../docker

# 3) Predecir con el container
docker run --rm -v "$(pwd)/datos:/data" lluvia-aus          # Linux / macOS / Git Bash
docker run --rm -v "${PWD}\datos:/data" lluvia-aus          # Windows PowerShell

# 4) Comparar con los valores reales
python comparar.py
```

> En Git Bash de Windows, anteponer `MSYS_NO_PATHCONV=1` al `docker run` para que no modifique la ruta del volumen.

## Resultado obtenido

El container predijo 20.842 de las 21.113 filas. En las otras 271 faltaba más del 50 % de las variables y devolvió `SIN_PREDICCION`, tal como está diseñado. Otras 1.883 filas se predijeron con advertencia (faltaba entre 30 % y 50 %).

| Métrica (clase `Yes`) | Container (filas predichas) | Notebook (test completo) |
|---|---|---|
| Precision | 0,6455 | 0,6452 |
| Recall | 0,7021 | 0,6989 |
| F1 | 0,6726 | 0,6710 |
| ROC-AUC | 0,8953 | 0,8941 |
| Accuracy | 0,8484 | – |

La diferencia mínima (≈ 0,002) se explica porque el container no predice las 271 filas con demasiados faltantes, que el notebook sí evalúa. Eso confirma que la limpieza, los imputers, los scalers y el umbral dentro del container reproducen el pipeline del notebook.
