# Inferencia: ¿lloverá mañana? (`RainTomorrow`)

Container que toma un CSV con observaciones meteorológicas diarias, repite la limpieza del notebook (`TP-clasificacion-AA1.ipynb`) y predice con la **red neuronal optimizada** elegida como modelo final, usando su umbral de decisión (0,54).

## Contenido de la carpeta

| Archivo | Descripción |
|---|---|
| `inferencia.py` | Script que se ejecuta al correr el container |
| `requirements.txt` | Librerías necesarias solo para inferir |
| `Dockerfile` | Definición de la imagen |
| `artefactos/` | Red (`red.keras`), imputers y scalers ajustados con train (`.pkl`), `coords.csv` (latitud y longitud de cada estación) y `config.json` (columnas, variables eliminadas por VIF y umbral) |

## Construir la imagen

Desde esta carpeta (`docker/`):

```bash
docker build -t lluvia-aus .
```

## Ejecutar el container

Crear una carpeta `datos` con el CSV de entrada llamado `input.csv` y montarla en `/data`:

```bash
# Linux / macOS / Git Bash
docker run --rm -v "$(pwd)/datos:/data" lluvia-aus

# Windows PowerShell
docker run --rm -v "${PWD}\datos:/data" lluvia-aus

# Windows cmd
docker run --rm -v "%cd%\datos:/data" lluvia-aus
```

Al terminar, las predicciones quedan en `datos/predicciones.csv` y se muestra un resumen por consola.

Opciones (argumentos al final del comando o variables de entorno con `-e`):

| Argumento | Variable | Por defecto | Descripción |
|---|---|---|---|
| `--input` | `INPUT_CSV` | `/data/input.csv` | CSV de entrada |
| `--output` | `OUTPUT_CSV` | `/data/predicciones.csv` | CSV de salida |
| `--umbral-advertencia` | `UMBRAL_ADVERTENCIA` | `0.30` | Fracción de variables faltantes desde la cual se predice con advertencia |
| `--umbral-maximo` | `UMBRAL_MAXIMO` | `0.50` | Fracción de variables faltantes desde la cual no se predice |

Ejemplo: `docker run --rm -v "$(pwd)/datos:/data" lluvia-aus --umbral-maximo 0.6`

## Formato de entrada

CSV con las mismas columnas que `weatherAUS_2026C2.csv`; una fila por día y estación:

`Date`, `Location`, `MinTemp`, `MaxTemp`, `Rainfall`, `Evaporation`, `Sunshine`, `WindGustDir`, `WindGustSpeed`, `WindDir9am`, `WindDir3pm`, `WindSpeed9am`, `WindSpeed3pm`, `Humidity9am`, `Humidity3pm`, `Pressure9am`, `Pressure3pm`, `Cloud9am`, `Cloud3pm`, `Temp9am`, `Temp3pm`, `RainToday`

- `Date` en formato `AAAA-MM-DD` y `Location` con los nombres del dataset original (por ejemplo `Sydney`, `MelbourneAirport`).
- Los valores faltantes pueden dejarse vacíos.
- `Unnamed: 0`, `RainTomorrow` y `RainfallTomorrow` se ignoran si vienen.

## Formato de salida

`predicciones.csv`, una fila por fila de entrada:

| Columna | Descripción |
|---|---|
| `Date`, `Location` | Identifican la fila |
| `prob_lluvia` | Probabilidad estimada de lluvia mañana (vacía si no se predijo) |
| `RainTomorrow_pred` | `Yes` si `prob_lluvia` ≥ 0,54, `No` en caso contrario, `NO_PREDICHO` si no se pudo predecir |
| `estado` | `OK`, `ADVERTENCIA` o `SIN_PREDICCION` (ver abajo) |
| `pct_variables_faltantes` | % de las 20 variables predictoras que llegaron vacías |
| `aviso` | Texto explicativo cuando el estado no es `OK` |

## Filas con muchos datos faltantes

El estado de cada fila depende de cuántas de las 20 variables predictoras llegan vacías:

| Estado | Condición | Qué hace |
|---|---|---|
| `OK` | menos de 30 % | Predice normalmente |
| `ADVERTENCIA` | de 30 % a menos de 50 % | Predice, pero el aviso aclara que los datos faltantes se estimaron y el resultado puede ser erróneo |
| `SIN_PREDICCION` | 50 % o más | No predice; el aviso indica que falta demasiada información |

En el dataset original, el 4,5 % de las filas supera el 30 % de faltantes y el 1,9 % supera el 50 %. Una `Location` que no figura en el dataset original también genera `ADVERTENCIA`, porque su latitud y longitud se estiman.

## Qué hace el script

1. Reemplaza `Location` por `Latitud` y `Longitud` (`coords.csv`).
2. Extrae año y día de `Date`; codifica el mes y las direcciones del viento con seno y coseno.
3. Si falta `RainToday`, lo deduce de `Rainfall` (> 1 mm → `Yes`) y lo codifica en `RainToday_Yes`.
4. Imputa con la mediana `Evaporation`, `Sunshine`, `Cloud9am` y `Cloud3pm`, y con KNN el resto.
5. Elimina las variables descartadas por multicolinealidad (VIF), estandariza y predice con la red.

A diferencia del notebook, en inferencia **no se eliminan filas**: cada fila recibe una respuesta o un aviso.

## Versiones

Los binarios se generaron con Python 3.11, `scikit-learn==1.4.2`, `numpy==1.26.4` y `tensorflow==2.21.0`. Las versiones del `requirements.txt` deben respetarse porque los `.pkl` no son portables entre versiones de scikit-learn.
