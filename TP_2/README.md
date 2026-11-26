## Aprendizaje Automático 1
## Facultad de Ciencias Exactas, Ingeniería y Agrimensura

## Trabajo Práctico 2

## Predicción de lluvia en Australia (clasificación)

**Integrantes:** Aldana Desiré Sánchez · Lisandro Odisio Martinelli · Marisa Silvina Zappalá

## Objetivo
Predecir si al día siguiente va a llover (`RainTomorrow`) a partir de las observaciones meteorológicas diarias de estaciones de Australia. El objetivo final del trabajo es construir un ejecutable de Docker que haga esa predicción a las 23:59 hs de cada día, comparando antes distintos modelos de clasificación (regresión logística, modelo base, redes neuronales, AutoML). El enunciado completo está en [2026C2 - TrabajoPrácticoClasificacionAA1.pdf](2026C2%20-%20TrabajoPr%C3%A1cticoClasificacionAA1.pdf).

## Contenido del repositorio

| Ruta | Descripción |
|---|---|
| [TP-clasificacion-AA1.ipynb](TP-clasificacion-AA1.ipynb) | Notebook con todo el trabajo: análisis descriptivo, limpieza, preprocesamiento, regresión logística, modelo base, hiperparámetros, SHAP, PyCaret, red neuronal, comparación de modelos y conclusión |
| [weatherAUS_2026C2.csv](weatherAUS_2026C2.csv) | Dataset (145.412 filas, 25 columnas) |
| [requirements.txt](requirements.txt) | Dependencias de Python |
| [2026C2 - TrabajoPrácticoClasificacionAA1.pdf](2026C2%20-%20TrabajoPr%C3%A1cticoClasificacionAA1.pdf) | Enunciado del TP |
| [docker/](docker/) | Ítem 9 (MLOps): `inferencia.py`, `requirements.txt`, `artefactos/` (red, imputers y scalers), `Dockerfile` y `readme.md` con las instrucciones de `docker build` y `docker run` |
| [prueba/](prueba/) | Prueba de punta a punta del container con el conjunto de test (`generar_csvs.py`, `comparar.py` y CSV de entrada/salida) |

## Dataset

Observaciones diarias de **49 estaciones** entre el **01/11/2007 y el 24/06/2017**. Cada fila es un día en una ubicación y reúne temperatura, lluvia, evaporación, horas de sol, viento, humedad, presión y nubosidad. La variable objetivo es `RainTomorrow` (`Yes`/`No`).

- Está **desbalanceado**: 75,8 % `No`, 21,9 % `Yes` y 2,2 % sin dato. Un modelo que siempre responda "No llueve" acierta ~78 %, por eso no alcanza con la accuracy y se usan precision, recall, F1, balanced accuracy, ROC-AUC y matriz de confusión.
- Faltantes: ~10 % en `WindGustDir`, `WindGustSpeed`, `WindDir9am`, `Pressure9am` y `Pressure3pm`; entre 38 % y 50 % en `Evaporation`, `Sunshine`, `Cloud9am` y `Cloud3pm`; menos de 5 % en el resto.
- `RainfallTomorrow` se descarta de los predictores porque informa el futuro (*data leakage*).

## Lo realizado

### 1. Análisis descriptivo
- Tamaño, estadísticas, tipos de datos y porcentaje de nulos por columna.
- Tabla con el significado, unidad, tipo y rango observado de cada variable.
- Gráficos: boxplots e histogramas de las numéricas continuas, barras de `Cloud9am`/`Cloud3pm` (discretas), barras de las categóricas, correlaciones de Pearson y Spearman, gráfico de pares y un **mapa de Australia** con la lluvia por ciudad (coordenadas obtenidas por geocodificación).
- Hallazgos: `Rainfall`, `Evaporation` y `RainfallTomorrow` muy sesgadas a la derecha; `Temp3pm`–`MaxTemp` y `Pressure9am`–`Pressure3pm` correlacionadas por encima de 0,96; la lluvia se concentra en la costa este y sudeste, lo que sugiere que la ubicación aporta información.

### 2. Limpieza y preprocesamiento
1. Se elimina `Unnamed: 0` (índice con valores erróneos).
2. `Location` se reemplaza por `Latitud` y `Longitud`.
3. **Codificación cíclica** (seno/coseno) del mes y de las direcciones del viento (`WindGustDir`, `WindDir9am`, `WindDir3pm`); el año y el día se mantienen numéricos.
4. `RainToday` con One Hot Encoding.
5. Se eliminan las filas sin `RainTomorrow` (2,24 %).
6. **Split estratificado 70 / 15 / 15** (train / validación / test, `random_state=42`) hecho *antes* de imputar y escalar para evitar data leakage.
7. Se eliminan en train las filas con más nulos hasta completar un 5 % de filas descartadas sobre el total.
8. **Imputación**: mediana (`SimpleImputer`) para `Evaporation`, `Sunshine`, `Cloud9am` y `Cloud3pm` (38–50 % de nulos, donde KNN no aporta información y es muy costoso); `KNNImputer` sobre datos estandarizados para el resto, ajustado con una muestra de 10.000 filas de train por costo computacional. Todo se ajusta solo con train.
9. **Multicolinealidad (VIF)**: se eliminan iterativamente variables con VIF > 10 (`Temp3pm`, `Pressure9am`, `Temp9am`). Quedan 26 predictoras, todas con VIF < 10.

### 3. Regresión logística
- `Pipeline` con `StandardScaler` + `LogisticRegression`.
- Métricas: accuracy, precision, recall, F1, balanced accuracy y ROC-AUC; matrices de confusión (absoluta y normalizada) y análisis de falsos positivos y falsos negativos.
- Curva ROC y búsqueda del **umbral que maximiza el F1** en validación: 0,33 (el 0,5 por defecto no es el mejor).
- Resultados con ese umbral (test usado una única vez):

| Conjunto | Umbral | F1 | Precision | Recall | ROC-AUC |
|---|---|---|---|---|---|
| Validación | 0,33 | 0,647 | 0,626 | 0,671 | 0,876 |
| Test | 0,33 | 0,635 | 0,613 | 0,658 | 0,870 |

El F1 en validación y en test son parecidos y las diferencias entre train y validación son menores a 0,015, por lo que el modelo no sobreajusta.

### 4. Modelo base
`DummyClassifier(strategy="most_frequent")`: siempre predice "No". Obtiene accuracy 0,776 pero precision, recall y F1 iguales a 0 y balanced accuracy de 0,5; su curva ROC coincide con la diagonal. La regresión logística aporta valor claro frente a esta línea base.

### 5. Optimización de hiperparámetros (regresión logística)
Grid search con validación cruzada (k-folds) sobre `C`, `penalty`/`l1_ratio` y `class_weight`. **No mejoró el modelo**: con su umbral reoptimizado (0,64) el F1 de validación es 0,6465 contra 0,6473 del modelo anterior (empate). `class_weight='balanced'` y bajar el umbral hacen lo mismo: correr el punto de corte entre precision y recall. La regresión logística llegó a su techo (F1 ≈ 0,65).

### 6. Explicabilidad con SHAP
Gráficos locales (waterfall y force plot) y globales (beeswarm y barras). La variable más influyente es `Humidity3pm` (relación directa), seguida de `WindGustSpeed`, `Pressure3pm` y `Sunshine`; `Cloud9am` y `WindGustDir_sin` aportan casi nada.

### 7. AutoML con PyCaret
PyCaret eligió **Random Forest** como mejor modelo por F1 de la clase `Yes`. Su tuneo automático lo empeoró (más recall pero muchas más falsas alarmas y peor AUC), así que se conserva el modelo sin tunear.

### 8. Red neuronal
- Clase `NeuralNetwork` (Keras) entrenada con los datos preprocesados.
- Optimización de hiperparámetros con **random search** (Optuna, 10 pruebas) por F1 de `Yes`, reentrenamiento de la mejor configuración, búsqueda de umbral (0,54) y **validación cruzada de 5 folds**: F1 = 0,667 ± 0,006 y ROC-AUC = 0,894 ± 0,004.
- SHAP sobre la red (`GradientExplainer`), con el mismo ranking de variables que en la regresión.

### 9. Comparación de modelos y elección
Todos los modelos se comparan con el F1 de `Yes`, cada uno con su propio umbral elegido en validación (test solo confirma):

| Modelo | Umbral | F1 val | F1 test | Precision test | Recall test | ROC-AUC test |
|---|---|---|---|---|---|---|
| Random Forest (PyCaret) | 0,34 | 0,6773 | 0,6731 | 0,6361 | 0,7147 | 0,8913 |
| **Red neuronal optimizada** | 0,54 | 0,6766 | 0,6710 | 0,6452 | 0,6989 | 0,8941 |
| Red neuronal inicial | 0,54 | 0,6643 | 0,6592 | 0,6060 | 0,7226 | 0,8857 |
| Regresión logística tuneada | 0,64 | 0,6465 | 0,6370 | 0,6187 | 0,6564 | 0,8706 |
| Regresión logística | 0,33 | 0,6473 | 0,6350 | 0,6133 | 0,6583 | 0,8702 |
| Base (clase más frecuente) | – | 0,0000 | 0,0000 | 0,0000 | 0,0000 | 0,5000 |

Random Forest y red neuronal empatan en F1; se elige la **red neuronal optimizada** por ocupar ~5.000 veces menos memoria (~0,02 MB contra ~103 MB) y tener un ROC-AUC levemente mayor. Random Forest predice más rápido, pero la diferencia es de milisegundos por día.

### 10. Docker (MLOps) y conclusión
- Los artefactos (red, imputers, scalers, coordenadas y configuración) se exportan a [docker/](docker/); `inferencia.py` repite la limpieza del notebook y predice. Las instrucciones de `docker build` y `docker run` están en [docker/readme.md](docker/readme.md).
- Prueba de punta a punta en [prueba/](prueba/) con las 21.113 filas de test sin la variable objetivo: el container predijo 20.842 (en 271 faltaba más del 50 % de las variables y devuelve `SIN_PREDICCION`) y obtuvo F1 = 0,6726 y ROC-AUC = 0,8953, contra 0,6710 y 0,8941 del notebook.
- La conclusión del notebook resume hallazgos y limitaciones: el umbral importa más que los hiperparámetros, aún se pierden ~3 de cada 10 lluvias y el modelo no usa información de días anteriores.

## Cómo ejecutarlo

El notebook se ejecutó con **Python 3.11** (`pycaret==3.3.2` no es compatible con versiones más nuevas), así que crear el entorno con esa versión.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (en Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
jupyter notebook TP-clasificacion-AA1.ipynb
```
Recomendacion: reiniciar kernel si se quiere ejecutar el notebook en la misma sesion.
Para ello, presionar Ctrl + Shift + P, y elegir en entorno virtual correspondiente.

Notas:
- El notebook lee `weatherAUS_2026C2.csv` desde esta carpeta.
- `requirements.txt` incluye todas las librerías que importa el notebook. Son las del entrenamiento; las de inferencia están en `docker/requirements.txt`, que es más corto.
- Las coordenadas de las estaciones (listado oficial del BOM) y el mapa descargan datos de internet.
- La imputación con KNN puede tardar algunos minutos.
