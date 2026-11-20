# Misma versión de Python con la que se generaron los artefactos (.pkl y .keras)
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY inferencia.py .
COPY artefactos/ artefactos/

# El CSV de entrada y el de salida se intercambian por la carpeta montada en /data
ENTRYPOINT ["python", "inferencia.py"]
