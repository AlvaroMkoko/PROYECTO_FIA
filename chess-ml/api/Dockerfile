# Imagen base — Python slim para mantener el contenedor ligero
FROM python:3.11-slim

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar e instalar dependencias PRIMERO
# Esto aprovecha el cache de Docker: si requirements.txt no cambia,
# Docker no reinstala nada aunque cambies main.py
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código y los artefactos
COPY main.py .
COPY monitoring.py .
COPY artifacts/ artifacts/

# Crear carpeta de logs dentro del contenedor
RUN mkdir -p logs reports

# Puerto que expone el contenedor
EXPOSE 8000

# Comando de inicio
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]