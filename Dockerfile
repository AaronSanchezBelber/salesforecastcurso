# Usamos la imagen oficial de Apache Airflow versión 2.9.3 como base
FROM apache/airflow:2.9.3

# Copiamos el archivo requirements.txt al contenedor
COPY requirements.txt /requirements.txt

# Instalamos las dependencias de Python listadas en requirements.txt
# --no-cache-dir evita guardar caché y reduce el tamaño de la imagen
RUN pip install --no-cache-dir -r /requirements.txt

