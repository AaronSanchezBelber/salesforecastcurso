# Importa el módulo logging para gestionar logs del sistema
import logging

# Importa os para manejar rutas y directorios
import os

# Importa datetime para generar nombres de archivo basados en fecha y hora
from datetime import datetime



# Genera el nombre del archivo de log usando la fecha y hora actual
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Define la ruta del directorio donde se guardarán los logs
logs_dir = os.path.join(os.getcwd(), "logs")

# Crea el directorio de logs si no existe (no lanza error si ya existe)
os.makedirs(logs_dir, exist_ok=True)

# Construye la ruta completa del archivo de log
LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)



# Configura el sistema de logging:
# - filename: ruta del archivo donde se guardarán los logs
# - format: formato del mensaje de log (fecha, línea, nombre, nivel, mensaje)
# - level: nivel mínimo de logs que se registrarán (INFO)
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)



