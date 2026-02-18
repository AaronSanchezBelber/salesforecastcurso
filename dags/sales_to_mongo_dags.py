# Importamos módulos estándar de Python
import os
import json
import hashlib
from datetime import datetime

# Importamos pandas para manipulación de datos
import pandas as pd

# Importamos el cliente de MongoDB
from pymongo import MongoClient

# Importamos clases de Airflow
from airflow import DAG
from airflow.decorators import task


# VARIABLES DE ENTORNO
# URL de conexión a MongoDB
MONGO_DB_URL = os.getenv("MONGO_DB_URL")
# Nombre de la base de datos (valor por defecto si no existe en el entorno)
DB_NAME = os.getenv("MONGO_DB", "SalesForecastCurso")
# Nombre de la colección
COLLECTION_NAME = os.getenv("MONGO_COLLECTION", "forecastCurso")
# Ruta del CSV principal
CSV_PATH = os.getenv(
    "CSV_PATH", "/opt/airflow/data/forecast.csv"
)
# Ruta del CSV que contiene lotes de 100 filas
BATCH_CSV_PATH = os.getenv(
    "BATCH_CSV_PATH", "/opt/airflow/data/sales_train_merged_batch_100.csv"
)
# Ruta del archivo de estado para el conteo de filas procesadas
STATE_PATH = os.getenv(
    "STATE_PATH", "/opt/airflow/state/row_count.json"
)
# Ruta del archivo de estado para el batch incremental
BATCH_STATE_PATH = os.getenv(
    "BATCH_STATE_PATH", "/opt/airflow/state/batch_state.json"
)


# FUNCIONES AUXILIARES

def _read_records(path, start=None, end=None):
    """
    Lee un CSV y devuelve los registros como lista de diccionarios.
    Permite leer solo un rango de filas usando start y end.
    """
    # Leemos el archivo CSV
    df = pd.read_csv(path)

    # Si se especifica un rango, seleccionamos solo esas filas
    if start is not None and end is not None:
        df = df.iloc[start:end]

    # Eliminamos columna de índice autogenerada si existe
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Convertimos la columna 'date' a datetime si existe
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Reseteamos el índice
    df.reset_index(drop=True, inplace=True)

    # Convertimos el DataFrame a una lista de diccionarios
    return df.to_dict(orient="records")

def _insert_records(records):
    """
    Inserta una lista de registros en MongoDB
    """
    # Si no hay registros, no insertamos nada
    if not records:
        return 0

    # Validamos que la URL de MongoDB esté definida
    if not MONGO_DB_URL:
        raise ValueError("MONGO_DB_URL is not set")

    # Creamos el cliente de MongoDB
    client = MongoClient(MONGO_DB_URL)

    # Seleccionamos la base de datos
    db = client[DB_NAME]

    # Seleccionamos la colección
    col = db[COLLECTION_NAME]

    # Insertamos los documentos
    result = col.insert_many(records)

    # Retornamos la cantidad de documentos insertados
    return len(result.inserted_ids)


def _load_state(path, default):
    """
    Carga el estado desde un archivo JSON.
    Si no existe, retorna un valor por defecto.
    """
    if not os.path.exists(path):
        return default

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def _save_state(path, payload):
    """
    Guarda el estado de forma atómica usando un archivo temporal
    """
    # Creamos el directorio si no existe
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Guardamos primero en un archivo temporal
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    # Reemplazamos el archivo original de forma atómica
    os.replace(tmp_path, path)

def _file_hash(path):
    """
    Calcula el hash SHA256 de un archivo para detectar cambios
    """
    h = hashlib.sha256()

    with open(path, "rb") as f:
        # Leemos el archivo en bloques de 1MB
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()



# DAG 1: BATCH SEMANAL

with DAG(
    dag_id="sales_batch_weekly",
    start_date=datetime(2026, 2, 1),
    schedule="@weekly",
    catchup=False,
    is_paused_upon_creation=True,
    tags=["sales", "mongo"],
):

    @task
    def push_weekly_batch():
        # Leemos todos los registros del CSV batch
        records = _read_records(BATCH_CSV_PATH)

        # Insertamos los registros en MongoDB
        return _insert_records(records)

    # Ejecutamos la tarea
    push_weekly_batch()



# DAG 2: BATCH INCREMENTAL POR CAMBIO DE ARCHIVO

with DAG(
    dag_id="sales_incremental_on_100",
    start_date=datetime(2026, 2, 1),
    schedule=None,
    catchup=False,
    tags=["sales", "mongo"],
):
    
    @task
    def push_batch_if_new():
        # Calculamos el hash actual del archivo batch
        current_hash = _file_hash(BATCH_CSV_PATH)

        # Cargamos el estado anterior
        state = _load_state(BATCH_STATE_PATH, {"hash": None})

        # Si el archivo no cambió, no insertamos nada
        if state.get("hash") == current_hash:
            return {"inserted": 0, "reason": "batch unchanged"}

        # Leemos e insertamos los registros
        records = _read_records(BATCH_CSV_PATH)
        inserted = _insert_records(records)

        # Guardamos el nuevo estado
        _save_state(
            BATCH_STATE_PATH,
            {"hash": current_hash, "rows": len(records)}
        )

        return {"inserted": inserted}

    # Ejecutamos la tarea
    push_batch_if_new()

# DAG 3: INCREMENTAL CADA 100 FILAS (CSV PRINCIPAL)


with DAG(
    dag_id="sales_incremental_on_100_main",
    start_date=datetime(2026, 2, 1),
    schedule="*/10 * * * *",
    catchup=False,
    tags=["sales", "mongo"],
):
    
    @task
    def compute_window():
        # Calculamos el total de filas del CSV
        total = len(pd.read_csv(CSV_PATH))

        # Cargamos el estado del último conteo
        state = _load_state(STATE_PATH, {"last_count": 0})
        last_count = int(state.get("last_count", 0))

        # Calculamos cuántas filas nuevas hay
        delta = total - last_count

        # Si hay menos de 100 filas nuevas, no insertamos
        if delta < 100:
            return {
                "do_insert": False,
                "total": total,
                "last_count": last_count,
                "delta": delta,
            }

        # Retornamos la ventana de filas a insertar
        return {
            "do_insert": True,
            "start": last_count,
            "end": total,
            "total": total,
            "delta": delta,
        }
    
    @task
    def push_new_rows(window):
        # Si no hay que insertar, salimos
        if not window.get("do_insert"):
            return {"inserted": 0, "delta": window.get("delta")}

        # Leemos solo las nuevas filas
        records = _read_records(
            CSV_PATH,
            start=window["start"],
            end=window["end"]
        )

        # Insertamos los nuevos registros
        inserted = _insert_records(records)

        # Actualizamos el estado
        _save_state(
            STATE_PATH,
            {"last_count": window["end"]}
        )

        return {
            "inserted": inserted,
            "new_last_count": window["end"]
        }

    # Definimos el flujo de tareas
    window = compute_window()
    push_new_rows(window)
