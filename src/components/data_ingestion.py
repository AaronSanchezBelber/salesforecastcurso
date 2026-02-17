"""Data ingestion component."""

# Importa módulos del sistema y utilidades
import os
import sys
from dataclasses import dataclass

# Importa librerías para manejo de datos
import numpy as np
import pandas as pd
import pymongo

# Carga variables de entorno desde archivo .env
from dotenv import load_dotenv


# Importa la excepción personalizada
from src.exception.exception import SalesForecastException
# Importa el logger configurado
from src.logging.logger import logging

# Carga variables de entorno
load_dotenv()

# Obtiene la URL de MongoDB desde el entorno
MONGO_DB_URL = os.getenv("MONGO_DB_URL")


# Define la configuración necesaria para la ingestión de datos
@dataclass
class DataIngestionConfig:
    database_name: str
    collection_name: str
    feature_store_file_path: str
    training_file_path: str = ""
    testing_file_path: str = ""
    train_test_split_ratio: float = 0.2



# Define el artefacto que se devolverá tras la ingestión
@dataclass
class DataIngestionArtifact:
    feature_store_file_path: str


# Clase principal para la ingestión de datos
class DataIngestion:
    # Constructor que recibe la configuración
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            # Guarda la configuración en la instancia
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            # Lanza excepción personalizada si algo falla
            raise SalesForecastException(e, sys)
        
    # Exporta una colección de MongoDB como un DataFrame
    def export_collection_as_dataframe(self) -> pd.DataFrame:
        try:
            # Obtiene nombre de base de datos y colección
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name

            # Verifica que la URL de Mongo esté definida
            if not MONGO_DB_URL:
                raise ValueError("MONGO_DB_URL no esta definido en el entorno.")

            # Conecta a MongoDB
            mongo_client = pymongo.MongoClient(MONGO_DB_URL)

            # Accede a la colección
            collection = mongo_client[database_name][collection_name]

            # Convierte los documentos en un DataFrame
            df = pd.DataFrame(list(collection.find()))

            # Elimina la columna _id si existe
            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"])

            # Reemplaza valores "na" por NaN
            df.replace({"na": np.nan}, inplace=True)

            # Devuelve el DataFrame final
            return df

        except Exception as e:
            # Manejo de errores con excepción personalizada
            raise SalesForecastException(e, sys)

    # Guarda el DataFrame en el feature store
    def export_data_into_feature_store(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        try:
            # Obtiene la ruta del archivo de feature store
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path

            # Crea el directorio si no existe
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)

            # Guarda el DataFrame como CSV
            dataframe.to_csv(feature_store_file_path, index=False, header=True)

            # Devuelve el DataFrame
            return dataframe

        except Exception as e:
            raise SalesForecastException(e, sys)
        
    # Orquesta todo el proceso de ingestión
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            if os.path.exists(feature_store_file_path):
                dataframe = pd.read_csv(feature_store_file_path)
                logging.info("Using existing feature store file: %s", feature_store_file_path)
            else:
                dataframe = self.export_collection_as_dataframe()
                dataframe = self.export_data_into_feature_store(dataframe)
                logging.info("Feature store created from Mongo: %s", feature_store_file_path)

            # Crea el artefacto de salida
            data_ingestion_artifact = DataIngestionArtifact(
                feature_store_file_path=feature_store_file_path,
            )

            # Devuelve el artefacto
            return data_ingestion_artifact
        
        except Exception as e:
            raise SalesForecastException(e, sys)


# Punto de entrada del script
if __name__ == "__main__": 
    config = DataIngestionConfig(
        database_name='SalesForecastCurso',
        collection_name='forecastCurso',
        feature_store_file_path='artifacts/feature_store/forecast.csv',
    )
    
    ingestion = DataIngestion(config)
    artifact = ingestion.initiate_data_ingestion()

    print("Data Ingestion completed")
    print(artifact)













