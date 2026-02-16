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


# Para dividir los datos en entrenamiento y prueba
from sklearn.model_selection import train_test_split

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
    training_file_path: str
    testing_file_path: str
    train_test_split_ratio: float = 0.2



# Define el artefacto que se devolverá tras la ingestión
@dataclass
class DataIngestionArtifact:
    trained_file_path: str
    test_file_path: str


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
        
    # Divide los datos en entrenamiento y prueba
    def split_data_as_train_test(self, dataframe: pd.DataFrame) -> None:
        try:
            # Realiza la división train/test
            train_set, test_set = train_test_split(
                dataframe, test_size=self.data_ingestion_config.train_test_split_ratio
            )

            # Loggea la operación
            logging.info("Performed train test split on the dataframe")

            # Obtiene directorios de salida
            train_dir = os.path.dirname(self.data_ingestion_config.training_file_path)
            test_dir = os.path.dirname(self.data_ingestion_config.testing_file_path)

            # Crea los directorios si no existen
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(test_dir, exist_ok=True)

            # Guarda los archivos CSV
            train_set.to_csv(
                self.data_ingestion_config.training_file_path, index=False, header=True
            )
            test_set.to_csv(
                self.data_ingestion_config.testing_file_path, index=False, header=True
            )

            # Loggea la exportación
            logging.info("Exported train and test file path.")

        except Exception as e:
            raise SalesForecastException(e, sys)


    # Orquesta todo el proceso de ingestión
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            # Exporta datos desde MongoDB
            dataframe = self.export_collection_as_dataframe()

            # Guarda datos en el feature store
            dataframe = self.export_data_into_feature_store(dataframe)

            # Divide en train/test
            self.split_data_as_train_test(dataframe=dataframe)

            # Crea el artefacto de salida
            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path,
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
        training_file_path='artifacts/train/train.csv',
        testing_file_path='artifacts/test/test.csv',
        train_test_split_ratio=0.2,)
    
    ingestion = DataIngestion(config)
    artifact = ingestion.initiate_data_ingestion()

    print("Data Ingestion completed")
    print(artifact)













