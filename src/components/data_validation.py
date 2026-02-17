"""Data validation component."""

# Importa módulos del sistema
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict

# Librerías para manejo de datos
import pandas as pd

# Prueba estadística KS para detectar drift
from scipy.stats import ks_2samp


# Lectura y escritura de archivos YAML
import yaml

# Excepción personalizada
from src.exception.exception import SalesForecastException

# Logger configurado
from src.logging.logger import logging


# Ruta del archivo de esquema que define columnas esperadas
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")

# Función para leer un archivo YAML y devolverlo como diccionario
def read_yaml_file(file_path: str) -> Dict[str, Any]:
    try:
        # Abre el archivo YAML en modo lectura
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
        
    except Exception as e:
        # Manejo de errores con excepción personalizada
        raise SalesForecastException(e, sys)

# Función para escribir un diccionario en un archivo YAML
def write_yaml_file(file_path: str, content: Dict[str, Any]) -> None:
    try:
        # Crea el directorio si no existe
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Escribe el contenido en YAML
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(content, f, sort_keys=False)

    except Exception as e:
        raise SalesForecastException(e, sys)

# Artefacto generado por la etapa de ingestión de datos

@dataclass
class DataIngestionArtifact:
    trained_file_path: str
    test_file_path: str

# Configuración necesaria para la validación de datos
@dataclass
class DataValidationConfig:
    valid_train_file_path: str
    valid_test_file_path: str
    drift_report_file_path: str

# Artefacto generado por la validación de datos
@dataclass
class DataValidationArtifact:
    validation_status: bool
    valid_train_file_path: str
    valid_test_file_path: str
    invalid_train_file_path: str | None
    invalid_test_file_path: str | None
    drift_report_file_path: str


# Clase principal para la validación de datos
class DataValidation:
    # Constructor que recibe artefactos y configuración
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig,
    ):
        try:
            # Guarda los artefactos y configuración
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config

            # Carga el esquema desde YAML
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)

        except Exception as e:
            raise SalesForecastException(e, sys)

    # Valida que el número de columnas coincida con el esquema
    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            # Número de columnas esperadas
            number_of_columns = len(self._schema_config["columns"])

            # Loggea información
            logging.info(f"Required number of columns: {number_of_columns}")
            logging.info(f"Data frame has columns: {len(dataframe.columns)}")

            # Compara cantidad de columnas
            return len(dataframe.columns) == number_of_columns
        
        except Exception as e:
            raise SalesForecastException(e, sys)
        
    # Verifica que existan todas las columnas numéricas requeridas
    def is_numerical_column_exist(self, dataframe: pd.DataFrame) -> bool:
        try:
            numerical_columns = self._schema_config["numerical_columns"]
            dataframe_columns = dataframe.columns

            numerical_column_present = True
            missing_numerical_columns = []

            # Revisa columna por columna
            for num_column in numerical_columns:
                if num_column not in dataframe_columns:
                    numerical_column_present = False
                    missing_numerical_columns.append(num_column)

            # Loggea columnas faltantes
            logging.info(f"Missing numerical columns: {missing_numerical_columns}")

            return numerical_column_present
        except Exception as e:
            raise SalesForecastException(e, sys)
        

    # Lee un archivo CSV y lo convierte en DataFrame
    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        
        except Exception as e:
            raise SalesForecastException(e, sys)


    # Detecta drift entre dataset base y dataset actual
    def detect_dataset_drift(
        self,
        base_df: pd.DataFrame,
        current_df: pd.DataFrame,
        threshold: float = 0.05,
    ) -> bool:
        try:
            status = True
            report: Dict[str, Any] = {}

            # Recorre cada columna para comparar distribuciones
            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]

                # Prueba KS
                is_same_dist = ks_2samp(d1, d2)

                # Determina si hay drift
                if is_same_dist.pvalue >= threshold:
                    is_found = False
                else:
                    is_found = True
                    status = False

                # Guarda resultados en el reporte
                report[column] = {
                    "p_value": float(is_same_dist.pvalue),
                    "drift_status": is_found,
                }

            # Guarda el reporte en YAML
            drift_report_file_path = self.data_validation_config.drift_report_file_path
            write_yaml_file(file_path=drift_report_file_path, content=report)

            return status
        except Exception as e:
            raise SalesForecastException(e, sys)

    def write_no_drift_report_same_source(self, dataframe: pd.DataFrame) -> None:
        try:
            report: Dict[str, Any] = {}
            for column in dataframe.columns:
                report[column] = {
                    "p_value": 1.0,
                    "drift_status": False,
                    "note": "same_source_file",
                }
            write_yaml_file(
                file_path=self.data_validation_config.drift_report_file_path,
                content=report,
            )
        except Exception as e:
            raise SalesForecastException(e, sys)



    # Orquesta todo el proceso de validación
    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            # Rutas de los archivos de train y test
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            # Carga los DataFrames
            train_dataframe = DataValidation.read_data(train_file_path)
            test_dataframe = DataValidation.read_data(test_file_path)

            # Valida columnas en train
            status = self.validate_number_of_columns(train_dataframe)
            if not status:
                logging.warning("Train dataframe does not contain all columns.")

            # Valida columnas en test
            status = self.validate_number_of_columns(test_dataframe)
            if not status:
                logging.warning("Test dataframe does not contain all columns.")

            # Valida columnas numericas requeridas
            if not self.is_numerical_column_exist(train_dataframe):
                logging.warning("Train dataframe missing numerical columns from schema.")
                status = False
            if not self.is_numerical_column_exist(test_dataframe):
                logging.warning("Test dataframe missing numerical columns from schema.")
                status = False

            # Detecta drift solo si son datasets distintos
            if os.path.abspath(train_file_path) != os.path.abspath(test_file_path):
                drift_ok = self.detect_dataset_drift(
                    base_df=train_dataframe, current_df=test_dataframe
                )
                status = status and drift_ok
            else:
                logging.info("Se omite drift: train y test apuntan al mismo archivo.")
                self.write_no_drift_report_same_source(train_dataframe)

            # Crea directorio de salida
            dir_path = os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path, exist_ok=True)

            # Guarda datasets validados
            train_dataframe.to_csv(
                self.data_validation_config.valid_train_file_path,
                index=False,
                header=True,
            )
            test_dataframe.to_csv(
                self.data_validation_config.valid_test_file_path,
                index=False,
                header=True,
            )

            # Crea artefacto final
            data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )

            # Loggea artefacto
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            print("Data validation completed")

            return data_validation_artifact

        except Exception as e:
            raise SalesForecastException(e, sys)


# Ejecución independiente para pruebas
if __name__ == "__main__":
    # Validación sobre feature_store (sin split previo)
    ingestion_artifact = DataIngestionArtifact(
        trained_file_path="artifacts/feature_store/forecast.csv",
        test_file_path="artifacts/feature_store/forecast.csv",
    )

    # Configuración de validación
    validation_config = DataValidationConfig(
        valid_train_file_path="artifacts/valid/train.csv",
        valid_test_file_path="artifacts/valid/test.csv",
        drift_report_file_path="artifacts/drift/report.yaml",
    )

    # Ejecuta validación
    validator = DataValidation(ingestion_artifact, validation_config)
    artifact = validator.initiate_data_validation()
    print(artifact)


