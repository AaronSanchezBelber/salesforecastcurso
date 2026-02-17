"""End-to-end training pipeline for SalesForecast."""


# Importa módulos del sistema
import os
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict

# Importa YAML para guardar configuraciones y reportes
import yaml

# Importa los componentes del pipeline
from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.data_validation import (
    DataIngestionArtifact as ValidationIngestionArtifact,
    DataValidation,
    DataValidationConfig,
)
from src.components.model_evaluation import DataEvaluation, DataEvaluationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig


# Importa excepciones y logger personalizados
from src.exception.exception import SalesForecastException
from src.logging.logger import logging

# Define la configuración general del pipeline
@dataclass
class PipelineConfig:
    database_name: str = "SalesForecastCurso"  # Nombre de la base de datos
    collection_name: str = "forecastCurso"  # Nombre de la colección
    feature_store_path: str = os.path.join("artifacts", "feature_store", "forecast.csv")  # Ruta del feature store
    drift_report_path: str = os.path.join("artifacts", "drift", "report.yaml")  # Ruta del reporte de drift
    preprocessed_dir: str = os.path.join("artifacts", "preprocessed")  # Carpeta de datos preprocesados
    model_dir: str = os.path.join("artifacts", "model")  # Carpeta donde se guardará el modelo
    model_name: str = "xgb_model.joblib"  # Nombre del archivo del modelo
    evaluation_report_path: str = os.path.join("artifacts", "evaluation", "report.yaml")  # Ruta del reporte de evaluación
    train_ratio: float = 0.7  # Porcentaje de datos para entrenamiento
    valid_ratio: float = 0.15  # Porcentaje de datos para validación
    test_ratio: float = 0.15  # Porcentaje de datos para test

# Clase principal del pipeline
class TrainingPipeline:
    # Constructor que recibe la configuración
    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()

    # Asegura que la salida estándar use UTF-8
    @staticmethod
    def _ensure_utf8_stdout() -> None:
        try:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass  # Si no se puede, continúa sin error

    # Carga segura de YAML (si existe)
    @staticmethod
    def _safe_load_yaml(file_path: str) -> Dict[str, Any]:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    # Método principal que ejecuta todo el pipeline
    def run(self) -> Dict[str, Any]:
        try:
            # Asegura UTF-8
            self._ensure_utf8_stdout()

            # Log de inicio del pipeline
            logging.info("Pipeline started with config: %s", asdict(self.config))

            # -----------------------------
            # 1) INGESTIÓN DE DATOS
            # -----------------------------
            ingestion = DataIngestion(
                DataIngestionConfig(
                    database_name=self.config.database_name,
                    collection_name=self.config.collection_name,
                    feature_store_file_path=self.config.feature_store_path,
                )
            )
            ingestion_artifact = ingestion.initiate_data_ingestion()
            logging.info("Ingestion completed: %s", ingestion_artifact)

            # -----------------------------
            # 2) VALIDACIÓN DE DATOS
            # -----------------------------
            validation = DataValidation(
                data_ingestion_artifact=ValidationIngestionArtifact(
                    trained_file_path=self.config.feature_store_path,
                    test_file_path=self.config.feature_store_path,
                ),
                data_validation_config=DataValidationConfig(
                    valid_train_file_path=os.path.join("artifacts", "valid", "train.csv"),
                    valid_test_file_path=os.path.join("artifacts", "valid", "test.csv"),
                    drift_report_file_path=self.config.drift_report_path,
                ),
            )
            validation_artifact = validation.initiate_data_validation()
            logging.info("Validation completed: %s", validation_artifact)

            # 3) TRANSFORMACIÓN DE DATOS
            # -----------------------------
            transformation = DataTransformation(
                DataTransformationConfig(
                    root_dir=self.config.preprocessed_dir,
                    source_path=self.config.feature_store_path,
                    train_ratio=self.config.train_ratio,
                    valid_ratio=self.config.valid_ratio,
                    test_ratio=self.config.test_ratio,
                )
            )
            transformation.run_pipeline()
            logging.info("Transformation completed")
            # -----------------------------
            # 4) ENTRENAMIENTO DEL MODELO
            # -----------------------------
            trainer = ModelTrainer(
                ModelTrainerConfig(
                    root_dir=self.config.model_dir,
                    model_name=self.config.model_name,
                    data_path_X_train=os.path.join(self.config.preprocessed_dir, "train_transformed.csv"),
                    data_path_X_valida=os.path.join(self.config.preprocessed_dir, "valid_transformed.csv"),
                    data_path_Y_train=os.path.join(self.config.preprocessed_dir, "train_transformed.csv"),
                    data_path_Y_valida=os.path.join(self.config.preprocessed_dir, "valid_transformed.csv"),
                )
            )
            trainer.train_model()
            logging.info("Training completed")

            # -----------------------------
            # 5) EVALUACIÓN DEL MODELO
            # -----------------------------
            evaluator = DataEvaluation(
                DataEvaluationConfig(
                    model_path=os.path.join(self.config.model_dir, self.config.model_name),
                    train_data_path=os.path.join(self.config.preprocessed_dir, "train_transformed.csv"),
                    valid_data_path=os.path.join(self.config.preprocessed_dir, "valid_transformed.csv"),
                    test_data_path=os.path.join(self.config.preprocessed_dir, "test_transformed.csv"),
                    report_file_path=self.config.evaluation_report_path,
                )
            )
            evaluation_report = evaluator.evaluate()
            logging.info("Evaluation completed")

            # -----------------------------
            # RESULTADO FINAL DEL PIPELINE
            # -----------------------------
            result = {
                "status": "success",
                "feature_store_path": ingestion_artifact.feature_store_file_path,
                "drift_report_path": validation_artifact.drift_report_file_path,
                "model_path": os.path.join(self.config.model_dir, self.config.model_name),
                "evaluation_report_path": self.config.evaluation_report_path,
                "evaluation_report": evaluation_report,
            }

            logging.info("Pipeline finished successfully")
            return result
        
        except Exception as e:
            # Manejo de errores con excepción personalizada
            raise SalesForecastException(e, sys)



# Función auxiliar para ejecutar el pipeline desde otros módulos
def run_pipeline() -> Dict[str, Any]:
    return TrainingPipeline().run()

# Punto de entrada cuando se ejecuta este archivo directamente
if __name__ == "__main__":
    output = run_pipeline()
    print("Pipeline completed")
    print(output)





