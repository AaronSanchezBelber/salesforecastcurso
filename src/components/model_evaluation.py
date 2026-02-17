"""Model evaluation component for SalesForecast."""

# Importa módulos del sistema
import os
import sys
from dataclasses import dataclass
from typing import Dict

# Importa librerías externas necesarias
import joblib
import mlflow
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Carga variables de entorno desde .env
# -----------------------------
from dotenv import load_dotenv
load_dotenv()  # Esto permite que MLFLOW_TRACKING_URI se cargue automáticamente



# Importa excepciones y logger personalizados del proyecto
from src.exception.exception import SalesForecastException
from src.logging.logger import logging


# Define la configuración del evaluador usando dataclass
@dataclass
class DataEvaluationConfig:
    model_path: str  # Ruta del modelo entrenado
    train_data_path: str  # Ruta del dataset de entrenamiento
    valid_data_path: str  # Ruta del dataset de validación
    test_data_path: str  # Ruta del dataset de test
    target_column: str = "MONTHLY_SALES"  # Nombre de la columna objetivo
    report_file_path: str = os.path.join("artifacts", "evaluation", "report.yaml")  # Ruta del reporte YAML
    mlflow_experiment_name: str = "salesforecast-evaluation"  # Nombre del experimento MLflow


# Clase principal encargada de la evaluación del modelo
class DataEvaluation:
    # Constructor que recibe la configuración
    def __init__(self, config: DataEvaluationConfig):
        self.config = config

    # Método para cargar X e Y desde un archivo CSV
    def _load_xy(self, file_path: str) -> tuple[pd.DataFrame, pd.Series]:
        # Carga el archivo CSV
        df = pd.read_csv(file_path)

        # Verifica que exista la columna objetivo
        if self.config.target_column not in df.columns:
            raise ValueError(f"No existe target_column={self.config.target_column} en {file_path}")

        # Extrae Y como float
        y = df[self.config.target_column].astype(float)

        # Extrae X eliminando la columna objetivo
        x = df.drop(columns=[self.config.target_column]).fillna(0)

        # Verifica que no existan columnas no numéricas
        invalid_cols = x.select_dtypes(include=["object", "category"]).columns.tolist()
        if invalid_cols:
            raise ValueError(
                f"DataEvaluation solo acepta features numericas. "
                f"Columnas no numericas en {file_path}: {invalid_cols}"
            )

        # Devuelve X e Y
        return x, y

    # Método estático para calcular métricas
    @staticmethod
    def _metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
        # Calcula RMSE
        rmse = mean_squared_error(y_true, y_pred) ** 0.5
        # Calcula MAE
        mae = mean_absolute_error(y_true, y_pred)
        # Calcula R2
        r2 = r2_score(y_true, y_pred)
        # Devuelve diccionario con métricas
        return {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}

    # Método estático para escribir un archivo YAML
    @staticmethod
    def _write_yaml(file_path: str, content: Dict) -> None:
        # Crea carpeta si no existe
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # Escribe el archivo YAML
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(content, f, sort_keys=False)

    # Método principal de evaluación
    def evaluate(self) -> Dict:
        try:
            # Verifica que el modelo exista
            if not os.path.exists(self.config.model_path):
                raise FileNotFoundError(f"No existe el modelo en: {self.config.model_path}")

            # Carga el modelo entrenado
            model = joblib.load(self.config.model_path)

            # Carga datasets de train, valid y test
            x_train, y_train = self._load_xy(self.config.train_data_path)
            x_valid, y_valid = self._load_xy(self.config.valid_data_path)
            x_test, y_test = self._load_xy(self.config.test_data_path)

            # Obtiene columnas comunes entre los tres datasets
            common_columns = sorted(set(x_train.columns) & set(x_valid.columns) & set(x_test.columns))
            x_train = x_train[common_columns]
            x_valid = x_valid[common_columns]
            x_test = x_test[common_columns]

            # Calcula métricas para cada dataset
            train_metrics = self._metrics(y_train, model.predict(x_train))
            valid_metrics = self._metrics(y_valid, model.predict(x_valid))
            test_metrics = self._metrics(y_test, model.predict(x_test))

            # Calcula indicadores de sobreajuste
            overfit_gap_rmse = valid_metrics["rmse"] - train_metrics["rmse"]
            overfit_ratio_rmse = (
                valid_metrics["rmse"] / train_metrics["rmse"] if train_metrics["rmse"] > 0 else float("inf")
            )

            # Construye el reporte final
            report = {
                "model_path": self.config.model_path,
                "target_column": self.config.target_column,
                "n_features": len(common_columns),
                "n_train": int(len(y_train)),
                "n_valid": int(len(y_valid)),
                "n_test": int(len(y_test)),
                "train_metrics": train_metrics,
                "valid_metrics": valid_metrics,
                "test_metrics": test_metrics,
                "overfit": {
                    "rmse_gap_valid_minus_train": float(overfit_gap_rmse),
                    "rmse_ratio_valid_div_train": float(overfit_ratio_rmse),
                },
            }

            # Guarda el reporte YAML
            self._write_yaml(self.config.report_file_path, report)
            logging.info("Data evaluation report saved: %s", self.config.report_file_path)

            # Log de métricas principales
            logging.info(
                "Evaluation | TRAIN_RMSE=%.6f VALID_RMSE=%.6f TEST_RMSE=%.6f",
                train_metrics["rmse"],
                valid_metrics["rmse"],
                test_metrics["rmse"],
            )

            # Configura MLflow usando .env
            # -----------------------------
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(self.config.mlflow_experiment_name)


            # Inicia un run de MLflow
            with mlflow.start_run(run_name="data_evaluation"):
                # Registra parámetros
                mlflow.log_params(
                    {
                        "target_column": self.config.target_column,
                        "model_path": self.config.model_path,
                        "n_features": len(common_columns),
                        "n_train": int(len(y_train)),
                        "n_valid": int(len(y_valid)),
                        "n_test": int(len(y_test)),
                    }
                )

                # Registra métricas
                mlflow.log_metrics(
                    {
                        "train_rmse": train_metrics["rmse"],
                        "train_mae": train_metrics["mae"],
                        "train_r2": train_metrics["r2"],
                        "valid_rmse": valid_metrics["rmse"],
                        "valid_mae": valid_metrics["mae"],
                        "valid_r2": valid_metrics["r2"],
                        "test_rmse": test_metrics["rmse"],
                        "test_mae": test_metrics["mae"],
                        "test_r2": test_metrics["r2"],
                        "overfit_rmse_gap": report["overfit"]["rmse_gap_valid_minus_train"],
                        "overfit_rmse_ratio": report["overfit"]["rmse_ratio_valid_div_train"],
                    }
                )

                # Guarda el archivo YAML como artefacto
                mlflow.log_artifact(self.config.report_file_path, artifact_path="evaluation")

            # Devuelve el reporte completo
            return report

        except Exception as e:
            # Manejo de errores con excepción personalizada
            raise SalesForecastException(e, sys)
        

# Punto de entrada cuando se ejecuta este archivo directamente
if __name__ == "__main__":

    # Crea la configuración del evaluador
    config = DataEvaluationConfig(
        model_path="artifacts/model/xgb_model.joblib",  # Ruta del modelo entrenado
        train_data_path="artifacts/preprocessed/train_transformed.csv",  # Train
        valid_data_path="artifacts/preprocessed/valid_transformed.csv",  # Valid
        test_data_path="artifacts/preprocessed/test_transformed.csv",  # Test
    )

    # Crea el evaluador
    evaluator = DataEvaluation(config)

    # Ejecuta la evaluación
    report = evaluator.evaluate()

    # Imprime confirmación
    print("Evaluación del modelo completada")


