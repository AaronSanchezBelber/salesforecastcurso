"""Model training component for SalesForecast."""

# Importa módulos del sistema
import os
import sys
from dataclasses import dataclass

# Importa librerías externas
import joblib
import mlflow
import mlflow.xgboost
import pandas as pd
import xgboost as xgb
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ajusta el path si el paquete no está configurado correctamente
if __package__ is None or __package__ == "":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Importa excepciones y logger personalizados
from src.exception.exception import SalesForecastException
from src.logging.logger import logging


# Define la configuración del entrenador usando dataclass
@dataclass
class ModelTrainerConfig:
    root_dir: str  # Carpeta donde se guardará el modelo
    model_name: str  # Nombre del archivo del modelo
    data_path_X_train: str  # Ruta del dataset X de entrenamiento
    data_path_X_valida: str  # Ruta del dataset X de validación
    data_path_Y_train: str  # Ruta del dataset Y de entrenamiento
    data_path_Y_valida: str  # Ruta del dataset Y de validación
    target_column: str = "MONTHLY_SALES"  # Nombre de la columna objetivo
    mlflow_experiment_name: str = "salesforecast-training"


# Clase principal del entrenador del modelo
class ModelTrainer:
    # Constructor que recibe la configuración
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    # Método para separar X e Y desde los archivos
    def _split_xy(self, data_path_x: str, data_path_y: str) -> tuple[pd.DataFrame, pd.Series]:
        # Carga el dataset X
        data_x = pd.read_csv(data_path_x)

        # Si la columna objetivo está en X, se separa directamente
        if self.config.target_column in data_x.columns:
            y = data_x[self.config.target_column].astype(float)
            x = data_x.drop(columns=[self.config.target_column])
        else:
            # Si no está en X, se carga desde el archivo Y
            data_y = pd.read_csv(data_path_y)

            # Si la columna objetivo está en Y, se usa
            if self.config.target_column in data_y.columns:
                y = data_y[self.config.target_column].astype(float)
            else:
                # Si no, se toma la primera columna como Y
                y = data_y.iloc[:, 0].astype(float)

            # X queda igual
            x = data_x

        # Devuelve X sin nulos y Y
        return x.fillna(0), y

    # Método principal de entrenamiento
    def train_model(self):
        try:
            # Carga X_train y Y_train
            x_train, y_train = self._split_xy(
                data_path_x=self.config.data_path_X_train, data_path_y=self.config.data_path_Y_train
            )

            # Carga X_val y Y_val
            x_val, y_val = self._split_xy(
                data_path_x=self.config.data_path_X_valida, data_path_y=self.config.data_path_Y_valida
            )

            # Obtiene columnas comunes entre train y valid
            common_columns = sorted(set(x_train.columns).intersection(x_val.columns))
            x_train = x_train[common_columns]
            x_val = x_val[common_columns]

            # Detecta columnas no numéricas
            invalid_train_cols = x_train.select_dtypes(include=["object", "category"]).columns.tolist()
            invalid_val_cols = x_val.select_dtypes(include=["object", "category"]).columns.tolist()

            # Si hay columnas no numéricas, se lanza error
            if invalid_train_cols or invalid_val_cols:
                raise ValueError(
                    "ModelTrainer solo acepta features numericas. "
                    f"Columnas no numericas train={invalid_train_cols}, valid={invalid_val_cols}. "
                    "Haz esta conversion en data_transformation."
                )

            # Define el modelo XGBoost con hiperparámetros
            model = xgb.XGBRegressor(
                n_estimators=1200,
                learning_rate=0.03,
                max_depth=6,
                min_child_weight=5,
                gamma=0.2,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=1.0,
                reg_lambda=3.0,
                objective="reg:squarederror",
                eval_metric="rmse",
                early_stopping_rounds=50,
                random_state=175,
            )

            # Log del inicio del entrenamiento
            logging.info("Entrenando modelo: %s", model.__class__.__name__)

            # Entrena el modelo con validación
            model.fit(
                x_train,
                y_train,
                eval_set=[(x_train, y_train), (x_val, y_val)],
                verbose=False,
            )

            # Predicciones para métricas
            y_train_pred = model.predict(x_train)
            y_val_pred = model.predict(x_val)

            # Cálculo de métricas
            rmse_train = mean_squared_error(y_train, y_train_pred) ** 0.5
            rmse = mean_squared_error(y_val, y_val_pred) ** 0.5
            mae = mean_absolute_error(y_val, y_val_pred)
            r2 = r2_score(y_val, y_val_pred)

            # Crea carpeta del modelo si no existe
            os.makedirs(self.config.root_dir, exist_ok=True)

            # Ruta final del modelo
            model_path = os.path.join(self.config.root_dir, self.config.model_name)

            # Guarda el modelo entrenado
            joblib.dump(model, model_path)

            # Log de métricas y guardado
            logging.info(
                "Modelo guardado en: %s | RMSE_TRAIN=%.6f | RMSE_VALID=%.6f | MAE_VALID=%.6f | R2_VALID=%.6f",
                model_path,
                rmse_train,
                rmse,
                mae,
                r2,
            )

            # Registro en MLflow
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(self.config.mlflow_experiment_name)
            with mlflow.start_run(run_name="model_trainer_xgboost"):
                mlflow.log_params(
                    {
                        "model_type": "xgboost_regressor",
                        "target_column": self.config.target_column,
                        "n_estimators": model.get_params().get("n_estimators"),
                        "learning_rate": model.get_params().get("learning_rate"),
                        "max_depth": model.get_params().get("max_depth"),
                        "min_child_weight": model.get_params().get("min_child_weight"),
                        "gamma": model.get_params().get("gamma"),
                        "subsample": model.get_params().get("subsample"),
                        "colsample_bytree": model.get_params().get("colsample_bytree"),
                        "reg_alpha": model.get_params().get("reg_alpha"),
                        "reg_lambda": model.get_params().get("reg_lambda"),
                    }
                )
                mlflow.log_metrics(
                    {
                        "rmse_train": float(rmse_train),
                        "rmse_valid": float(rmse),
                        "mae_valid": float(mae),
                        "r2_valid": float(r2),
                    }
                )
                mlflow.log_artifact(model_path, artifact_path="model_artifact")
                mlflow.xgboost.log_model(model, artifact_path="model")

            # Envío de métricas a Prometheus si está configurado
            pushgateway_url = os.getenv("PUSHGATEWAY_URL")
            if pushgateway_url:
                try:
                    registry = CollectorRegistry()
                    Gauge("salesforecast_val_rmse", "Validation RMSE", registry=registry).set(rmse)
                    Gauge("salesforecast_val_mae", "Validation MAE", registry=registry).set(mae)
                    Gauge("salesforecast_val_r2", "Validation R2", registry=registry).set(r2)
                    push_to_gateway(pushgateway_url, job="model_trainer", registry=registry)
                except Exception as push_err:
                    logging.warning("No se pudo enviar metricas a Prometheus: %s", push_err)

            # Devuelve el modelo entrenado
            return model

        except Exception as e:
            # Manejo de errores con excepción personalizada
            raise SalesForecastException(e, sys)


# Ejecución directa del script
if __name__ == "__main__":
    # Configuración de ejemplo para ejecución local
    config = ModelTrainerConfig(
        root_dir="artifacts/model",
        model_name="xgb_model.joblib",
        data_path_X_train="artifacts/preprocessed/train_transformed.csv",
        data_path_X_valida="artifacts/preprocessed/valid_transformed.csv",
        data_path_Y_train="artifacts/preprocessed/train_transformed.csv",
        data_path_Y_valida="artifacts/preprocessed/valid_transformed.csv",
    )

    # Crea el entrenador y ejecuta el entrenamiento
    trainer = ModelTrainer(config)
    trainer.train_model()
    print("Entrenamiento del modelo completado")
