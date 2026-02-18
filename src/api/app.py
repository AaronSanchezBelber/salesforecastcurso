"""FastAPI app to upload batch CSV and run the pipeline."""

import os
import sys
from pathlib import Path

import pandas as pd
import joblib
from pandas.tseries.offsets import MonthEnd
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# =========================
# PROJECT ROOT
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.chdir(PROJECT_ROOT)

# =========================
# IMPORTS INTERNOS
# =========================
from src.exception.exception import SalesForecastException
from src.pipeline.main import run_pipeline
from src.components.data_transformation import (
    DataTransformation,
    DataTransformationConfig,
)

# =========================
# PATHS
# =========================
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

UPLOAD_PATH = PROJECT_ROOT / "data" / "forecast.csv"
PREDICT_UPLOAD_PATH = PROJECT_ROOT / "data" / "batch_predict.csv"
MODEL_PATH = PROJECT_ROOT / "artifacts" / "model" / "xgb_model.joblib"

# =========================
# APP INIT
# =========================
app = FastAPI(title="SalesForecast API")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


# =========================
# INDEX
# =========================
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# =========================
# UPLOAD + PREDICT
# =========================
@app.post("/upload-and-predict")
def upload_and_predict(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith(".csv"):
            return JSONResponse(
                status_code=400,
                content={"error": "Solo se acepta CSV"},
            )

        os.makedirs(PREDICT_UPLOAD_PATH.parent, exist_ok=True)

        with open(PREDICT_UPLOAD_PATH, "wb") as f:
            f.write(file.file.read())

        # Ejecuta pipeline (si ya tienes modelo entrenado puedes comentar esto)
        run_pipeline()

        transform_config = DataTransformationConfig(
            root_dir=str(PROJECT_ROOT / "artifacts" / "preprocessed"),
            data_path=str(PREDICT_UPLOAD_PATH),
        )

        transformer = DataTransformation(transform_config)

        df = transformer.preprocess_df(save=False)
        df = transformer.time_vars(df)
        df = transformer.cash_vars(df)
        df_monthly = transformer.groupby_month(df)

        max_date = pd.to_datetime(df["DATE"], errors="coerce").max()
        if pd.isna(max_date):
            raise ValueError("No se pudo determinar la fecha máxima del batch")

        next_month_end = (max_date + MonthEnd(1)).strftime("%Y-%m-%d")

        full_df = transformer.build_full_range(
            df=df,
            df_monthly_agg=df_monthly,
            date=next_month_end,
        )

        full_df = transformer.drop_nulls(full_df)
        full_df = transformer.execute_transformations(full_df)
        final_df = transformer.columns_drop(full_df)

        if "UNIQUE_ID" not in final_df.columns:
            raise ValueError("UNIQUE_ID no existe en el dataframe final")

        drop_cols = [
            c for c in ["UNIQUE_ID", "DATE", "MONTHLY_SALES"]
            if c in final_df.columns
        ]

        X = final_df.drop(columns=drop_cols)

        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"No existe el modelo en {MODEL_PATH}")

        model = joblib.load(MODEL_PATH)
        preds = model.predict(X)

        results = [
            {
                "unique_id": uid,
                "prediction_next_month": float(pred),
            }
            for uid, pred in zip(final_df["UNIQUE_ID"].tolist(), preds)
        ]

        return {"status": "ok", "predictions": results}

    except Exception as e:
        raise SalesForecastException(e, sys)
