"""Data transformation pipeline for SalesForecast."""

import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from src.exception.exception import SalesForecastException
from src.logging.logger import logging


@dataclass
class DataTransformationConfig:
    root_dir: str
    source_path: str
    train_ratio: float = 0.7
    valid_ratio: float = 0.15
    test_ratio: float = 0.15


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    @staticmethod
    def _validate_schema(dataframe: pd.DataFrame) -> None:
        expected_columns = {
            "date",
            "shop_id",
            "item_category_id",
            "item_cnt_day",
            "item_name",
            "item_category_name",
            "unique_id",
            "shop_name",
            "item_id",
            "city",
            "item_price",
        }
        missing = sorted(expected_columns.difference(set(dataframe.columns)))
        if missing:
            raise ValueError(f"Faltan columnas en feature_store: {missing}")

    def build_modeling_dataframe(self, source_df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = source_df.copy()
            df.columns = [c.upper() for c in df.columns]
            df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
            df = df.dropna(subset=["DATE"])
            df["UNIQUE_ID"] = df["UNIQUE_ID"].astype(str)
            df["CITY_ID"] = OrdinalEncoder().fit_transform(df[["CITY"]]).astype(int)

            monthly = (
                df.groupby([pd.Grouper(key="DATE", freq="MS"), "UNIQUE_ID"], as_index=False)
                .agg(
                    MONTHLY_SALES=("ITEM_CNT_DAY", "sum"),
                    ITEM_ID=("ITEM_ID", "first"),
                    ITEM_CATEGORY_ID=("ITEM_CATEGORY_ID", "first"),
                    CITY_ID=("CITY_ID", "first"),
                    ITEM_PRICE_MEAN=("ITEM_PRICE", "mean"),
                )
                .sort_values(["UNIQUE_ID", "DATE"])
                .reset_index(drop=True)
            )

            # Feature engineering sin fuga: solo historial y calendario.
            monthly["LAG_1"] = monthly.groupby("UNIQUE_ID")["MONTHLY_SALES"].shift(1)
            monthly["LAG_2"] = monthly.groupby("UNIQUE_ID")["MONTHLY_SALES"].shift(2)
            monthly["LAG_3"] = monthly.groupby("UNIQUE_ID")["MONTHLY_SALES"].shift(3)
            monthly["ROLLING_MEAN_3"] = monthly.groupby("UNIQUE_ID")["MONTHLY_SALES"].shift(1).rolling(3).mean().reset_index(level=0, drop=True)
            monthly["ROLLING_STD_3"] = monthly.groupby("UNIQUE_ID")["MONTHLY_SALES"].shift(1).rolling(3).std().reset_index(level=0, drop=True)

            monthly["DATE_YEAR"] = monthly["DATE"].dt.year
            monthly["DATE_MONTH"] = monthly["DATE"].dt.month
            monthly["DATE_QUARTER"] = monthly["DATE"].dt.quarter

            parts = monthly["UNIQUE_ID"].str.split("-", n=1, expand=True)
            monthly["UNIQUE_ID_SHOP"] = pd.to_numeric(parts[0], errors="coerce").fillna(-1).astype(int)
            monthly["UNIQUE_ID_ITEM"] = pd.to_numeric(parts[1], errors="coerce").fillna(-1).astype(int)

            monthly = monthly.drop(columns=["DATE", "UNIQUE_ID"])
            monthly = monthly.fillna(0)
            monthly = monthly.drop_duplicates()

            return monthly
        except Exception as e:
            raise SalesForecastException(e, sys)

    def split_transformed_data(self, transformed_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        try:
            if not np.isclose(self.config.train_ratio + self.config.valid_ratio + self.config.test_ratio, 1.0):
                raise ValueError("train_ratio + valid_ratio + test_ratio debe sumar 1.0")

            df = transformed_df.copy()
            split_date = pd.to_datetime(
                df["DATE_YEAR"].astype(int).astype(str)
                + "-"
                + df["DATE_MONTH"].astype(int).astype(str)
                + "-01",
                errors="coerce",
            )
            df["SPLIT_DATE"] = split_date
            df = df.dropna(subset=["SPLIT_DATE"]).sort_values("SPLIT_DATE").reset_index(drop=True)

            unique_dates = np.array(sorted(df["SPLIT_DATE"].dt.date.unique()))
            if len(unique_dates) < 3:
                raise ValueError("No hay suficientes meses para split train/valid/test.")

            train_end_idx = max(1, int(len(unique_dates) * self.config.train_ratio))
            valid_end_idx = max(train_end_idx + 1, int(len(unique_dates) * (self.config.train_ratio + self.config.valid_ratio)))
            train_end_idx = min(train_end_idx, len(unique_dates) - 2)
            valid_end_idx = min(valid_end_idx, len(unique_dates) - 1)

            train_end_date = pd.Timestamp(unique_dates[train_end_idx - 1])
            valid_end_date = pd.Timestamp(unique_dates[valid_end_idx - 1])

            train_df = df[df["SPLIT_DATE"] <= train_end_date].drop(columns=["SPLIT_DATE"])
            valid_df = df[(df["SPLIT_DATE"] > train_end_date) & (df["SPLIT_DATE"] <= valid_end_date)].drop(columns=["SPLIT_DATE"])
            test_df = df[df["SPLIT_DATE"] > valid_end_date].drop(columns=["SPLIT_DATE"])

            return train_df, valid_df, test_df
        except Exception as e:
            raise SalesForecastException(e, sys)

    def run_pipeline(self) -> None:
        try:
            os.makedirs(self.config.root_dir, exist_ok=True)
            if not os.path.exists(self.config.source_path):
                raise FileNotFoundError(f"No existe source_path: {self.config.source_path}")

            source_df = pd.read_csv(self.config.source_path)
            self._validate_schema(source_df)

            transformed_df = self.build_modeling_dataframe(source_df)
            train_df, valid_df, test_df = self.split_transformed_data(transformed_df)

            train_path = os.path.join(self.config.root_dir, "train_transformed.csv")
            valid_path = os.path.join(self.config.root_dir, "valid_transformed.csv")
            test_path = os.path.join(self.config.root_dir, "test_transformed.csv")
            all_path = os.path.join(self.config.root_dir, "preprocessed.csv")

            train_df.to_csv(train_path, index=False)
            valid_df.to_csv(valid_path, index=False)
            test_df.to_csv(test_path, index=False)
            pd.concat([train_df, valid_df, test_df], ignore_index=True).to_csv(all_path, index=False)

            logging.info(
                "Transformacion OK | train=%s valid=%s test=%s",
                train_df.shape,
                valid_df.shape,
                test_df.shape,
            )
            print("All preprocessing stages completed. Output in:", all_path)
        except Exception as e:
            raise SalesForecastException(e, sys)


if __name__ == "__main__":
    config = DataTransformationConfig(
        root_dir=os.path.join("artifacts", "preprocessed"),
        source_path=os.path.join("artifacts", "feature_store", "forecast.csv"),
    )
    DataTransformation(config).run_pipeline()
