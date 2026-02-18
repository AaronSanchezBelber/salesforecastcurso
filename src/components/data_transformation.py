"""Data transformation pipeline for SalesForecast."""

import os
import sys
from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from src.exception.exception import SalesForecastException
from src.logging.logger import logging


@dataclass
class DataTransformationConfig:
    root_dir: str
    train_path: str
    test_path: str


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    # -------------------- PREPROCESS --------------------
    def preprocess_df(self, df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
        try:
            # Convertir nombres de columnas a mayúsculas
            df.columns = map(str.upper, df.columns)

            # Convertir columna DATE a datetime
            df["DATE"] = pd.to_datetime(df["DATE"], format="%Y-%m-%d", errors="coerce")

            # Extraer hora y día de la semana
            df["HOUR"] = df["DATE"].dt.hour
            df["DAY_OF_WEEK"] = df["DATE"].dt.day_of_week

            # Codificar ciudad en variable numérica
            df["CITY_ID"] = OrdinalEncoder().fit_transform(df[["CITY"]])

            # Renombrar columnas
            df.rename(columns={"CITY": "CITY_NAME", "ITEM_CNT_DAY": "SALES"}, inplace=True)

            # Guardar si es necesario
            if save:
                os.makedirs(self.config.root_dir, exist_ok=True)
                out = os.path.join(self.config.root_dir, "01preprocess_df.csv")
                df.to_csv(out, index=False)
                logging.info(f"Saved preprocess dataframe to {out}")

            print("Preprocess stage completed")
            return df

        except Exception as e:
            raise SalesForecastException(e, sys)

    # -------------------- TIME VARIABLES --------------------
    def time_vars(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            # Días festivos
            df.loc[df["DAY_OF_WEEK"] > 4, "HOLIDAYS_DAYS_REVENUE"] = df["SALES"] * df["ITEM_PRICE"]
            df.loc[df["DAY_OF_WEEK"] < 5, "HOLIDAYS_DAYS_REVENUE"] = 0

            # Ventas días laborales y festivos
            df["WORK_DAYS_SALES"] = np.where(df["DAY_OF_WEEK"] < 5, df["SALES"], 0)
            df["HOLIDAYS_DAYS_SALES"] = np.where(df["DAY_OF_WEEK"] > 4, df["SALES"], 0)

            # Guardar
            out = os.path.join(self.config.root_dir, "02time_vars.csv")
            df.to_csv(out, index=False)
            logging.info(f"Saved time vars dataframe to {out}")

            print("Time vars stage completed")
            return df

        except Exception as e:
            raise SalesForecastException(e, sys)

    # -------------------- CASH VARIABLES --------------------
    def cash_vars(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df["REVENUE"] = df["ITEM_PRICE"] * df["SALES"]
            df["UNIQUE_DAYS_WITH_SALES"] = df["DATE"]
            df["TOTAL_TRANSACTIONS"] = df["SALES"]
            df["MONTH_DAY"] = df["DATE"].dt.month

            out = os.path.join(self.config.root_dir, "03cash_vars.csv")
            df.to_csv(out, index=False)
            logging.info(f"Saved cash vars dataframe to {out}")

            print("Cash vars stage completed")
            return df

        except Exception as e:
            raise SalesForecastException(e, sys)

    # -------------------- MONTHLY AGGREGATION --------------------
    def groupby_month(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

            df_monthly_agg = (
                df.set_index("DATE")
                .groupby(["UNIQUE_ID"])
                .resample("MS")
                .agg(
                    {
                        "SALES": np.sum,
                        "REVENUE": np.sum,
                        "UNIQUE_DAYS_WITH_SALES": lambda x: len(set(x)),
                        "TOTAL_TRANSACTIONS": len,
                        "ITEM_PRICE": np.mean,
                        "HOLIDAYS_DAYS_REVENUE": np.sum,
                        "HOLIDAYS_DAYS_SALES": np.sum,
                        "WORK_DAYS_SALES": np.sum,
                    }
                )
                .rename(
                    columns={
                        "SALES": "MONTHLY_SALES",
                        "REVENUE": "MONTHLY_REVENUE",
                        "ITEM_PRICE": "MONTHLY_MEAN_PRICE",
                        "HOLIDAYS_DAYS_REVENUE": "MONTHLY_HOLIDAYS_DAYS_REVENUE",
                        "HOLIDAYS_DAYS_SALES": "MONTHLY_HOLIDAYS_DAYS_SALES",
                        "WORK_DAYS_SALES": "MONTHLY_WORK_DAYS_SALES",
                    }
                )
                .reset_index()
            )

            out = os.path.join(self.config.root_dir, "04df_monthly_agg.csv")
            df_monthly_agg.to_csv(out, index=False)
            logging.info(f"Saved monthly agg dataframe to {out}")

            print("Monthly aggregation stage completed")
            return df_monthly_agg

        except Exception as e:
            raise SalesForecastException(e, sys)

    # -------------------- BUILD FULL RANGE --------------------
    def build_full_range(
        self, df: pd.DataFrame, df_monthly_agg: pd.DataFrame, date: str = "2015-10-31"
    ) -> pd.DataFrame:
        try:
            df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
            df_monthly_agg["DATE"] = pd.to_datetime(df_monthly_agg["DATE"], errors="coerce")
            df["UNIQUE_ID"] = df["UNIQUE_ID"].astype(str)
            df_monthly_agg["UNIQUE_ID"] = df_monthly_agg["UNIQUE_ID"].astype(str)

            min_date = df["DATE"].min()
            date_prediction = np.datetime64(date)

            unique_id = sorted(df_monthly_agg["UNIQUE_ID"].unique())
            date_range = pd.date_range(min_date, date_prediction, freq="ME")

            cartesian_product = pd.MultiIndex.from_product(
                [date_range, unique_id], names=["DATE", "UNIQUE_ID"]
            )

            full_df = pd.DataFrame(index=cartesian_product).reset_index()
            full_df = pd.merge(df_monthly_agg, full_df, on=["DATE", "UNIQUE_ID"], how="right")

            add_info = df[
                [
                    "UNIQUE_ID",
                    "CITY_NAME",
                    "CITY_ID",
                    "SHOP_NAME",
                    "SHOP_ID",
                    "ITEM_CATEGORY_NAME",
                    "ITEM_CATEGORY_ID",
                    "ITEM_NAME",
                    "ITEM_ID",
                ]
            ].drop_duplicates()

            full_df = pd.merge(full_df, add_info, how="left", on="UNIQUE_ID")

            out = os.path.join(self.config.root_dir, "05full_df.csv")
            full_df.to_csv(out, index=False)
            logging.info(f"Saved full range dataframe to {out}")

            print("Full range stage completed")
            return full_df

        except Exception as e:
            raise SalesForecastException(e, sys)

    # -------------------- DROP NULLS --------------------
    def drop_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            cols = [
                "MONTHLY_SALES",
                "MONTHLY_REVENUE",
                "UNIQUE_DAYS_WITH_SALES",
                "TOTAL_TRANSACTIONS",
                "MONTHLY_MEAN_PRICE",
                "MONTHLY_HOLIDAYS_DAYS_REVENUE",
                "MONTHLY_HOLIDAYS_DAYS_SALES",
                "MONTHLY_WORK_DAYS_SALES",
            ]
            df[cols] = df[cols].fillna(0)

            out = os.path.join(self.config.root_dir, "06full_df.csv")
            df.to_csv(out, index=False)
            logging.info(f"Saved null-filled dataframe to {out}")

            print("Drop nulls stage completed")
            return df

        except Exception as e:
            raise SalesForecastException(e, sys)

    # -------------------- EXECUTE TRANSFORMATIONS --------------------
    def execute_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            transformations = [
                (["DATE", "ITEM_ID"], "MONTHLY_SALES", np.sum, "SUM"),
                (["DATE", "ITEM_ID"], "MONTHLY_HOLIDAYS_DAYS_SALES", np.sum, "SUM"),
                (["DATE", "ITEM_ID"], "TOTAL_TRANSACTIONS", np.sum, "SUM"),
                (["DATE", "ITEM_CATEGORY_ID"], "MONTHLY_HOLIDAYS_DAYS_SALES", np.sum, "SUM"),
            ]

            for gl, target, func, name in transformations:
                new_col = "_".join(gl + [target, name])
                temp = df.groupby(gl)[target].agg(func).reset_index().rename(columns={target: new_col})
                temp[f"{new_col}_LAG1"] = temp.groupby(gl[1:])[new_col].shift(1)
                df = pd.merge(df, temp, on=gl, how="left")

            out = os.path.join(self.config.root_dir, "07full_df.csv")
            df.to_csv(out, index=False)
            logging.info(f"Saved transformed dataframe to {out}")

            print("Transformations stage completed")
            return df

        except Exception as e:
            raise SalesForecastException(e, sys)

    # -------------------- DROP COLUMNS --------------------
    def columns_drop(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            columns_to_drop = [
                "DATE_ITEM_ID_MONTHLY_SALES_SUM",
                "DATE_ITEM_ID_MONTHLY_HOLIDAYS_DAYS_SALES_SUM",
                "DATE_ITEM_ID_TOTAL_TRANSACTIONS_SUM",
                "DATE_ITEM_CATEGORY_ID_MONTHLY_HOLIDAYS_DAYS_SALES_SUM",
                "MONTHLY_REVENUE",
                "UNIQUE_DAYS_WITH_SALES",
                "TOTAL_TRANSACTIONS",
                "MONTHLY_MEAN_PRICE",
                "CITY_NAME",
                "SHOP_NAME",
                "ITEM_CATEGORY_NAME",
                "ITEM_NAME",
                "MONTHLY_HOLIDAYS_DAYS_SALES",
                "MONTHLY_WORK_DAYS_SALES",
                "SHOP_ID",
            ]
            df = df.drop(columns=[c for c in columns_to_drop if c in df.columns])
            df = df.drop_duplicates()

            out = os.path.join(self.config.root_dir, "08full_df.csv")
            df.to_csv(out, index=False)
            logging.info(f"Saved final dataframe to {out}")

            print("Columns drop stage completed")
            return df

        except Exception as e:
            raise SalesForecastException(e, sys)

    # -------------------- RUN PIPELINE --------------------
    def run_pipeline(self):
        try:
            # ---------- TRAIN ----------
            train_df = pd.read_csv(self.config.train_path)
            train_df = self.preprocess_df(train_df)
            train_df = self.time_vars(train_df)
            train_df = self.cash_vars(train_df)
            train_monthly = self.groupby_month(train_df)
            train_full = self.build_full_range(train_df, train_monthly)
            train_full = self.drop_nulls(train_full)
            train_full = self.execute_transformations(train_full)
            train_final = self.columns_drop(train_full)

            train_final_path = os.path.join(self.config.root_dir, "train_transformed.csv")
            train_final.to_csv(train_final_path, index=False)

            # ---------- TEST ----------
            test_df = pd.read_csv(self.config.test_path)
            test_df = self.preprocess_df(test_df)
            test_df = self.time_vars(test_df)
            test_df = self.cash_vars(test_df)
            test_monthly = self.groupby_month(test_df)
            test_full = self.build_full_range(test_df, test_monthly)
            test_full = self.drop_nulls(test_full)
            test_full = self.execute_transformations(test_full)
            test_final = self.columns_drop(test_full)

            test_final_path = os.path.join(self.config.root_dir, "test_transformed.csv")
            test_final.to_csv(test_final_path, index=False)

            # ---------- COMBINAR Y GUARDAR FINAL ----------
            final_df = pd.concat([train_final, test_final], ignore_index=True)
            final_path = os.path.join(self.config.root_dir, "preprocessed.csv")
            final_df.to_csv(final_path, index=False)
            logging.info(f"Saved final preprocessed dataframe to {final_path}")

            print("All preprocessing stages completed. Output in:", final_path)

        except Exception as e:
            raise SalesForecastException(e, sys)


# -------------------- MAIN --------------------
if __name__ == "__main__":
    preprocess_dir = os.path.join("artifacts", "preprocessed")
    os.makedirs(preprocess_dir, exist_ok=True)

    config = DataTransformationConfig(
        root_dir=preprocess_dir,
        train_path="artifacts/train/train.csv",
        test_path="artifacts/test/test.csv",
    )

    transformer = DataTransformation(config)
    transformer.run_pipeline()
  
  