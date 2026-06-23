import pandas as pd
from google.cloud import bigquery
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import importlib

class BigQueryLoader:

    def __init__(self, project_id, dataset, write_mode="append"):
        self.client = bigquery.Client(project=project_id)
        self.dataset = dataset
        self.write_mode = write_mode

    def _load_schema(self, schema_name):

        schema_module = importlib.import_module(
            f"schemas.{schema_name}"
        )

        return schema_module.SCHEMA

    def _prepare_dataframe(self, df, schema):

        df.columns = (
            df.columns
            .str.lower()
            .str.strip()
            .str.replace(" ", "_", regex=False)
            .str.replace("-", "_", regex=False)
        )

        for col, dtype in schema.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)

        df["ingestion_timestamp"] = pd.Timestamp.utcnow()

        return df

    def load_csv(self, file_path, table_name, schema_name):

        schema = self._load_schema(schema_name)

        df = pd.read_csv(file_path)
        df = self._prepare_dataframe(df, schema)

        table_id = (
            f"{self.client.project}"
            f".{self.dataset}"
            f".{table_name}"
        )

        job_config = bigquery.LoadJobConfig(
            write_disposition=(
                bigquery.WriteDisposition.WRITE_APPEND
                if self.write_mode == "append"
                else bigquery.WriteDisposition.WRITE_TRUNCATE
            )
        )

        job = self.client.load_table_from_dataframe(
            df,
            table_id,
            job_config=job_config
        )

        job.result()

        print(
            f"[SUCCESS] {len(df)} rows loaded → {table_id}"
        )