import re
import shutil
import sqlite3

from pathlib import Path

import pandas as pd

from backend.app.config import (
    DATA_DIR,
    ARTIFACTS_DIR,
    OUTPUT_DIR,
)


def resolve_source(source_path: str) -> Path:

    candidate = (DATA_DIR / source_path).resolve()
    base = DATA_DIR.resolve()

    if base not in candidate.parents:
        raise ValueError("Source must be inside data/")

    if not candidate.exists() or not candidate.is_file():
        raise FileNotFoundError(f"Source not found: {source_path}")

    if candidate.suffix.lower() != ".csv":
        raise ValueError("Only CSV files are supported currently.")

    return candidate


def extract_csv(source_path: str, run_id: str) -> str:

    source = resolve_source(source_path)

    destination = ARTIFACTS_DIR / run_id
    destination.mkdir(parents=True, exist_ok=True)

    raw_file = destination / "raw.csv"

    shutil.copy2(source, raw_file)

    # Confirm the extracted CSV can be read.
    pd.read_csv(raw_file, nrows=5)

    return str(raw_file)


def profile_csv(file_path: str) -> dict:

    df = pd.read_csv(file_path)

    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns),
        "dtypes": {
            column: str(dtype)
            for column, dtype in df.dtypes.items()
        },
        "missing_values": {
            column: int(count)
            for column, count in df.isnull().sum().items()
        },
        "duplicate_rows": int(df.duplicated().sum()),
        "sample": df.head(5).fillna("").to_dict(orient="records"),
    }


def execute_transformations(
    file_path: str,
    operations: list[dict],
    run_id: str,
) -> str:

    df = pd.read_csv(file_path)

    for item in operations:

        operation = item["operation"]
        columns = item.get("columns", [])

        if operation == "drop_duplicates":

            df = df.drop_duplicates()

        elif operation == "fill_numeric_missing":

            selected = columns or list(
                df.select_dtypes(include="number").columns
            )

            for column in selected:

                if column not in df.columns:
                    raise ValueError(f"Unknown column: {column}")

                if pd.api.types.is_numeric_dtype(df[column]):
                    median = df[column].median()

                    if pd.notna(median):
                        df[column] = df[column].fillna(median)

        elif operation == "fill_text_missing":

            selected = columns or list(
                df.select_dtypes(include=["object", "string"]).columns
            )

            for column in selected:

                if column not in df.columns:
                    raise ValueError(f"Unknown column: {column}")

                mode = df[column].mode(dropna=True)

                if not mode.empty:
                    df[column] = df[column].fillna(mode.iloc[0])

        elif operation == "trim_whitespace":

            selected = columns or list(
                df.select_dtypes(include=["object", "string"]).columns
            )

            for column in selected:

                if column not in df.columns:
                    raise ValueError(f"Unknown column: {column}")

                df[column] = df[column].apply(
                    lambda value: value.strip()
                    if isinstance(value, str)
                    else value
                )

        elif operation == "standardize_column_names":

            df.columns = [
                re.sub(
                    r"_+",
                    "_",
                    re.sub(
                        r"[^a-zA-Z0-9]+",
                        "_",
                        str(column).strip().lower(),
                    ),
                ).strip("_")
                for column in df.columns
            ]

        else:
            raise ValueError(f"Unsupported operation: {operation}")

    destination = ARTIFACTS_DIR / run_id
    destination.mkdir(parents=True, exist_ok=True)

    output_file = destination / "processed.csv"

    df.to_csv(output_file, index=False)

    return str(output_file)


def validate_csv(file_path: str) -> dict:

    df = pd.read_csv(file_path)

    issues = []

    if df.empty:
        issues.append("Dataset is empty.")

    duplicate_count = int(df.duplicated().sum())

    if duplicate_count:
        issues.append(
            f"{duplicate_count} duplicate rows remain."
        )

    if len(df.columns) == 0:
        issues.append("Dataset has no columns.")

    return {
        "passed": len(issues) == 0,
        "row_count": len(df),
        "column_count": len(df.columns),
        "duplicate_rows": duplicate_count,
        "issues": issues,
    }


def load_to_sqlite(
    file_path: str,
    target_table: str,
) -> dict:

    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", target_table):
        raise ValueError("Invalid SQLite table name.")

    database_path = OUTPUT_DIR / "etl_output.db"

    df = pd.read_csv(file_path)

    with sqlite3.connect(database_path) as connection:

        existing = connection.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type='table' AND name=?
            """,
            (target_table,),
        ).fetchone()

        # Never silently overwrite an existing table.
        if existing:
            raise ValueError(
                f"Table '{target_table}' already exists. "
                "Choose another table or implement an approved load strategy."
            )

        df.to_sql(
            target_table,
            connection,
            if_exists="fail",
            index=False,
        )

        loaded_count = connection.execute(
            f'SELECT COUNT(*) FROM "{target_table}"'
        ).fetchone()[0]

    return {
        "database": str(database_path),
        "table": target_table,
        "loaded_rows": loaded_count,
        "status": "success",
    }