"""DLMDSPWP01 single-file solution.

This module implements an end-to-end pipeline that:
1. Loads and validates train/ideal/test CSV files.
2. Selects best-fit ideal functions for y1..y4 via least squares (SSE).
3. Maps test points with threshold delta <= max_dev * sqrt(2).
4. Writes results to SQLite using SQLAlchemy.
5. Generates a Bokeh visualization.

It also includes pytest unit tests in the same file.
Run tests with: pytest main.py
"""

from __future__ import annotations

import argparse
import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from bokeh.io import output_file, save
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.palettes import Category10
from bokeh.plotting import figure
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


# ---------------------------
# Custom Exceptions
# ---------------------------


class DataSetError(Exception):
    """Base exception for dataset and pipeline errors."""


class MissingColumnError(DataSetError):
    """Raised when required columns are missing in an input CSV."""


class DataValidationError(DataSetError):
    """Raised when data content fails validation checks."""


class MappingError(DataSetError):
    """Raised when mapping cannot be completed due to invalid state or data."""


# ---------------------------
# Dataset Classes
# ---------------------------


class BaseDataSet(ABC):
    """Abstract dataset loader/validator."""

    def __init__(self, csv_path: str | Path) -> None:
        self.csv_path = Path(csv_path)
        self.data: pd.DataFrame | None = None

    @property
    @abstractmethod
    def expected_columns(self) -> List[str]:
        """Required columns for this dataset."""

    def load(self) -> pd.DataFrame:
        """Load and validate CSV data."""
        try:
            df = pd.read_csv(self.csv_path)
        except FileNotFoundError as exc:
            raise DataValidationError(f"File not found: {self.csv_path}") from exc
        except Exception as exc:
            raise DataValidationError(f"Failed to read CSV: {self.csv_path}") from exc

        self.validate(df)
        self.data = df
        return df

    def validate(self, df: pd.DataFrame) -> None:
        """Validate required columns, nulls, and numeric types."""
        missing = [c for c in self.expected_columns if c not in df.columns]
        if missing:
            raise MissingColumnError(
                f"{self.csv_path}: missing required columns: {missing}"
            )

        if df[self.expected_columns].isnull().any().any():
            raise DataValidationError(f"{self.csv_path}: null values in required columns")

        for col in self.expected_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise DataValidationError(
                    f"{self.csv_path}: column '{col}' must be numeric"
                )


class TrainingDataSet(BaseDataSet):
    """Dataset class for train.csv."""

    @property
    def expected_columns(self) -> List[str]:
        return ["x", "y1", "y2", "y3", "y4"]


class IdealDataSet(BaseDataSet):
    """Dataset class for ideal.csv."""

    @property
    def expected_columns(self) -> List[str]:
        return ["x"] + [f"y{i}" for i in range(1, 51)]


class TestDataSet(BaseDataSet):
    """Dataset class for test.csv."""
    __test__ = False

    @property
    def expected_columns(self) -> List[str]:
        return ["x", "y"]


# ---------------------------
# Selection + Mapping Classes
# ---------------------------


class IdealFunctionSelector:
    """Selects best ideal functions for each training function by SSE."""

    def __init__(self) -> None:
        self.mapping: Dict[str, str] = {}
        self.sse_scores: Dict[str, float] = {}
        self.max_deviation_by_train: Dict[str, float] = {}

    def select(
        self, training_df: pd.DataFrame, ideal_df: pd.DataFrame
    ) -> Tuple[Dict[str, str], Dict[str, float], Dict[str, float]]:
        merged = pd.merge(
            training_df, ideal_df, on="x", how="inner", suffixes=("_train", "_ideal")
        )
        if merged.empty:
            raise MappingError("No overlapping x values between training and ideal datasets")

        train_cols = ["y1", "y2", "y3", "y4"]
        ideal_cols = [f"y{i}" for i in range(1, 51)]

        for t_col in train_cols:
            best_ideal = None
            best_sse = float("inf")
            best_max_dev = None

            t_vals = merged[f"{t_col}_train"].to_numpy()
            for i_col in ideal_cols:
                i_col_merged = f"{i_col}_ideal" if i_col in train_cols else i_col
                i_vals = merged[i_col_merged].to_numpy()
                diff = t_vals - i_vals
                sse = float((diff**2).sum())
                if sse < best_sse:
                    best_sse = sse
                    best_ideal = i_col
                    best_max_dev = float(abs(diff).max())

            if best_ideal is None or best_max_dev is None:
                raise MappingError(f"Failed selecting ideal function for {t_col}")

            self.mapping[t_col] = best_ideal
            self.sse_scores[t_col] = best_sse
            self.max_deviation_by_train[t_col] = best_max_dev

        return self.mapping, self.sse_scores, self.max_deviation_by_train


class BaseMapper(ABC):
    """Abstract base class for test-point mapping strategies."""

    @abstractmethod
    def map_points(self, test_df: pd.DataFrame, ideal_df: pd.DataFrame) -> pd.DataFrame:
        """Map test points to ideal functions and return result frame."""


class ThresholdMapper(BaseMapper):
    """Maps test points using minimal delta under threshold max_dev * factor."""

    def __init__(
        self,
        train_to_ideal: Dict[str, str],
        max_deviation_by_train: Dict[str, float],
        threshold_factor: float = math.sqrt(2.0),
    ) -> None:
        self.train_to_ideal = train_to_ideal
        self.threshold_factor = threshold_factor

        # Convert train-based max deviations to ideal-based thresholds.
        self.ideal_thresholds: Dict[str, float] = {}
        for train_col, ideal_col in train_to_ideal.items():
            threshold = max_deviation_by_train[train_col] * threshold_factor
            if ideal_col not in self.ideal_thresholds:
                self.ideal_thresholds[ideal_col] = threshold
            else:
                self.ideal_thresholds[ideal_col] = max(self.ideal_thresholds[ideal_col], threshold)

    def map_points(self, test_df: pd.DataFrame, ideal_df: pd.DataFrame) -> pd.DataFrame:
        selected_ideals = sorted(self.ideal_thresholds.keys())
        if not selected_ideals:
            raise MappingError("No selected ideal functions available for mapping")

        ideal_subset = ideal_df[["x"] + selected_ideals].copy()
        merged = pd.merge(test_df, ideal_subset, on="x", how="left")

        results = []
        for _, row in merged.iterrows():
            x_val = float(row["x"])
            y_test = float(row["y"])

            best_ideal = None
            best_delta = float("inf")

            for ideal_col in selected_ideals:
                y_ideal = row[ideal_col]
                if pd.isna(y_ideal):
                    continue
                delta = abs(y_test - float(y_ideal))
                if delta < best_delta:
                    best_delta = delta
                    best_ideal = ideal_col

            if best_ideal is None:
                results.append(
                    {
                        "x": x_val,
                        "y": y_test,
                        "delta_y": None,
                        "ideal_func": None,
                    }
                )
                continue

            threshold = self.ideal_thresholds[best_ideal]
            is_mapped = best_delta <= threshold

            results.append(
                {
                    "x": x_val,
                    "y": y_test,
                    "delta_y": float(best_delta),
                    "ideal_func": best_ideal if is_mapped else None,
                }
            )

        return pd.DataFrame(results)


# ---------------------------
# Persistence + Plotting
# ---------------------------


class SQLiteRepository:
    """Stores dataframes in SQLite via SQLAlchemy."""

    def __init__(self, db_url: str = "sqlite:///results.db") -> None:
        self.db_url = db_url
        self.engine: Engine = create_engine(db_url)

    def write_all(
        self,
        training_df: pd.DataFrame,
        ideal_df: pd.DataFrame,
        test_results_df: pd.DataFrame,
    ) -> None:
        training_df.to_sql("training_data", self.engine, if_exists="replace", index=False)
        ideal_df.to_sql("ideal_functions", self.engine, if_exists="replace", index=False)
        test_results_df.to_sql("test_results", self.engine, if_exists="replace", index=False)

    def count_rows(self, table_name: str) -> int:
        with self.engine.connect() as conn:
            return int(conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar_one())


class PlotBuilder:
    """Builds and saves Bokeh visualization for training, ideal, and test mappings."""

    @staticmethod
    def build_and_save(
        training_df: pd.DataFrame,
        ideal_df: pd.DataFrame,
        selected_ideals: List[str],
        test_results_df: pd.DataFrame,
        output_html: str = "outputs/plot.html",
    ) -> None:
        output_path = Path(output_html)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        p = figure(
            title="DLMDSPWP01: Training, Selected Ideal Functions, and Test Mapping",
            x_axis_label="x",
            y_axis_label="y",
            width=1200,
            height=700,
            tools="pan,wheel_zoom,box_zoom,reset,save",
        )

        # Training lines
        for i, t_col in enumerate(["y1", "y2", "y3", "y4"]):
            p.line(
                training_df["x"],
                training_df[t_col],
                line_width=2,
                alpha=0.9,
                color=Category10[10][i],
                legend_label=f"train_{t_col}",
            )

        # Selected ideal lines
        for i, i_col in enumerate(selected_ideals):
            p.line(
                ideal_df["x"],
                ideal_df[i_col],
                line_width=3,
                alpha=0.85,
                line_dash="dashed",
                color=Category10[10][(i + 4) % 10],
                legend_label=f"selected_{i_col}",
            )

        # All test points
        p.scatter(
            test_results_df["x"],
            test_results_df["y"],
            size=5,
            alpha=0.35,
            color="gray",
            marker="circle",
            legend_label="all_test_points",
        )

        # Mapped test points (highlighted)
        mapped_df = test_results_df[test_results_df["ideal_func"].notna()].copy()
        if not mapped_df.empty:
            source = ColumnDataSource(mapped_df)
            mapped_renderer = p.scatter(
                x="x",
                y="y",
                size=9,
                alpha=0.95,
                color="firebrick",
                marker="diamond",
                source=source,
                legend_label="mapped_test_points",
            )

            hover = HoverTool(
                renderers=[mapped_renderer],
                tooltips=[
                    ("x", "@x"),
                    ("y", "@y"),
                    ("delta", "@delta_y"),
                    ("ideal", "@ideal_func"),
                ],
            )
            p.add_tools(hover)

        p.legend.click_policy = "hide"
        output_file(str(output_path), title="DLMDSPWP01 Results")
        save(p)


# ---------------------------
# Pipeline
# ---------------------------


def run_pipeline(
    train_csv: str = "train.csv",
    ideal_csv: str = "ideal.csv",
    test_csv: str = "test.csv",
    db_url: str = "sqlite:///results.db",
    plot_html: str = "outputs/plot.html",
    threshold_factor: float = math.sqrt(2.0),
) -> Dict[str, object]:
    """Run the complete DLMDSPWP01 pipeline and return summary info."""
    training_df = TrainingDataSet(train_csv).load()
    ideal_df = IdealDataSet(ideal_csv).load()
    test_df = TestDataSet(test_csv).load()

    selector = IdealFunctionSelector()
    train_to_ideal, sse_scores, max_dev_by_train = selector.select(training_df, ideal_df)

    mapper = ThresholdMapper(
        train_to_ideal=train_to_ideal,
        max_deviation_by_train=max_dev_by_train,
        threshold_factor=threshold_factor,
    )
    test_results_df = mapper.map_points(test_df, ideal_df)

    repo = SQLiteRepository(db_url=db_url)
    repo.write_all(training_df, ideal_df, test_results_df)

    selected_ideals = sorted(set(train_to_ideal.values()))
    PlotBuilder.build_and_save(
        training_df=training_df,
        ideal_df=ideal_df,
        selected_ideals=selected_ideals,
        test_results_df=test_results_df,
        output_html=plot_html,
    )

    mapped_count = int(test_results_df["ideal_func"].notna().sum())
    unmapped_count = int(test_results_df["ideal_func"].isna().sum())

    return {
        "mapping": train_to_ideal,
        "sse_scores": sse_scores,
        "max_dev_by_train": max_dev_by_train,
        "mapped_count": mapped_count,
        "unmapped_count": unmapped_count,
        "selected_ideal_count": len(selected_ideals),
        "db_url": db_url,
        "plot_html": plot_html,
    }


def print_summary(summary: Dict[str, object]) -> None:
    """Print a concise execution summary."""
    print("Selected ideal functions:")
    mapping = summary["mapping"]
    for train_col, ideal_col in mapping.items():
        sse = summary["sse_scores"][train_col]
        max_dev = summary["max_dev_by_train"][train_col]
        print(f"  {train_col} -> {ideal_col} | SSE={sse:.6f} | max_dev={max_dev:.6f}")

    print(f"Mapped test points:   {summary['mapped_count']}")
    print(f"Unmapped test points: {summary['unmapped_count']}")
    print(f"Database:             {summary['db_url']}")
    print(f"Plot:                 {summary['plot_html']}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="DLMDSPWP01 single-file pipeline")
    parser.add_argument("--train", default="train.csv", help="Path to train.csv")
    parser.add_argument("--ideal", default="ideal.csv", help="Path to ideal.csv")
    parser.add_argument("--test", default="test.csv", help="Path to test.csv")
    parser.add_argument(
        "--db-url",
        default="sqlite:///results.db",
        help="SQLAlchemy database URL (default: sqlite:///results.db)",
    )
    parser.add_argument(
        "--plot",
        default="outputs/plot.html",
        help="Output HTML path for Bokeh plot",
    )
    parser.add_argument(
        "--threshold-factor",
        type=float,
        default=math.sqrt(2.0),
        help="Mapping threshold factor; default sqrt(2)",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    summary = run_pipeline(
        train_csv=args.train,
        ideal_csv=args.ideal,
        test_csv=args.test,
        db_url=args.db_url,
        plot_html=args.plot,
        threshold_factor=args.threshold_factor,
    )
    print_summary(summary)


# ---------------------------
# Pytest Unit Tests
# ---------------------------


def test_missing_columns_raises_custom_exception(tmp_path: Path) -> None:
    """CSV validation should raise MissingColumnError when required columns are absent."""
    bad_df = pd.DataFrame({"x": [0, 1], "y1": [1.0, 2.0]})
    csv_path = tmp_path / "bad_train.csv"
    bad_df.to_csv(csv_path, index=False)

    ds = TrainingDataSet(csv_path)
    try:
        ds.load()
        assert False, "Expected MissingColumnError"
    except MissingColumnError:
        assert True


def test_least_squares_selection_synthetic() -> None:
    """SSE selector should choose exact/smallest-error ideal functions."""
    train_df = pd.DataFrame(
        {
            "x": [0, 1, 2],
            "y1": [1, 2, 3],
            "y2": [2, 3, 4],
            "y3": [5, 5, 5],
            "y4": [0, 1, 0],
        }
    )

    ideal_data = {"x": [0, 1, 2]}
    for i in range(1, 51):
        ideal_data[f"y{i}"] = [100, 100, 100]
    ideal_data["y7"] = [1, 2, 3]   # best for train y1
    ideal_data["y8"] = [2, 3, 4]   # best for train y2
    ideal_data["y9"] = [5, 5, 5]   # best for train y3
    ideal_data["y10"] = [0, 1, 0]  # best for train y4

    ideal_df = pd.DataFrame(ideal_data)

    selector = IdealFunctionSelector()
    mapping, sse_scores, _ = selector.select(train_df, ideal_df)

    assert mapping["y1"] == "y7"
    assert mapping["y2"] == "y8"
    assert mapping["y3"] == "y9"
    assert mapping["y4"] == "y10"
    assert sse_scores["y1"] == 0.0


def test_mapping_threshold_rule_sqrt2() -> None:
    """Point maps only when delta <= max_dev * sqrt(2)."""
    test_df = pd.DataFrame({"x": [0, 1], "y": [10.0, 20.0]})
    ideal_df = pd.DataFrame(
        {
            "x": [0, 1],
            "y3": [9.0, 18.0],
        }
    )

    mapper = ThresholdMapper(
        train_to_ideal={"y1": "y3"},
        max_deviation_by_train={"y1": 1.5},
        threshold_factor=math.sqrt(2.0),
    )

    result = mapper.map_points(test_df, ideal_df)
    # deltas are 1.0 and 2.0; threshold = 1.5 * sqrt(2) ~= 2.121 -> both map
    assert pd.notna(result.loc[0, "ideal_func"])
    assert pd.notna(result.loc[1, "ideal_func"])

    mapper_strict = ThresholdMapper(
        train_to_ideal={"y1": "y3"},
        max_deviation_by_train={"y1": 1.0},
        threshold_factor=math.sqrt(2.0),
    )
    result_strict = mapper_strict.map_points(test_df, ideal_df)
    # threshold ~= 1.414; second point delta=2.0 should not map
    assert pd.notna(result_strict.loc[0, "ideal_func"])
    assert pd.isna(result_strict.loc[1, "ideal_func"])


def test_sqlite_insert_and_query_memory() -> None:
    """Basic SQLite write/read should work with in-memory DB."""
    training_df = pd.DataFrame({"x": [0], "y1": [1], "y2": [2], "y3": [3], "y4": [4]})
    ideal_data = {"x": [0]}
    for i in range(1, 51):
        ideal_data[f"y{i}"] = [float(i)]
    ideal_df = pd.DataFrame(ideal_data)
    test_results_df = pd.DataFrame(
        [{"x": 0.0, "y": 1.0, "delta_y": 0.0, "ideal_func": "y1"}]
    )

    repo = SQLiteRepository(db_url="sqlite:///:memory:")
    repo.write_all(training_df, ideal_df, test_results_df)

    assert repo.count_rows("training_data") == 1
    assert repo.count_rows("ideal_functions") == 1
    assert repo.count_rows("test_results") == 1


if __name__ == "__main__":
    main()
