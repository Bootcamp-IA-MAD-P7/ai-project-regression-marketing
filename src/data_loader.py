"""Data loading utilities for the revenue regression project."""

from pathlib import Path

import pandas as pd

from src.config import RAW_DATA_PATH


def load_raw_data(path=None):
    """Load the raw digital marketing dataset.

    Parameters
    ----------
    path : str or pathlib.Path, optional
        Custom CSV path. When omitted, the configured raw data path is used.

    Returns
    -------
    pandas.DataFrame
        Loaded raw dataset.

    Raises
    ------
    FileNotFoundError
        If the dataset file does not exist.
    ValueError
        If the CSV cannot be parsed or is empty.
    """
    data_path = Path(path) if path is not None else RAW_DATA_PATH

    if not data_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {data_path}")

    try:
        data = pd.read_csv(data_path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Raw data file is empty: {data_path}") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"Could not parse raw data file: {data_path}") from exc

    if data.empty:
        raise ValueError(f"Raw data file contains no rows: {data_path}")

    return data
