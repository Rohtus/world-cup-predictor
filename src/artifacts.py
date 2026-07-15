
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def load_model(model_path: str | Path) -> Any:
    """
    Load a serialized machine learning model.

    Parameters
    ----------
    model_path : str or Path
        Path to the serialized model artifact.

    Returns
    -------
    Any
        Loaded model object.
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {model_path}"
        )

    return joblib.load(model_path)


def load_dataframe(
    file_path: str | Path,
    **read_csv_kwargs
) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    Parameters
    ----------
    file_path : str or Path
        Path to the CSV file.

    **read_csv_kwargs
        Additional keyword arguments passed to pandas.read_csv.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {file_path}"
        )

    dataframe = pd.read_csv(file_path, **read_csv_kwargs)

    if dataframe.empty:
        raise ValueError(
            f"Loaded DataFrame is empty: {file_path}"
        )

    return dataframe
