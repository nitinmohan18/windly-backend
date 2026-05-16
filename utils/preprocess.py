"""
preprocess.py — shared data-cleaning utilities.

Used by train.py and any future pipeline work.
All column names refer to GlobalWeatherRepository.csv, not weatherAUS.csv.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    'temperature_celsius',
    'humidity',
    'pressure_mb',
    'wind_kph',
    'cloud',
]

RAIN_SOURCE_COLUMN = 'precip_mm'


def load_and_clean(path: str) -> pd.DataFrame:
    """
    Load the CSV and return only the columns we need, with NaNs dropped.
    """
    df = pd.read_csv(path)
    logger.info('Loaded %d rows from %s', len(df), path)

    required = FEATURE_COLUMNS + [RAIN_SOURCE_COLUMN]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f'Missing columns: {missing}\n'
            f'Available: {list(df.columns)}'
        )

    df     = df[required].copy()
    before = len(df)
    df     = df.dropna()
    logger.info('Dropped %d NaN rows (%d remaining)', before - len(df), len(df))
    return df


def add_rain_target(df: pd.DataFrame, threshold_mm: float = 0.0) -> pd.DataFrame:
    """
    Add a binary RainTomorrow column: 1 if precip_mm > threshold, else 0.
    Default threshold of 0 means any measurable rain counts.
    """
    df = df.copy()
    df['RainTomorrow'] = (df[RAIN_SOURCE_COLUMN] > threshold_mm).astype(int)
    logger.info('Rain rate: %.1f%%', df['RainTomorrow'].mean() * 100)
    return df


def validate_input(data: dict) -> dict:
    """
    Validate and clamp a single inference request dict.
    Logs a warning for out-of-range values instead of failing silently.
    """
    ranges = {
        'temperature': (-90,  60),
        'humidity':    (0,    100),
        'pressure':    (870,  1085),
        'wind':        (0,    300),
        'cloud':       (0,    100),
    }

    cleaned = {}
    for key, (lo, hi) in ranges.items():
        val = data.get(key)
        if val is None:
            raise ValueError(f"Missing field: '{key}'")
        val = float(val)
        if val < lo or val > hi:
            logger.warning(
                "'%s' value %.2f out of range [%s, %s] — clamping.",
                key, val, lo, hi,
            )
            val = max(lo, min(hi, val))
        cleaned[key] = val

    return cleaned