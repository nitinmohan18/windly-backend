"""
train.py — trains the Random Forest classifier and saves the model.

Run from the backend/ directory:
    python model/train.py

Output: model/random_forest.pkl
"""

import os
import logging
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection  import train_test_split
from sklearn.ensemble         import RandomForestClassifier
from sklearn.calibration      import CalibratedClassifierCV
from sklearn.metrics          import accuracy_score, classification_report, roc_auc_score

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR   = os.path.dirname(__file__)
DATA_PATH  = os.path.join(BASE_DIR, '../data/GlobalWeatherRepository.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'random_forest.pkl')

# CSV column names — must match what the dataset actually contains.
# The API-facing names (temperature, wind, etc.) are mapped in predict.py.
FEATURES_CSV = [
    'temperature_celsius',
    'humidity',
    'pressure_mb',
    'wind_kph',
    'cloud',
]

RAIN_COLUMN = 'precip_mm'


def load_data(path: str) -> pd.DataFrame:
    logger.info('Loading dataset from %s', path)
    df = pd.read_csv(path)
    logger.info('Raw shape: %s', df.shape)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Drop NaN rows and add a binary rain target column."""
    missing = [c for c in FEATURES_CSV + [RAIN_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f'Missing columns in dataset: {missing}')

    df = df[FEATURES_CSV + [RAIN_COLUMN]].dropna().copy()

    # Any non-zero precipitation = rain
    df['RainTomorrow'] = (df[RAIN_COLUMN] > 0).astype(int)

    logger.info(
        'After cleaning: %d rows  |  Rain rate: %.1f%%',
        len(df), df['RainTomorrow'].mean() * 100,
    )
    return df


def train(df: pd.DataFrame):
    X = df[FEATURES_CSV].values
    y = df['RainTomorrow'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    logger.info('Train: %d  |  Test: %d', len(X_train), len(X_test))

    # Sigmoid calibration gives realistic probability outputs instead
    # of the overconfident raw probabilities from the forest
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=15,
        min_samples_split=4,
        n_jobs=-1,
        random_state=42,
    )
    model = CalibratedClassifierCV(rf, method='sigmoid', cv=5)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc     = accuracy_score(y_test, y_pred)
    auc     = roc_auc_score(y_test, y_proba)

    logger.info('Accuracy: %.2f%%  |  ROC-AUC: %.4f', acc * 100, auc)
    logger.info('\n%s', classification_report(y_test, y_pred, target_names=['No Rain', 'Rain']))

    return model


def save_artifacts(model):
    joblib.dump(model, MODEL_PATH)
    logger.info('Model saved: %s', MODEL_PATH)


if __name__ == '__main__':
    df    = load_data(DATA_PATH)
    df    = preprocess(df)
    model = train(df)
    save_artifacts(model)
    logger.info('Training complete.')