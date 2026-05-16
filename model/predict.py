"""
predict.py — loads the trained model and runs inference.

Call predict_rain(data_dict) from app.py to get a prediction.
The model is loaded once at import time so it's ready instantly on
the first request without any cold-start delay.
"""

import os
import logging
import numpy as np
import joblib

logger = logging.getLogger(__name__)

BASE_DIR   = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'random_forest.pkl')

# Feature order must match exactly what train.py used.
# The keys match the JSON field names sent by the frontend.
FEATURE_ORDER = [
    'temperature',
    'humidity',
    'pressure',
    'wind',
    'cloud',
]

# Load once at startup — avoids re-loading on every request
try:
    _model = joblib.load(MODEL_PATH)
    logger.info('Random Forest model loaded from %s', MODEL_PATH)
except FileNotFoundError:
    _model = None
    logger.error(
        'Model not found at %s. Run model/train.py to generate it.',
        MODEL_PATH,
    )


def predict_rain(data: dict) -> dict:
    """
    Predict whether it will rain based on 5 weather features.

    Parameters
    ----------
    data : dict — keys: temperature, humidity, pressure, wind, cloud

    Returns
    -------
    dict — { prediction: 0 or 1, probability: 0.0–1.0 }
    """
    if _model is None:
        raise RuntimeError(
            'Model is not loaded. Run model/train.py first.'
        )

    features = np.array([[
        float(data.get('temperature', 0)),
        float(data.get('humidity',    0)),
        float(data.get('pressure',    1013)),
        float(data.get('wind',        0)),
        float(data.get('cloud',       50)),
    ]])

    prediction  = int(_model.predict(features)[0])
    probability = float(_model.predict_proba(features)[0][1])

    return {
        'prediction':  prediction,
        'probability': round(probability, 4),
    }