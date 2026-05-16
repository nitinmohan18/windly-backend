from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
import logging, os, time, httpx
from dotenv import load_dotenv
from model.predict import predict_rain

load_dotenv()
WEATHER_API_KEY  = os.getenv('WEATHER_API_KEY')
WEATHER_BASE_URL = 'https://api.weatherapi.com/v1'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Simple in-memory TTL cache — avoids hammering WeatherAPI on repeated searches
# Format: { key: (timestamp, data) }
_cache: dict[str, tuple[float, object]] = {}

def _cache_get(key: str, ttl: int):
    """Return cached value if still fresh, otherwise None."""
    entry = _cache.get(key)
    if entry and (time.time() - entry[0]) < ttl:
        return entry[1]
    return None

def _cache_set(key: str, value: object):
    _cache[key] = (time.time(), value)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Create a single shared httpx client for the lifetime of the server.
    This reuses TCP connections across requests instead of opening a new
    connection on every single API call — saves ~30-80ms per weather fetch.
    """
    app.state.http = httpx.AsyncClient(
        timeout=8.0,
        limits=httpx.Limits(
            max_connections=20,
            max_keepalive_connections=10,
            keepalive_expiry=30.0,
        ),
    )
    logger.info('Shared HTTP client created.')
    yield
    await app.state.http.aclose()
    logger.info('Shared HTTP client closed.')


app = FastAPI(
    title='Windly Atmospheric Analysis API',
    description='Random Forest classifier + WeatherAPI proxy',
    version='2.0.0',
    lifespan=lifespan,
)

# Compress responses over 500 bytes — weather JSON can be 10-20 KB
app.add_middleware(GZipMiddleware, minimum_size=500)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


class WeatherInput(BaseModel):
    temperature: float = Field(..., ge=-90,  le=60)
    humidity:    float = Field(..., ge=0,    le=100)
    pressure:    float = Field(..., ge=870,  le=1085)
    wind:        float = Field(..., ge=0,    le=300)
    cloud:       float = Field(50,  ge=0,    le=100)


class PredictionResponse(BaseModel):
    prediction:  int
    probability: float


@app.get('/', tags=['Health'])
def health():
    return {'status': 'ok', 'service': 'Windly API v2.0'}


@app.post('/predict', response_model=PredictionResponse, tags=['Prediction'])
def predict(data: WeatherInput):
    try:
        result = predict_rain(data.model_dump())
        logger.info('Prediction: %s  prob=%.3f', result['prediction'], result['probability'])
        return result
    except Exception as exc:
        logger.error('Prediction error: %s', exc, exc_info=True)
        raise HTTPException(status_code=500, detail='Atmospheric analysis unavailable.')


@app.get('/weather/forecast', tags=['Weather Proxy'])
async def weather_forecast(q: str = Query(...)):
    if not WEATHER_API_KEY:
        raise HTTPException(status_code=500, detail='Weather API key not configured.')

    # Cache weather data for 5 minutes — prevents duplicate calls when user
    # searches the same city quickly (unit toggle, back navigation, etc.)
    cache_key = f'forecast:{q.strip().lower()}'
    cached = _cache_get(cache_key, ttl=300)
    if cached:
        logger.info('Cache hit: %s', cache_key)
        return cached

    try:
        response = await app.state.http.get(
            f'{WEATHER_BASE_URL}/forecast.json',
            params={'key': WEATHER_API_KEY, 'q': q, 'days': 3, 'aqi': 'yes'},
        )
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=response.json().get('error', {}).get('message', 'Weather API error'),
            )
        data = response.json()
        _cache_set(cache_key, data)
        return data

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail='Weather API timed out.')
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f'Unreachable: {exc}')


@app.get('/weather/search', tags=['Weather Proxy'])
async def weather_search(q: str = Query(...)):
    if not WEATHER_API_KEY:
        raise HTTPException(status_code=500, detail='Weather API key not configured.')

    # Cache city suggestions for 15 minutes — they change rarely and this
    # endpoint fires on every keystroke after 3 chars, so caching matters.
    cache_key = f'search:{q.strip().lower()}'
    cached = _cache_get(cache_key, ttl=900)
    if cached is not None:
        return cached

    try:
        response = await app.state.http.get(
            f'{WEATHER_BASE_URL}/search.json',
            params={'key': WEATHER_API_KEY, 'q': q},
        )
        data = response.json() if response.is_success else []
        _cache_set(cache_key, data)
        return data
    except Exception:
        return []