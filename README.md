# 🌬️ Windly — Backend

> *The atmospheric intelligence engine powering Windly's cinematic weather experience.*

**Live API:** [windly-backend.onrender.com](https://windly-backend.onrender.com) &nbsp;|&nbsp; **Frontend Repo:** [windly-frontend](https://github.com/nitinmohan18/windly-frontend) &nbsp;|&nbsp; **Docs:** [/docs](https://windly-backend.onrender.com/docs) &nbsp;|&nbsp; ![FastAPI](https://img.shields.io/badge/FastAPI-0.136-009688) ![Python](https://img.shields.io/badge/python-3.10+-blue) ![Render](https://img.shields.io/badge/deployed-render-46E3B7)

---

## 🧠 About the Backend

This is not a simple weather proxy. The Windly backend is a purpose-built **atmospheric intelligence service** responsible for four distinct concerns:

- **Weather Aggregation** — Proxies WeatherAPI's forecast and search endpoints, normalises errors, and returns clean structured JSON to the frontend
- **AI Prediction Engine** — Runs a calibrated Random Forest classifier to estimate rain probability from 5 atmospheric inputs, served via a dedicated `/predict` endpoint
- **TTL Cache Layer** — In-memory caching with per-endpoint TTLs prevents redundant API calls on repeated searches and rapid unit-toggle interactions
- **Async API Orchestration** — A single shared `httpx.AsyncClient` persists across the server's lifetime, reusing TCP connections and keep-alive sessions for every outbound request

The result is a backend that feels fast, handles failure gracefully, and keeps the WeatherAPI key off the client entirely.

---

## 🏗️ Architecture

```
Browser / Vercel Frontend
          │
          │ HTTPS
          ▼
┌─────────────────────────────────────────┐
│         FastAPI Backend (Render)        │
│                                         │
│  ┌─────────────────────────────────┐    │
│  │       GZip Middleware           │    │
│  │  (compresses responses ≥500B)   │    │
│  └──────────────┬──────────────────┘    │
│                 │                       │
│         ┌───────┴────────┐              │
│         │                │              │
│  ┌──────▼──────┐  ┌──────▼──────┐      │
│  │ WeatherAPI  │  │  ML Inference│      │
│  │   Proxy     │  │   Engine     │      │
│  │ /forecast   │  │  /predict    │      │
│  │ /search     │  │              │      │
│  └──────┬──────┘  └──────┬──────┘      │
│         │                │              │
│  ┌──────▼──────┐  ┌──────▼──────┐      │
│  │  TTL Cache  │  │ RandomForest │      │
│  │  (in-memory)│  │  .pkl model  │      │
│  └──────┬──────┘  └─────────────┘      │
│         │                               │
│  ┌──────▼────────────────────────┐      │
│  │  Shared Async httpx.Client    │      │
│  │  (connection pool, keep-alive)│      │
│  └──────┬────────────────────────┘      │
└─────────┼───────────────────────────────┘
          │
          ▼
    WeatherAPI.com
```

---

## 🔌 API Reference

### Endpoints

| Method | Endpoint | Cache TTL | Description |
|---|---|---|---|
| `GET` | `/` | — | Health check |
| `POST` | `/predict` | — | Rain probability prediction |
| `GET` | `/weather/forecast?q={city}` | 5 min | 3-day weather forecast |
| `GET` | `/weather/search?q={query}` | 15 min | City autocomplete suggestions |

---

### `POST /predict`

Runs the Random Forest classifier and returns a calibrated rain probability.

**Request body:**
```json
{
  "temperature": 22.5,
  "humidity": 78,
  "pressure": 1008,
  "wind": 35,
  "cloud": 60
}
```

**Response:**
```json
{
  "prediction": 1,
  "probability": 0.7341
}
```

**Field validation** (enforced by Pydantic):

| Field | Type | Range |
|---|---|---|
| `temperature` | float | −90 to 60 °C |
| `humidity` | float | 0 to 100 % |
| `pressure` | float | 870 to 1085 mb |
| `wind` | float | 0 to 300 km/h |
| `cloud` | float | 0 to 100 % |

**cURL example:**
```bash
curl -X POST https://windly-backend.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "temperature": 22.5,
    "humidity": 78,
    "pressure": 1008,
    "wind": 35,
    "cloud": 60
  }'
```

---

### `GET /weather/forecast`

Proxies WeatherAPI's 3-day forecast endpoint. Returns current conditions, hourly data, and forecast days including AQI.

```bash
curl "https://windly-backend.onrender.com/weather/forecast?q=London"
```

---

### `GET /weather/search`

Returns city name suggestions for autocomplete. Results are cached for 15 minutes — this endpoint fires on every keystroke after 3 characters, so caching meaningfully reduces WeatherAPI usage.

```bash
curl "https://windly-backend.onrender.com/weather/search?q=mum"
```

---

## 📘 Interactive API Docs

FastAPI generates full interactive documentation automatically from the code.

| Interface | URL | Description |
|---|---|---|
| **Swagger UI** | [`/docs`](https://windly-backend.onrender.com/docs) | Try every endpoint live in the browser |
| **ReDoc** | [`/redoc`](https://windly-backend.onrender.com/redoc) | Clean reference documentation |

No API key needed to explore the docs — the WeatherAPI key is server-side only.

---

## 🧪 AI Model & Training Pipeline

### Model Overview

A **Random Forest classifier** with sigmoid probability calibration, trained on the [Global Weather Repository](https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository) dataset. Raw forest probabilities tend to be overconfident; sigmoid calibration via `CalibratedClassifierCV` produces realistic, well-distributed outputs across the full 0–1 range.

**Model configuration:**
```python
RandomForestClassifier(
    n_estimators=120,
    max_depth=10,
    min_samples_split=4,
    n_jobs=-1,
    random_state=42,
)
# Wrapped with:
CalibratedClassifierCV(rf, method='sigmoid', cv=5)
```

### Training Pipeline

```
GlobalWeatherRepository.csv
        │
        ▼
1. Load CSV — select 5 feature columns + precip_mm
        │
        ▼
2. Drop NaN rows — ensures clean training signal
        │
        ▼
3. Generate binary target — precip_mm > 0 → RainTomorrow = 1
        │
        ▼
4. Stratified 80/20 train/test split (random_state=42)
        │
        ▼
5. Fit RandomForest + sigmoid calibration (5-fold CV)
        │
        ▼
6. Evaluate — Accuracy, ROC-AUC, classification report
        │
        ▼
7. Export → model/random_forest.pkl (joblib)
```

### Run training locally

```bash
# Place GlobalWeatherRepository.csv in data/
python model/train.py

# Output: model/random_forest.pkl
# Logs:   accuracy, ROC-AUC, rain rate, classification report
```

---

## ⚡ Performance Optimizations

### Shared Async HTTP Client
A single `httpx.AsyncClient` is created at server startup via FastAPI's `lifespan` context manager and stored on `app.state`. Every weather request reuses this client instead of opening a new TCP connection — saving 30–80ms per request.

```python
app.state.http = httpx.AsyncClient(
    timeout=8.0,
    limits=httpx.Limits(
        max_connections=20,
        max_keepalive_connections=10,
        keepalive_expiry=30.0,
    ),
)
```

### TTL Cache
An in-memory dict cache avoids redundant WeatherAPI calls. Weather data changes slowly enough that caching for a few minutes is invisible to users but saves meaningful API quota.

| Endpoint | TTL | Reason |
|---|---|---|
| `/weather/forecast` | 5 minutes | Forecast data doesn't change second-to-second |
| `/weather/search` | 15 minutes | City names are stable; fires on every keystroke |

### GZip Compression
`GZipMiddleware` compresses all responses over 500 bytes. A typical forecast JSON response is 10–20 KB — compression cuts this to 2–4 KB, meaningfully improving load time on mobile connections.

### Persistent Model Loading
The `random_forest.pkl` model is loaded once at module import time in `predict.py`. It is never re-loaded between requests. This eliminates model deserialisation overhead (which can take 200–400ms) from every prediction call.

### Async Request Handling
All WeatherAPI proxy routes are `async def`, allowing the server to handle other requests while waiting on the external API. Under concurrent load, this prevents request queuing that would occur with synchronous handlers.

---

## 💡 Why FastAPI?

| Feature | Benefit |
|---|---|
| **Async-native** | `async def` routes handle I/O-bound tasks (external API calls) without blocking the server |
| **Pydantic validation** | Request body validation and type coercion with zero boilerplate — invalid inputs are rejected automatically with clear error messages |
| **Auto-generated docs** | `/docs` and `/redoc` are generated from the code itself — always up to date, no manual maintenance |
| **Lightweight** | No ORM, no admin panel, no unnecessary abstractions — just routes, middleware, and models |
| **Production-ready** | Uvicorn + FastAPI is a mature, battle-tested stack used in production at scale |

---

## 🗂️ File Structure

```
backend/
├── app.py                      # FastAPI app — routes, middleware, lifespan, caching
├── requirements.txt
├── .env                        # WEATHER_API_KEY — never committed
├── model/
│   ├── train.py                # Full training pipeline — loads, cleans, trains, exports
│   ├── predict.py              # Model loader + inference function
│   ├── random_forest.pkl       # Trained model artifact — must be committed for deployment
│   └── feature_columns.json   # Feature name reference
└── data/
    └── GlobalWeatherRepository.csv   # Training dataset — gitignored, download from Kaggle
```

---

## 🚀 Running Locally

### 1. Clone & install

```bash
git clone https://github.com/nitinmohan18/windly-backend.git
cd windly-backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
echo "WEATHER_API_KEY=your_key_here" > .env
```

Get a free key at [weatherapi.com](https://www.weatherapi.com/). The free tier covers all of Windly's usage.

### 3. Train the model

Download `GlobalWeatherRepository.csv` from [Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository) and place it in `data/`.

```bash
python model/train.py
# Outputs: model/random_forest.pkl
# Logs accuracy, ROC-AUC, and classification report to the console
```

### 4. Start the server

```bash
uvicorn app:app --reload --port 8000
```

- API: `http://localhost:8000`
- Interactive docs: `http://localhost:8000/docs`

---

## 🌐 Deployment (Render)

### Setup

1. Push this repo to GitHub (**including `model/random_forest.pkl`** — see note below)
2. Create a new **Web Service** on [Render](https://render.com)
3. Configure:

| Setting | Value |
|---|---|
| **Runtime** | Python 3 |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `uvicorn app:app --host 0.0.0.0 --port $PORT` |

4. Add environment variable in the Render dashboard:

| Key | Value |
|---|---|
| `WEATHER_API_KEY` | your WeatherAPI key |

### Important Notes

**`random_forest.pkl` must be committed to the repo.**
`predict.py` loads the model at import time — if the file doesn't exist when the server starts, the `/predict` endpoint will be unavailable. Render has no way to run `train.py` at deploy time (no training dataset on the server), so the trained artifact must be version-controlled.

**Cold starts on the free tier.**
Render's free tier spins down services after ~10 minutes of inactivity. The first request after sleep can take 30–40 seconds while the server wakes up and loads the model. Windly's frontend handles this with a 50-second timeout and a user-facing waiting message — no action needed on the backend side.

**`.env` and `data/` must be gitignored.**
Your WeatherAPI key and the training CSV should never be committed. Confirm both are in `.gitignore` before pushing.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Framework** | FastAPI |
| **Server** | Uvicorn |
| **HTTP Client** | httpx (async, connection-pooled, keep-alive) |
| **Compression** | Starlette GZipMiddleware |
| **Validation** | Pydantic v2 |
| **ML — Classifier** | scikit-learn `RandomForestClassifier` |
| **ML — Calibration** | scikit-learn `CalibratedClassifierCV` (sigmoid) |
| **ML — Serialisation** | joblib |
| **Environment** | python-dotenv |
| **Hosting** | Render |

---

## 📄 License

MIT — fork it, extend it, build on it.
