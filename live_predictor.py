#!/usr/bin/env python3
"""
CropSignal — Live Prediction Engine
Loads trained models and provides real-time predictions with Open-Meteo weather data.
"""

import os, json, joblib, re
import numpy as np
import requests
from datetime import datetime, timedelta

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Cache loaded models in memory
_model_cache = {}

# ── Indian states & UTs with capital coordinates (for weather) ────────────────
INDIA_REGIONS = {
    # ── States ──
    'andhra_pradesh':    {'lat': 16.51, 'lon': 80.52, 'name': 'Andhra Pradesh'},
    'arunachal_pradesh': {'lat': 27.10, 'lon': 93.62, 'name': 'Arunachal Pradesh'},
    'assam':             {'lat': 26.14, 'lon': 91.74, 'name': 'Assam'},
    'bihar':             {'lat': 25.60, 'lon': 85.10, 'name': 'Bihar'},
    'chhattisgarh':      {'lat': 21.25, 'lon': 81.63, 'name': 'Chhattisgarh'},
    'goa':               {'lat': 15.50, 'lon': 73.83, 'name': 'Goa'},
    'gujarat':           {'lat': 23.02, 'lon': 72.57, 'name': 'Gujarat'},
    'haryana':           {'lat': 30.73, 'lon': 76.78, 'name': 'Haryana'},
    'himachal_pradesh':  {'lat': 31.10, 'lon': 77.17, 'name': 'Himachal Pradesh'},
    'jharkhand':         {'lat': 23.36, 'lon': 85.33, 'name': 'Jharkhand'},
    'karnataka':         {'lat': 12.97, 'lon': 77.59, 'name': 'Karnataka'},
    'kerala':            {'lat': 8.52,  'lon': 76.94, 'name': 'Kerala'},
    'madhya_pradesh':    {'lat': 23.26, 'lon': 77.41, 'name': 'Madhya Pradesh'},
    'maharashtra':       {'lat': 19.08, 'lon': 72.88, 'name': 'Maharashtra'},
    'manipur':           {'lat': 24.82, 'lon': 93.95, 'name': 'Manipur'},
    'meghalaya':         {'lat': 25.57, 'lon': 91.88, 'name': 'Meghalaya'},
    'mizoram':           {'lat': 23.73, 'lon': 92.72, 'name': 'Mizoram'},
    'nagaland':          {'lat': 25.67, 'lon': 94.12, 'name': 'Nagaland'},
    'odisha':            {'lat': 20.30, 'lon': 85.83, 'name': 'Odisha'},
    'punjab':            {'lat': 30.73, 'lon': 76.78, 'name': 'Punjab'},
    'rajasthan':         {'lat': 26.91, 'lon': 75.79, 'name': 'Rajasthan'},
    'sikkim':            {'lat': 27.33, 'lon': 88.62, 'name': 'Sikkim'},
    'tamil_nadu':        {'lat': 13.08, 'lon': 80.27, 'name': 'Tamil Nadu'},
    'telangana':         {'lat': 17.39, 'lon': 78.49, 'name': 'Telangana'},
    'tripura':           {'lat': 23.83, 'lon': 91.28, 'name': 'Tripura'},
    'uttar_pradesh':     {'lat': 26.85, 'lon': 80.95, 'name': 'Uttar Pradesh'},
    'uttarakhand':       {'lat': 30.32, 'lon': 78.03, 'name': 'Uttarakhand'},
    'west_bengal':       {'lat': 22.57, 'lon': 88.36, 'name': 'West Bengal'},
    # ── Union Territories ──
    'delhi':             {'lat': 28.61, 'lon': 77.23, 'name': 'Delhi'},
    'jammu_kashmir':     {'lat': 34.08, 'lon': 74.80, 'name': 'Jammu & Kashmir'},
    'ladakh':            {'lat': 34.15, 'lon': 77.58, 'name': 'Ladakh'},
    'chandigarh':        {'lat': 30.73, 'lon': 76.78, 'name': 'Chandigarh'},
    'puducherry':        {'lat': 11.94, 'lon': 79.83, 'name': 'Puducherry'},
    'andaman_nicobar':   {'lat': 11.67, 'lon': 92.74, 'name': 'Andaman & Nicobar'},
    'dadra_nagar_haveli': {'lat': 20.27, 'lon': 73.01, 'name': 'Dadra & Nagar Haveli'},
    'lakshadweep':       {'lat': 10.57, 'lon': 72.64, 'name': 'Lakshadweep'},
    # ── Default ──
    'average':           {'lat': 22.50, 'lon': 78.50, 'name': 'All India (Avg)'},
}


def fetch_weather(region='average'):
    """Fetch current weather using wttr.in (primary) or Open-Meteo (fallback).
    wttr.in is fast (~1s) and works reliably from India."""
    coords = INDIA_REGIONS.get(region, INDIA_REGIONS['average'])
    region_name = coords['name']

    # ── PRIMARY: wttr.in (fast, reliable from India) ──
    try:
        location = region_name.replace(' ', '+').replace('&', 'and')
        resp = requests.get(f'https://wttr.in/{location}?format=j1', timeout=(8, 15))
        resp.raise_for_status()
        data = resp.json()

        cc = data.get('current_condition', [{}])[0]
        # Get 7-day forecast for rainfall average
        forecast = data.get('weather', [])
        daily_precip = []
        daily_temp_max = []
        daily_temp_min = []
        daily_dates = []
        for day in forecast:
            daily_precip.append(float(day.get('totalSnow_cm', 0)) * 10 +
                                sum(float(h.get('precipMM', 0)) for h in day.get('hourly', [])) / max(len(day.get('hourly', [])), 1) * 24)
            daily_temp_max.append(float(day.get('maxtempC', 0)))
            daily_temp_min.append(float(day.get('mintempC', 0)))
            daily_dates.append(day.get('date', ''))

        # Simpler rain calculation: use precipMM from current + daily avg
        precip_today = float(cc.get('precipMM', 0))
        avg_rain = sum(daily_precip) / max(len(daily_precip), 1) if daily_precip else precip_today

        # Map weather description to WMO code (approximate)
        desc = cc.get('weatherDesc', [{}])[0].get('value', 'Unknown')

        return {
            'region':       region_name,
            'temperature':  float(cc.get('temp_C', 25)),
            'humidity':     int(cc.get('humidity', 60)),
            'precipitation': precip_today,
            'wind_speed':   float(cc.get('windspeedKmph', 5)),
            'weather_code': int(cc.get('weatherCode', 0)),
            'rain_7d_avg':  round(avg_rain, 2),
            'daily_precip': daily_precip,
            'daily_temp_max': daily_temp_max,
            'daily_temp_min': daily_temp_min,
            'daily_dates':   daily_dates,
            'description':  desc,
            'fetched_at':   datetime.now().isoformat(),
            'source':       'wttr.in',
        }
    except Exception as e_wttr:
        pass  # Fall through to Open-Meteo

    # ── FALLBACK: Open-Meteo ──
    try:
        resp = requests.get('https://api.open-meteo.com/v1/forecast', params={
            'latitude':  coords['lat'],
            'longitude': coords['lon'],
            'current':   'temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,weather_code',
            'daily':     'precipitation_sum,temperature_2m_max,temperature_2m_min',
            'timezone':  'Asia/Kolkata',
            'forecast_days': 7,
        }, timeout=(10, 30))
        resp.raise_for_status()
        data = resp.json()

        current = data.get('current', {})
        daily   = data.get('daily', {})
        precip_7d = daily.get('precipitation_sum', [0])
        avg_rain  = sum(precip_7d) / max(len(precip_7d), 1)

        return {
            'region':       region_name,
            'temperature':  current.get('temperature_2m', 25),
            'humidity':     current.get('relative_humidity_2m', 60),
            'precipitation': current.get('precipitation', 0),
            'wind_speed':   current.get('wind_speed_10m', 5),
            'weather_code': current.get('weather_code', 0),
            'rain_7d_avg':  round(avg_rain, 2),
            'daily_precip': precip_7d,
            'daily_temp_max': daily.get('temperature_2m_max', []),
            'daily_temp_min': daily.get('temperature_2m_min', []),
            'daily_dates':   daily.get('time', []),
            'fetched_at':   datetime.now().isoformat(),
            'source':       'open-meteo',
        }
    except Exception:
        pass

    # ── All APIs failed ──
    return {
        'region': region_name,
        'error': 'All weather APIs unavailable',
        'temperature': 28, 'humidity': 65, 'precipitation': 5,
        'wind_speed': 8, 'rain_7d_avg': 5.0,
        'fetched_at': datetime.now().isoformat(),
        'source': 'fallback',
    }


def get_weather_description(code):
    """Convert WMO weather code to description."""
    codes = {
        0: 'Clear sky', 1: 'Mainly clear', 2: 'Partly cloudy', 3: 'Overcast',
        45: 'Foggy', 48: 'Rime fog', 51: 'Light drizzle', 53: 'Moderate drizzle',
        55: 'Dense drizzle', 61: 'Slight rain', 63: 'Moderate rain', 65: 'Heavy rain',
        71: 'Slight snow', 73: 'Moderate snow', 75: 'Heavy snow',
        80: 'Slight rain showers', 81: 'Moderate rain showers', 82: 'Violent rain showers',
        95: 'Thunderstorm', 96: 'Thunderstorm with hail', 99: 'Thunderstorm with heavy hail',
    }
    return codes.get(code, 'Unknown')


def load_model(commodity_name):
    """Load a trained model from disk (cached)."""
    if commodity_name in _model_cache:
        return _model_cache[commodity_name]

    safe_commodity = re.sub(r'[\\/*?:"<>|]', '_', commodity_name)
    pkl_path = os.path.join(MODELS_DIR, f"{safe_commodity}.pkl")
    if os.path.exists(pkl_path):
        model = joblib.load(pkl_path)
        _model_cache[commodity_name] = model
        return model

    return None


def get_available_crops():
    """List all crops with trained models."""
    crops = []
    if os.path.exists(MODELS_DIR):
        for f in os.listdir(MODELS_DIR):
            if f.endswith('.pkl'):
                crops.append(f.replace('.pkl', ''))
    return sorted(crops)


def _compute_price_features(history):
    """
    Compute real price lags and rolling statistics from the 180-day history array.
    This replaces the crude 'set everything to current_price' approach.
    """
    if not history or len(history) < 2:
        return {}

    # Extract prices in chronological order (history is already sorted)
    prices = [h['modal_price'] for h in history]
    n = len(prices)

    features = {}

    # ── Price lags (from the END of the array = most recent) ──
    for lag in [1, 3, 7, 14, 21, 30]:
        idx = n - lag
        features[f'price_lag_{lag}'] = prices[idx] if idx >= 0 else prices[0]

    # ── Rolling means ──
    for window in [7, 14, 30]:
        start = max(0, n - window)
        chunk = prices[start:]
        features[f'price_roll_mean_{window}'] = sum(chunk) / len(chunk)

    # ── Rolling std ──
    for window in [7, 30]:
        start = max(0, n - window)
        chunk = prices[start:]
        mean = sum(chunk) / len(chunk)
        variance = sum((x - mean) ** 2 for x in chunk) / max(len(chunk) - 1, 1)
        features[f'price_roll_std_{window}'] = variance ** 0.5

    # ── Rolling min/max (7-day) ──
    last_7 = prices[max(0, n - 7):]
    features['price_roll_min_7'] = min(last_7)
    features['price_roll_max_7'] = max(last_7)

    # ── Momentum: 7-day price change ──
    if n >= 8:
        features['price_change_7d'] = prices[-1] - prices[-8]
        features['price_pct_change_7d'] = features['price_change_7d'] / (prices[-8] + 1e-6)
    else:
        features['price_change_7d'] = 0
        features['price_pct_change_7d'] = 0

    # ── Price spread approximation (use recent volatility) ──
    last_7_range = max(last_7) - min(last_7)
    features['price_spread'] = last_7_range
    if n >= 14:
        prev_7 = prices[max(0, n - 14):max(0, n - 7)]
        features['price_spread_lag7'] = max(prev_7) - min(prev_7) if prev_7 else last_7_range
    else:
        features['price_spread_lag7'] = last_7_range

    return features


# Weather cache (avoid hammering API on repeated predictions)
_weather_cache = {}  # keyed by region
_WEATHER_CACHE_TTL = 300  # 5 minutes


def _get_cached_weather(region='average'):
    """Fetch weather with 5-minute cache."""
    now = datetime.now()
    cached = _weather_cache.get(region)
    if (cached is not None and
            (now - cached['fetched_at']).total_seconds() < _WEATHER_CACHE_TTL):
        return cached['data']

    data = fetch_weather(region)
    _weather_cache[region] = {'data': data, 'fetched_at': now}
    return data


def predict_live(commodity_name, rainfall=None, month=None, region='average', predictions_data=None):
    """
    Make a live prediction using a trained model and real-time data.

    Accuracy improvements over the basic version:
    1. Uses actual date/time for all temporal features
    2. Computes REAL price lags & rolling stats from 180-day history
    3. Fetches live weather from Open-Meteo for the selected STATE
    4. Calculates rainfall deviation from historical normal
    5. Caches weather data per-region to avoid API spam

    Args:
        commodity_name: Name of the crop
        rainfall: Override rainfall value (mm), None = use live weather for state
        month: Override month (1-12), None = use current month
        region: State/UT key from INDIA_REGIONS, used to fetch local weather
        predictions_data: Loaded predictions.json data for baseline features
    """
    model = load_model(commodity_name)
    if model is None:
        return {'error': f'No trained model found for "{commodity_name}"'}

    # Find baseline data from predictions.json
    baseline = None
    if predictions_data:
        for p in predictions_data.get('predictions', []):
            if p['commodity'].lower() == commodity_name.lower():
                baseline = p
                break

    if baseline is None:
        return {'error': f'No baseline data for "{commodity_name}". Run ml_pipeline.py first.'}

    # ── Current date/time ──
    now = datetime.now()
    m = month if month else now.month
    # If user overrides month, adjust weekday/week to match mid-month of that month
    if month and month != now.month:
        reference_date = datetime(now.year, month, 15)
    else:
        reference_date = now

    curr_price = baseline['current_price']
    history = baseline.get('history', [])

    # ── 1. Time features (from ACTUAL current date) ──
    features = {
        'month': m,
        'dayofweek': reference_date.weekday(),
        'weekofyear': reference_date.isocalendar()[1],
        'quarter': (m - 1) // 3 + 1,
        'is_kharif': 1 if m in [6, 7, 8, 9, 10] else 0,
        'is_rabi': 1 if m in [11, 12, 1, 2, 3] else 0,
    }

    # ── 2. Price features (computed from REAL history) ──
    price_feats = _compute_price_features(history)
    features.update(price_feats)

    # Fill any missing price features with current_price fallback
    price_defaults = {
        'price_lag_1': curr_price, 'price_lag_3': curr_price,
        'price_lag_7': curr_price, 'price_lag_14': curr_price,
        'price_lag_21': curr_price, 'price_lag_30': curr_price,
        'price_roll_mean_7': curr_price, 'price_roll_mean_14': curr_price,
        'price_roll_mean_30': curr_price,
        'price_roll_std_7': 0, 'price_roll_std_30': 0,
        'price_roll_min_7': curr_price, 'price_roll_max_7': curr_price,
        'price_change_7d': 0, 'price_pct_change_7d': 0,
        'price_spread': 0, 'price_spread_lag7': 0,
    }
    for k, default in price_defaults.items():
        if k not in features:
            features[k] = default

    # ── 3. Arrivals (not available live — mark as missing) ──
    features['arrivals_lag_1'] = -999
    features['arrivals_lag_7'] = -999
    features['arrivals_roll_7'] = -999
    features['arrivals_roll_30'] = -999
    features['n_markets_lag1'] = -999

    # ── 4. Rainfall (from live weather or user override) ──
    weather = _get_cached_weather(region or 'average')

    if rainfall is not None:
        # User provided explicit rainfall
        features['rain_actual'] = rainfall
        # Estimate normal rainfall for the month (rough India average mm/day by month)
        monthly_normal = {
            1: 1.0, 2: 1.2, 3: 1.5, 4: 2.0, 5: 3.5, 6: 8.0,
            7: 12.0, 8: 11.0, 9: 8.5, 10: 4.0, 11: 1.5, 12: 0.8,
        }
        normal = monthly_normal.get(m, 5.0)
        features['rain_normal'] = normal
        features['rain_deviation'] = rainfall - normal
    else:
        # Use live Open-Meteo weather data
        features['rain_actual'] = weather.get('rain_7d_avg', -999)
        # Open-Meteo doesn't give historical normals, estimate from month
        monthly_normal = {
            1: 1.0, 2: 1.2, 3: 1.5, 4: 2.0, 5: 3.5, 6: 8.0,
            7: 12.0, 8: 11.0, 9: 8.5, 10: 4.0, 11: 1.5, 12: 0.8,
        }
        normal = monthly_normal.get(m, 5.0)
        features['rain_normal'] = normal
        if features['rain_actual'] != -999:
            features['rain_deviation'] = features['rain_actual'] - normal
        else:
            features['rain_deviation'] = 0

    # ── 5. Trade & yield (not available in real-time) ──
    features['export_qty'] = -999
    features['export_val'] = -999
    features['import_qty'] = -999
    features['import_val'] = -999
    features['yield_val'] = -999
    features['area_val'] = -999
    features['production_val'] = -999

    # ── Build feature vector and predict ──
    from ml_pipeline import FEATURE_COLS
    X = np.array([[features.get(f, -999) for f in FEATURE_COLS]])

    pred_pct = float(model.predict(X)[0])
    pred_price = curr_price * (1 + pred_pct)
    pct_change = pred_pct * 100

    if pct_change > 5:
        signal = 'BUY'
    elif pct_change < -3:
        signal = 'SELL'
    else:
        signal = 'HOLD'

    residual_std = baseline.get('residual_std', curr_price * 0.05)

    return {
        'commodity': commodity_name,
        'current_price': curr_price,
        'predicted_price': round(pred_price, 2),
        'pct_change': round(pct_change, 2),
        'signal': signal,
        'confidence_low': round(pred_price - 1.96 * residual_std, 2),
        'confidence_high': round(pred_price + 1.96 * residual_std, 2),
        'inputs': {
            'rainfall': round(features['rain_actual'], 2) if features['rain_actual'] != -999 else None,
            'rain_normal': round(features['rain_normal'], 1),
            'rain_deviation': round(features['rain_deviation'], 2),
            'month': m,
            'month_name': ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][m-1],
            'season': 'Kharif' if features['is_kharif'] else ('Rabi' if features['is_rabi'] else 'Zaid'),
            'day_of_week': ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][features['dayofweek']],
            'temperature': weather.get('temperature'),
            'weather': weather.get('description', get_weather_description(weather.get('weather_code', 0))),
            'region': weather.get('region', INDIA_REGIONS.get(region or 'average', {}).get('name', 'India')),
        },
        'features_used': {
            'price_lag_1': round(features['price_lag_1'], 2),
            'price_lag_7': round(features['price_lag_7'], 2),
            'price_lag_30': round(features['price_lag_30'], 2),
            'roll_mean_7': round(features['price_roll_mean_7'], 2),
            'roll_mean_30': round(features['price_roll_mean_30'], 2),
            'momentum_7d': round(features['price_change_7d'], 2),
            'history_points_used': len(history),
        },
        'predicted_at': now.isoformat(),
        'prediction_for': str((reference_date + timedelta(days=7)).date()),
    }
