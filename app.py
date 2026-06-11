#!/usr/bin/env python3
"""
CropSignal — Flask Dashboard Server
Serves the premium prediction dashboard and real-time API endpoints.

Usage: python app.py
Then open http://localhost:5000
"""

import os, json
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from live_predictor import fetch_weather, get_weather_description, predict_live, get_available_crops, INDIA_REGIONS

app = Flask(__name__)
CORS(app)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


import time

def load_predictions():
    path = os.path.join(BASE_DIR, 'predictions.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

_live_cache = {'time': 0, 'data': None}

def get_live_predictions():
    """Returns predictions updated with real-time ML engine results based on current date/weather."""
    global _live_cache
    now = time.time()
    # Cache live bulk predictions for 1 hour
    if _live_cache['data'] and (now - _live_cache['time']) < 3600:
        return _live_cache['data']

    data = load_predictions()
    if not data:
        return None

    for p in data['predictions']:
        try:
            live = predict_live(p['commodity'], predictions_data=data)
            if 'error' not in live:
                p['current_price'] = live['current_price']
                p['predicted_price'] = live['predicted_price']
                p['pct_change'] = live['pct_change']
                p['signal'] = live['signal']
                p['current_date'] = live['predicted_at'].split('T')[0]
                p['predicted_date'] = live['prediction_for']
        except Exception:
            pass
            
    # Sort again by pct_change (descending magnitude)
    data['predictions'] = sorted(data['predictions'], key=lambda x: abs(x['pct_change']), reverse=True)
    
    _live_cache['data'] = data
    _live_cache['time'] = now
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')


# ═══════════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/signals')
def signals():
    """Get all predictions with summary stats."""
    data = get_live_predictions()
    if not data:
        return jsonify({'error': 'Run ml_pipeline.py first to generate predictions.'}), 404
    return jsonify(data)


@app.route('/api/crop/<path:name>')
def crop_detail(name):
    """Get detailed prediction for a specific crop."""
    data = get_live_predictions()
    if not data:
        return jsonify({'error': 'No predictions found'}), 404
    match = next(
        (p for p in data['predictions'] if p['commodity'].lower() == name.lower()),
        None
    )
    if not match:
        return jsonify({'error': f'Crop "{name}" not found'}), 404
    return jsonify(match)


@app.route('/api/search')
def search():
    """Search and filter crops."""
    data = get_live_predictions()
    if not data:
        return jsonify([])
    q   = request.args.get('q', '').lower()
    sig = request.args.get('signal', '').upper()
    results = data['predictions']
    if q:
        results = [p for p in results if q in p['commodity'].lower()]
    if sig in ('BUY', 'SELL', 'HOLD'):
        results = [p for p in results if p['signal'] == sig]
    return jsonify(results)


@app.route('/api/top-movers')
def top_movers():
    """Get crops with the biggest predicted price changes."""
    data = get_live_predictions()
    if not data:
        return jsonify([])
    n = int(request.args.get('n', 10))
    preds = data['predictions']
    top_buy  = sorted([p for p in preds if p['signal'] == 'BUY'],
                      key=lambda x: x['pct_change'], reverse=True)[:n]
    top_sell = sorted([p for p in preds if p['signal'] == 'SELL'],
                      key=lambda x: x['pct_change'])[:n]
    return jsonify({'top_buy': top_buy, 'top_sell': top_sell})


@app.route('/api/stats')
def stats():
    """Get aggregate dashboard statistics."""
    data = get_live_predictions()
    if not data:
        return jsonify({'error': 'No data'}), 404
    preds = data['predictions']
    buy_count  = sum(1 for p in preds if p['signal'] == 'BUY')
    sell_count = sum(1 for p in preds if p['signal'] == 'SELL')
    hold_count = sum(1 for p in preds if p['signal'] == 'HOLD')
    avg_r2     = sum(p.get('r2', 0) for p in preds) / max(len(preds), 1)
    avg_change = sum(p['pct_change'] for p in preds) / max(len(preds), 1)
    return jsonify({
        'total':      len(preds),
        'buy':        buy_count,
        'sell':       sell_count,
        'hold':       hold_count,
        'avg_r2':     round(avg_r2, 4),
        'avg_change': round(avg_change, 2),
        'generated':  data.get('generated_at', ''),
        'model_type': data.get('model_type', 'Unknown'),
    })


@app.route('/api/weather')
def weather():
    """Get live weather data."""
    region = request.args.get('region', 'average')
    data = fetch_weather(region)
    # Add description if not already present (wttr.in provides it, Open-Meteo needs lookup)
    if 'description' not in data and 'weather_code' in data:
        data['description'] = get_weather_description(data['weather_code'])
    return jsonify(data)


@app.route('/api/predict', methods=['POST'])
def predict():
    """Live prediction with custom inputs."""
    body = request.get_json(force=True)
    crop     = body.get('crop', '')
    rainfall = body.get('rainfall')
    month    = body.get('month')
    region   = body.get('region', 'average')

    if not crop:
        return jsonify({'error': 'Crop name is required'}), 400

    if rainfall is not None:
        rainfall = float(rainfall)
    if month is not None:
        month = int(month)

    data = load_predictions()
    result = predict_live(crop, rainfall=rainfall, month=month, region=region, predictions_data=data)

    if 'error' in result:
        return jsonify(result), 404
    return jsonify(result)


@app.route('/api/states')
def states():
    """Get list of available Indian states/UTs for the dropdown."""
    state_list = []
    for key, info in sorted(INDIA_REGIONS.items(), key=lambda x: x[1]['name']):
        state_list.append({'key': key, 'name': info['name']})
    return jsonify(state_list)


@app.route('/api/crops-list')
def crops_list():
    """Get list of available crops with trained models."""
    data = load_predictions()
    if not data:
        return jsonify([])
    return jsonify([p['commodity'] for p in data['predictions']])


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n🌾  CropSignal — Live Dashboard")
    print("   Open → http://localhost:5000")
    data = load_predictions()
    if data:
        print(f"   📊 Loaded {data['total_crops']} crops (generated {data['generated_at'][:19]})")
        print(f"   🤖 Model: {data.get('model_type', 'Unknown')}")
    else:
        print("   ⚠  No predictions.json found — run: python ml_pipeline.py --top 30")
    print()
    app.run(debug=True, port=5000)
