#!/usr/bin/env python3
"""
Flask dashboard server for crop price predictions.
Usage: python app.py
Then open http://localhost:5000
"""

import os, json
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_predictions():
    path = os.path.join(BASE_DIR, 'predictions.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/signals')
def signals():
    data = load_predictions()
    if not data:
        return jsonify({'error': 'Run ml_pipeline.py first to generate predictions.'}), 404
    return jsonify(data)


@app.route('/api/crop/<path:name>')
def crop_detail(name):
    data = load_predictions()
    if not data:
        return jsonify({'error': 'No predictions found'}), 404
    match = next((p for p in data['predictions'] if p['commodity'].lower() == name.lower()), None)
    if not match:
        return jsonify({'error': f'Crop "{name}" not found'}), 404
    return jsonify(match)


@app.route('/api/search')
def search():
    data = load_predictions()
    if not data:
        return jsonify([])
    q = request.args.get('q', '').lower()
    sig = request.args.get('signal', '').upper()
    results = data['predictions']
    if q:
        results = [p for p in results if q in p['commodity'].lower()]
    if sig in ('BUY', 'SELL', 'HOLD'):
        results = [p for p in results if p['signal'] == sig]
    return jsonify(results)


if __name__ == '__main__':
    print("🌾  Crop Price Dashboard")
    print("   Open → http://localhost:5000")
    data = load_predictions()
    if data:
        print(f"   Loaded {data['total_crops']} crops  (generated {data['generated_at'][:19]})")
    else:
        print("   ⚠  No predictions.json found — run ml_pipeline.py first")
    app.run(debug=True, port=5000)
