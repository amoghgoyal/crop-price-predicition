#!/usr/bin/env python3
"""
Crop Price Prediction ML Pipeline
Trains a RandomForest model per commodity and generates Buy/Sell/Hold signals.
Usage: python ml_pipeline.py
"""

import os, glob, json, warnings
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

PREDICTION_HORIZON = 7   # predict price N days ahead

# ── Commodity → Export product name map (partial) ───────────────────────────
COMMODITY_EXPORT_MAP = {
    'rice': 'NON BASMATI RICE', 'basmati': 'BASMATI RICE',
    'wheat': 'WHEAT', 'maize': 'MAIZE', 'onion': 'FRESH ONIONS',
    'groundnut': 'GROUNDNUTS', 'cashew': 'CASHEW KERNELS',
    'pulses': 'PULSES', 'millet': 'MILLET', 'honey': 'NATURAL HONEY',
    'grapes': 'FRESH GRAPES', 'mango': 'FRESH MANGOES',
    'buffalo': 'BUFFALO MEAT', 'guar': 'GUARGUM',
}

def get_export_key(name):
    name = str(name).lower()
    for k, v in COMMODITY_EXPORT_MAP.items():
        if k in name:
            return v
    return None

# ── 1. DATA LOADERS ──────────────────────────────────────────────────────────

def load_commodity_data():
    path = os.path.join(BASE_DIR, 'commodity-wise-report-for-last-5-years.csv')
    print("📊 Loading commodity price data (may take a few minutes)...")
    cols   = ['report_date','commodity','commodity_type','arrivals_tonnes',
              'min_price','max_price','modal_price']
    dtypes = {'commodity':'category','commodity_type':'category',
              'arrivals_tonnes':'float32','min_price':'float32',
              'max_price':'float32','modal_price':'float32'}
    chunks = []
    for chunk in pd.read_csv(path, usecols=cols, dtype=dtypes, chunksize=500_000):
        chunk['report_date'] = pd.to_datetime(chunk['report_date'], errors='coerce')
        chunk = chunk.dropna(subset=['report_date','modal_price'])
        chunk = chunk[chunk['modal_price'] > 0]
        agg = chunk.groupby(['report_date','commodity','commodity_type']).agg(
            modal_price   = ('modal_price',   'median'),
            min_price     = ('min_price',     'mean'),
            max_price     = ('max_price',     'mean'),
            arrivals_tonnes = ('arrivals_tonnes','sum'),
            n_markets     = ('modal_price',   'count'),
        ).reset_index()
        chunks.append(agg)
    df = pd.concat(chunks, ignore_index=True)
    df = df.groupby(['report_date','commodity','commodity_type']).agg(
        modal_price     = ('modal_price',     'mean'),
        min_price       = ('min_price',       'mean'),
        max_price       = ('max_price',       'mean'),
        arrivals_tonnes = ('arrivals_tonnes', 'sum'),
        n_markets       = ('n_markets',       'sum'),
    ).reset_index().sort_values('report_date')
    print(f"   ✓ {df['commodity'].nunique()} crops | {len(df):,} records | "
          f"{df['report_date'].min().date()} → {df['report_date'].max().date()}")
    return df

def load_crop_yield():
    path = os.path.join(BASE_DIR, 'crop_yield.csv')
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df['Crop'] = df['Crop'].str.strip().str.lower()
    return df

def load_rainfall():
    path = os.path.join(BASE_DIR, 'daily-rainfall-data-district-level.csv')
    print("🌧  Loading rainfall data...")
    cols = ['date','actual','normal','deviation']
    df = pd.read_csv(path, usecols=cols, chunksize=500_000)
    chunks = []
    for chunk in df:
        chunk['date'] = pd.to_datetime(chunk['date'], errors='coerce')
        for c in ['actual','normal','deviation']:
            chunk[c] = pd.to_numeric(chunk[c], errors='coerce')
        chunk['year']  = chunk['date'].dt.year
        chunk['month'] = chunk['date'].dt.month
        chunks.append(chunk.groupby(['year','month'])[['actual','normal','deviation']].mean().reset_index())
    rain = pd.concat(chunks).groupby(['year','month'])[['actual','normal','deviation']].mean().reset_index()
    return rain.rename(columns={'actual':'rain_actual','normal':'rain_normal','deviation':'rain_deviation'})

def load_annual_files(folder_pattern, qty_col='Qty(MT)', val_col='Rs(Crore)'):
    dfs = []
    for fp in sorted(glob.glob(folder_pattern)):
        yr = int(os.path.basename(fp).split('-')[0])
        tmp = pd.read_csv(fp)
        tmp.columns = tmp.columns.str.strip()
        tmp['year'] = yr
        dfs.append(tmp)
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    df['Product Name'] = df['Product Name'].str.strip().str.upper()
    return df[['year','Product Name', qty_col, val_col]]

# ── 2. FEATURE ENGINEERING ───────────────────────────────────────────────────

FEATURE_COLS = [
    'month','dayofweek','weekofyear','quarter','is_kharif','is_rabi',
    'price_lag_1','price_lag_3','price_lag_7','price_lag_14','price_lag_21','price_lag_30',
    'price_roll_mean_7','price_roll_mean_14','price_roll_mean_30',
    'price_roll_std_7','price_roll_std_30',
    'price_roll_min_7','price_roll_max_7',
    'price_change_7d','price_pct_change_7d',
    'price_spread','price_spread_lag7',
    'arrivals_lag_1','arrivals_lag_7','arrivals_roll_7','arrivals_roll_30',
    'n_markets_lag1',
    'rain_actual','rain_normal','rain_deviation',
    'export_qty','export_val','import_qty','import_val',
    'yield_val','area_val','production_val',
]

def make_features(series_df, rainfall, exports, imports, crop_yield):
    df = series_df.copy().sort_values('report_date').reset_index(drop=True)
    commodity_name = df['commodity'].iloc[0] if 'commodity' in df.columns else ''

    # Time
    df['year']       = df['report_date'].dt.year
    df['month']      = df['report_date'].dt.month
    df['dayofweek']  = df['report_date'].dt.dayofweek
    df['weekofyear'] = df['report_date'].dt.isocalendar().week.astype(int)
    df['quarter']    = df['report_date'].dt.quarter
    df['is_kharif']  = df['month'].isin([6,7,8,9,10]).astype(int)
    df['is_rabi']    = df['month'].isin([11,12,1,2,3]).astype(int)

    # Price lags & rolling
    for lag in [1,3,7,14,21,30]:
        df[f'price_lag_{lag}'] = df['modal_price'].shift(lag)
    for w in [7,14,30]:
        df[f'price_roll_mean_{w}'] = df['modal_price'].shift(1).rolling(w).mean()
    for w in [7,30]:
        df[f'price_roll_std_{w}']  = df['modal_price'].shift(1).rolling(w).std()
    df['price_roll_min_7'] = df['modal_price'].shift(1).rolling(7).min()
    df['price_roll_max_7'] = df['modal_price'].shift(1).rolling(7).max()

    # Momentum
    df['price_change_7d']     = df['modal_price'].shift(1) - df['modal_price'].shift(8)
    df['price_pct_change_7d'] = df['price_change_7d'] / (df['modal_price'].shift(8) + 1e-6)

    # Spread
    df['price_spread']      = df['max_price'] - df['min_price']
    df['price_spread_lag7'] = df['price_spread'].shift(7)

    # Arrivals
    for lag in [1,7]:
        df[f'arrivals_lag_{lag}'] = df['arrivals_tonnes'].shift(lag)
    df['arrivals_roll_7']  = df['arrivals_tonnes'].shift(1).rolling(7).mean()
    df['arrivals_roll_30'] = df['arrivals_tonnes'].shift(1).rolling(30).mean()
    df['n_markets_lag1']   = df['n_markets'].shift(1)

    # Rainfall join
    df = df.merge(rainfall, on=['year','month'], how='left')

    # Export / import join
    exp_key = get_export_key(commodity_name)
    if exp_key and len(exports):
        exp_sub = exports[exports['Product Name'] == exp_key].rename(
            columns={'Qty(MT)':'export_qty','Rs(Crore)':'export_val'})
        df = df.merge(exp_sub[['year','export_qty','export_val']], on='year', how='left')
        # import (loose match on first word)
        kw = exp_key.split()[0]
        imp_sub = imports[imports['Product Name'].str.startswith(kw, na=False)].rename(
            columns={'Qty(MT)':'import_qty','Rs(Crore)':'import_val'})
        imp_agg = imp_sub.groupby('year')[['import_qty','import_val']].sum().reset_index()
        df = df.merge(imp_agg, on='year', how='left')
    else:
        df[['export_qty','export_val','import_qty','import_val']] = np.nan

    # Crop yield join
    crop_key = str(commodity_name).lower().split('(')[0].strip()
    ysub = crop_yield[crop_yield['Crop'].str.contains(crop_key, na=False)]
    if len(ysub):
        yagg = ysub.groupby('Crop_Year').agg(
            yield_val=('Yield','mean'), area_val=('Area','sum'),
            production_val=('Production','sum')
        ).reset_index().rename(columns={'Crop_Year':'year'})
        df = df.merge(yagg, on='year', how='left')
    else:
        df[['yield_val','area_val','production_val']] = np.nan

    # Target
    df['target'] = df['modal_price'].shift(-PREDICTION_HORIZON)
    return df

# ── 3. TRAIN & PREDICT ───────────────────────────────────────────────────────

def train_model(commodity_name, feat_df):
    ready = feat_df[FEATURE_COLS + ['target','report_date','modal_price']].dropna()
    if len(ready) < 60:
        return None, None

    split = int(len(ready) * 0.8)
    X_tr, y_tr = ready.iloc[:split][FEATURE_COLS].fillna(-999), ready.iloc[:split]['target']
    X_te, y_te = ready.iloc[split:][FEATURE_COLS].fillna(-999), ready.iloc[split:]['target']

    model = RandomForestRegressor(
        n_estimators=200, max_depth=15, min_samples_leaf=5,
        n_jobs=-1, random_state=42)
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    mae = mean_absolute_error(y_te, y_pred)
    r2  = r2_score(y_te, y_pred)

    # Latest signal
    latest       = ready.iloc[[-1]]
    pred_price   = float(model.predict(latest[FEATURE_COLS].fillna(-999))[0])
    curr_price   = float(latest['modal_price'].values[0])
    curr_date    = pd.Timestamp(latest['report_date'].values[0])
    pct_change   = (pred_price - curr_price) / curr_price * 100

    if pct_change > 5:
        signal = 'BUY'
    elif pct_change < -3:
        signal = 'SELL'
    else:
        signal = 'HOLD'

    # Last 180 days for chart
    history = ready.tail(180)[['report_date','modal_price']].copy()
    history['report_date'] = history['report_date'].astype(str)

    result = dict(
        commodity      = str(commodity_name),
        current_price  = round(curr_price, 2),
        predicted_price= round(pred_price, 2),
        pct_change     = round(pct_change, 2),
        signal         = signal,
        mae            = round(mae, 2),
        r2             = round(r2, 4),
        current_date   = str(curr_date.date()),
        predicted_date = str((curr_date + timedelta(days=PREDICTION_HORIZON)).date()),
        history        = history.to_dict('records'),
        data_points    = len(ready),
    )
    return model, result

# ── 4. MAIN ──────────────────────────────────────────────────────────────────

def run_pipeline():
    print("\n🌾  CROP PRICE PREDICTION PIPELINE")
    print("=" * 55)

    commodity_df = load_commodity_data()
    crop_yield   = load_crop_yield()
    rainfall     = load_rainfall()

    exports = load_annual_files(
        os.path.join(BASE_DIR, 'export', 'export product wise', '*.csv'))
    imports = load_annual_files(
        os.path.join(BASE_DIR, 'import', 'import product wise', '*.csv'))

    commodities = commodity_df['commodity'].unique()
    print(f"\n🔄  Training models for {len(commodities)} crops...\n")

    predictions = []
    for i, commodity in enumerate(commodities, 1):
        print(f"  [{i:3d}/{len(commodities)}] {commodity:<45}", end='', flush=True)
        series  = commodity_df[commodity_df['commodity'] == commodity].copy()
        feat_df = make_features(series, rainfall, exports, imports, crop_yield)
        model, result = train_model(commodity, feat_df)
        if model is None:
            print("⚠  insufficient data")
            continue
        joblib.dump(model, os.path.join(MODELS_DIR, f"{commodity}.pkl"))
        predictions.append(result)
        print(f"{result['signal']:<5}  MAE={result['mae']:>8.1f}  R²={result['r2']:.3f}")

    output = dict(
        generated_at           = datetime.now().isoformat(),
        prediction_horizon_days= PREDICTION_HORIZON,
        total_crops            = len(predictions),
        predictions            = sorted(predictions, key=lambda x: abs(x['pct_change']), reverse=True),
    )
    out_path = os.path.join(BASE_DIR, 'predictions.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    sigs = [p['signal'] for p in predictions]
    print(f"\n{'='*55}")
    print(f"✅  Done!  🟢 BUY={sigs.count('BUY')}  🔴 SELL={sigs.count('SELL')}  🟡 HOLD={sigs.count('HOLD')}")
    print(f"📄  Saved → {out_path}")
    print(f"\n▶  Next step:  python app.py\n")


if __name__ == '__main__':
    run_pipeline()
