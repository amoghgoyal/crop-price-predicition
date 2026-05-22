"""Run this script once to generate CropSignal_GPU.ipynb"""
import json

def cell(source, cell_type="code"):
    c = {"cell_type": cell_type, "metadata": {}, "source": source}
    if cell_type == "code":
        c["outputs"] = []
        c["execution_count"] = None
    return c

cells = []

# ── 0. Title ─────────────────────────────────────────────────────────────────
cells.append(cell([
    "# 🌾 CropSignal — GPU-Accelerated Crop Price Prediction\n",
    "> Trains **XGBoost (CUDA)** models on every crop using Mandi price data, rainfall,\n",
    "> crop yield, and export/import trade data. Outputs `predictions.json` with\n",
    "> **Buy / Hold / Sell** signals for 7-day-ahead price forecasts.\n",
    "\n",
    "**Runtime:** `Runtime → Change runtime type → T4 GPU`",
], "markdown"))

# ── 1. Install ────────────────────────────────────────────────────────────────
cells.append(cell([
    "# ── Install dependencies ───────────────────────────────────────────────\n",
    "!pip install -q xgboost scikit-learn tqdm\n",
    "import xgboost as xgb\n",
    "print('XGBoost', xgb.__version__)\n",
    "# Verify GPU\n",
    "!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'No GPU found — switch runtime to GPU'",
]))

# ── 2. Mount Drive ────────────────────────────────────────────────────────────
cells.append(cell([
    "# ── Mount Google Drive (upload your CSV files there first) ─────────────\n",
    "from google.colab import drive\n",
    "drive.mount('/content/drive')\n",
    "\n",
    "# 📁 EDIT THIS PATH to where you put the project folder in Drive\n",
    "BASE_DIR = '/content/drive/MyDrive/CropSignal'\n",
    "print('BASE_DIR =', BASE_DIR)",
]))

# ── 2b. Alternative: direct upload ───────────────────────────────────────────
cells.append(cell([
    "# ── ALTERNATIVE: Upload files directly (skip if using Drive) ───────────\n",
    "# Uncomment and run this cell if you prefer direct upload:\n",
    "# from google.colab import files\n",
    "# import os\n",
    "# BASE_DIR = '/content/CropSignal'\n",
    "# os.makedirs(BASE_DIR, exist_ok=True)\n",
    "# uploaded = files.upload()  # upload all CSVs here\n",
    "# for fn, data in uploaded.items():\n",
    "#     with open(os.path.join(BASE_DIR, fn), 'wb') as f:\n",
    "#         f.write(data)\n",
    "print('Using BASE_DIR:', BASE_DIR)",
]))

# ── 3. Imports ────────────────────────────────────────────────────────────────
cells.append(cell([
    "import os, glob, json, warnings\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import xgboost as xgb\n",
    "from sklearn.metrics import mean_absolute_error, r2_score\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from datetime import datetime, timedelta\n",
    "from tqdm.notebook import tqdm\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "MODELS_DIR         = os.path.join(BASE_DIR, 'models')\n",
    "PREDICTION_HORIZON = 7\n",
    "os.makedirs(MODELS_DIR, exist_ok=True)\n",
    "print('All imports OK')",
]))

# ── 4. Commodity → export map ─────────────────────────────────────────────────
cells.append(cell([
    "COMMODITY_EXPORT_MAP = {\n",
    "    'rice':'NON BASMATI RICE','basmati':'BASMATI RICE','wheat':'WHEAT',\n",
    "    'maize':'MAIZE','onion':'FRESH ONIONS','groundnut':'GROUNDNUTS',\n",
    "    'cashew':'CASHEW KERNELS','pulses':'PULSES','millet':'MILLET',\n",
    "    'honey':'NATURAL HONEY','grapes':'FRESH GRAPES','mango':'FRESH MANGOES',\n",
    "    'buffalo':'BUFFALO MEAT','guar':'GUARGUM',\n",
    "}\n",
    "\n",
    "def get_export_key(name):\n",
    "    name = str(name).lower()\n",
    "    for k, v in COMMODITY_EXPORT_MAP.items():\n",
    "        if k in name: return v\n",
    "    return None",
]))

# ── 5. Load commodity data ─────────────────────────────────────────────────────
cells.append(cell([
    "# ── Load commodity price data (15M rows, chunked) ──────────────────────\n",
    "def load_commodity_data():\n",
    "    path  = os.path.join(BASE_DIR, 'commodity-wise-report-for-last-5-years.csv')\n",
    "    cols  = ['report_date','commodity','commodity_type','arrivals_tonnes',\n",
    "              'min_price','max_price','modal_price']\n",
    "    dtypes = {'commodity':'category','commodity_type':'category',\n",
    "               'arrivals_tonnes':'float32','min_price':'float32',\n",
    "               'max_price':'float32','modal_price':'float32'}\n",
    "    chunks = []\n",
    "    for chunk in pd.read_csv(path, usecols=cols, dtype=dtypes, chunksize=500_000):\n",
    "        chunk['report_date'] = pd.to_datetime(chunk['report_date'], errors='coerce')\n",
    "        chunk = chunk.dropna(subset=['report_date','modal_price'])\n",
    "        chunk = chunk[chunk['modal_price'] > 0]\n",
    "        agg = chunk.groupby(['report_date','commodity','commodity_type']).agg(\n",
    "            modal_price=('modal_price','median'), min_price=('min_price','mean'),\n",
    "            max_price=('max_price','mean'), arrivals_tonnes=('arrivals_tonnes','sum'),\n",
    "            n_markets=('modal_price','count')\n",
    "        ).reset_index()\n",
    "        chunks.append(agg)\n",
    "    df = pd.concat(chunks, ignore_index=True)\n",
    "    df = df.groupby(['report_date','commodity','commodity_type']).agg(\n",
    "        modal_price=('modal_price','mean'), min_price=('min_price','mean'),\n",
    "        max_price=('max_price','mean'), arrivals_tonnes=('arrivals_tonnes','sum'),\n",
    "        n_markets=('n_markets','sum')\n",
    "    ).reset_index().sort_values('report_date')\n",
    "    print(f'✓ {df[\"commodity\"].nunique()} crops | {len(df):,} records')\n",
    "    return df\n",
    "\n",
    "commodity_df = load_commodity_data()",
]))

# ── 6. Load supporting data ───────────────────────────────────────────────────
cells.append(cell([
    "# ── Load crop yield, rainfall, exports, imports ─────────────────────────\n",
    "def load_crop_yield():\n",
    "    df = pd.read_csv(os.path.join(BASE_DIR, 'crop_yield.csv'))\n",
    "    df.columns = df.columns.str.strip()\n",
    "    df['Crop'] = df['Crop'].str.strip().str.lower()\n",
    "    return df\n",
    "\n",
    "def load_rainfall():\n",
    "    cols = ['date','actual','normal','deviation']\n",
    "    chunks = []\n",
    "    for chunk in pd.read_csv(os.path.join(BASE_DIR, 'daily-rainfall-data-district-level.csv'),\n",
    "                             usecols=cols, chunksize=500_000):\n",
    "        chunk['date'] = pd.to_datetime(chunk['date'], errors='coerce')\n",
    "        for c in ['actual','normal','deviation']:\n",
    "            chunk[c] = pd.to_numeric(chunk[c], errors='coerce')\n",
    "        chunk['year'] = chunk['date'].dt.year\n",
    "        chunk['month'] = chunk['date'].dt.month\n",
    "        chunks.append(chunk.groupby(['year','month'])[['actual','normal','deviation']].mean().reset_index())\n",
    "    rain = pd.concat(chunks).groupby(['year','month'])[['actual','normal','deviation']].mean().reset_index()\n",
    "    return rain.rename(columns={'actual':'rain_actual','normal':'rain_normal','deviation':'rain_deviation'})\n",
    "\n",
    "def load_annual(folder_pattern):\n",
    "    dfs = []\n",
    "    for fp in sorted(glob.glob(folder_pattern)):\n",
    "        yr = int(os.path.basename(fp).split('-')[0])\n",
    "        tmp = pd.read_csv(fp); tmp.columns = tmp.columns.str.strip(); tmp['year'] = yr\n",
    "        dfs.append(tmp)\n",
    "    if not dfs: return pd.DataFrame()\n",
    "    df = pd.concat(dfs, ignore_index=True)\n",
    "    df['Product Name'] = df['Product Name'].str.strip().str.upper()\n",
    "    return df\n",
    "\n",
    "crop_yield = load_crop_yield()\n",
    "rainfall   = load_rainfall()\n",
    "exports    = load_annual(os.path.join(BASE_DIR, 'export', 'export product wise', '*.csv'))\n",
    "imports    = load_annual(os.path.join(BASE_DIR, 'import', 'import product wise', '*.csv'))\n",
    "print('✓ Supporting data loaded')",
]))

# ── 7. Feature engineering ────────────────────────────────────────────────────
cells.append(cell([
    "FEATURE_COLS = [\n",
    "    'month','dayofweek','weekofyear','quarter','is_kharif','is_rabi',\n",
    "    'price_lag_1','price_lag_3','price_lag_7','price_lag_14','price_lag_21','price_lag_30',\n",
    "    'price_roll_mean_7','price_roll_mean_14','price_roll_mean_30',\n",
    "    'price_roll_std_7','price_roll_std_30','price_roll_min_7','price_roll_max_7',\n",
    "    'price_change_7d','price_pct_change_7d','price_spread','price_spread_lag7',\n",
    "    'arrivals_lag_1','arrivals_lag_7','arrivals_roll_7','arrivals_roll_30','n_markets_lag1',\n",
    "    'rain_actual','rain_normal','rain_deviation',\n",
    "    'export_qty','export_val','import_qty','import_val',\n",
    "    'yield_val','area_val','production_val',\n",
    "]\n",
    "\n",
    "def make_features(series_df):\n",
    "    df = series_df.copy().sort_values('report_date').reset_index(drop=True)\n",
    "    name = str(df['commodity'].iloc[0])\n",
    "    df['year']       = df['report_date'].dt.year\n",
    "    df['month']      = df['report_date'].dt.month\n",
    "    df['dayofweek']  = df['report_date'].dt.dayofweek\n",
    "    df['weekofyear'] = df['report_date'].dt.isocalendar().week.astype(int)\n",
    "    df['quarter']    = df['report_date'].dt.quarter\n",
    "    df['is_kharif']  = df['month'].isin([6,7,8,9,10]).astype(int)\n",
    "    df['is_rabi']    = df['month'].isin([11,12,1,2,3]).astype(int)\n",
    "    for lag in [1,3,7,14,21,30]:\n",
    "        df[f'price_lag_{lag}'] = df['modal_price'].shift(lag)\n",
    "    for w in [7,14,30]:\n",
    "        df[f'price_roll_mean_{w}'] = df['modal_price'].shift(1).rolling(w).mean()\n",
    "    for w in [7,30]:\n",
    "        df[f'price_roll_std_{w}']  = df['modal_price'].shift(1).rolling(w).std()\n",
    "    df['price_roll_min_7']    = df['modal_price'].shift(1).rolling(7).min()\n",
    "    df['price_roll_max_7']    = df['modal_price'].shift(1).rolling(7).max()\n",
    "    df['price_change_7d']     = df['modal_price'].shift(1) - df['modal_price'].shift(8)\n",
    "    df['price_pct_change_7d'] = df['price_change_7d'] / (df['modal_price'].shift(8) + 1e-6)\n",
    "    df['price_spread']        = df['max_price'] - df['min_price']\n",
    "    df['price_spread_lag7']   = df['price_spread'].shift(7)\n",
    "    for lag in [1,7]:\n",
    "        df[f'arrivals_lag_{lag}'] = df['arrivals_tonnes'].shift(lag)\n",
    "    df['arrivals_roll_7']  = df['arrivals_tonnes'].shift(1).rolling(7).mean()\n",
    "    df['arrivals_roll_30'] = df['arrivals_tonnes'].shift(1).rolling(30).mean()\n",
    "    df['n_markets_lag1']   = df['n_markets'].shift(1)\n",
    "    df = df.merge(rainfall, on=['year','month'], how='left')\n",
    "    exp_key = get_export_key(name)\n",
    "    if exp_key and len(exports):\n",
    "        es = exports[exports['Product Name']==exp_key].rename(columns={'Qty(MT)':'export_qty','Rs(Crore)':'export_val'})\n",
    "        df = df.merge(es[['year','export_qty','export_val']], on='year', how='left')\n",
    "        kw = exp_key.split()[0]\n",
    "        im = imports[imports['Product Name'].str.startswith(kw,na=False)].rename(columns={'Qty(MT)':'import_qty','Rs(Crore)':'import_val'})\n",
    "        df = df.merge(im.groupby('year')[['import_qty','import_val']].sum().reset_index(), on='year', how='left')\n",
    "    else:\n",
    "        df[['export_qty','export_val','import_qty','import_val']] = np.nan\n",
    "    crop_key = name.lower().split('(')[0].strip()\n",
    "    ysub = crop_yield[crop_yield['Crop'].str.contains(crop_key, na=False)]\n",
    "    if len(ysub):\n",
    "        yagg = ysub.groupby('Crop_Year').agg(yield_val=('Yield','mean'),area_val=('Area','sum'),\n",
    "            production_val=('Production','sum')).reset_index().rename(columns={'Crop_Year':'year'})\n",
    "        df = df.merge(yagg, on='year', how='left')\n",
    "    else:\n",
    "        df[['yield_val','area_val','production_val']] = np.nan\n",
    "    df['target'] = df['modal_price'].shift(-PREDICTION_HORIZON)\n",
    "    return df\n",
    "\n",
    "print('✓ Feature engineering functions ready')",
]))

# ── 8. XGBoost GPU training loop ──────────────────────────────────────────────
cells.append(cell([
    "# ── GPU-Accelerated XGBoost Training ───────────────────────────────────\n",
    "# Check if GPU available\n",
    "try:\n",
    "    test_dm = xgb.DMatrix(np.random.randn(10,5), label=np.random.randn(10))\n",
    "    xgb.train({'device':'cuda','tree_method':'hist'}, test_dm, num_boost_round=1)\n",
    "    DEVICE = 'cuda'\n",
    "    print('🚀 GPU (CUDA) detected — using GPU training')\n",
    "except Exception:\n",
    "    DEVICE = 'cpu'\n",
    "    print('⚠  No GPU — falling back to CPU')\n",
    "\n",
    "XGB_PARAMS = dict(\n",
    "    objective         = 'reg:squarederror',\n",
    "    tree_method       = 'hist',\n",
    "    device            = DEVICE,\n",
    "    n_estimators      = 500,\n",
    "    learning_rate     = 0.05,\n",
    "    max_depth         = 8,\n",
    "    subsample         = 0.8,\n",
    "    colsample_bytree  = 0.8,\n",
    "    min_child_weight  = 5,\n",
    "    reg_alpha         = 0.1,\n",
    "    reg_lambda        = 1.0,\n",
    "    random_state      = 42,\n",
    "    n_jobs            = -1,\n",
    ")\n",
    "\n",
    "def train_model(commodity_name, feat_df):\n",
    "    ready = feat_df[FEATURE_COLS + ['target','report_date','modal_price']].dropna()\n",
    "    if len(ready) < 60:\n",
    "        return None, None\n",
    "    split = int(len(ready) * 0.8)\n",
    "    X_tr, y_tr = ready.iloc[:split][FEATURE_COLS].fillna(-999), ready.iloc[:split]['target']\n",
    "    X_te, y_te = ready.iloc[split:][FEATURE_COLS].fillna(-999), ready.iloc[split:]['target']\n",
    "    model = xgb.XGBRegressor(**XGB_PARAMS)\n",
    "    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], early_stopping_rounds=30, verbose=False)\n",
    "    y_pred     = model.predict(X_te)\n",
    "    mae        = mean_absolute_error(y_te, y_pred)\n",
    "    r2         = r2_score(y_te, y_pred)\n",
    "    latest     = ready.iloc[[-1]]\n",
    "    pred_price = float(model.predict(latest[FEATURE_COLS].fillna(-999))[0])\n",
    "    curr_price = float(latest['modal_price'].values[0])\n",
    "    curr_date  = pd.Timestamp(latest['report_date'].values[0])\n",
    "    pct_change = (pred_price - curr_price) / curr_price * 100\n",
    "    signal = 'BUY' if pct_change > 5 else ('SELL' if pct_change < -3 else 'HOLD')\n",
    "    history = ready.tail(180)[['report_date','modal_price']].copy()\n",
    "    history['report_date'] = history['report_date'].astype(str)\n",
    "    return model, dict(\n",
    "        commodity       = str(commodity_name),\n",
    "        current_price   = round(curr_price, 2),\n",
    "        predicted_price = round(pred_price, 2),\n",
    "        pct_change      = round(pct_change, 2),\n",
    "        signal          = signal,\n",
    "        mae             = round(mae, 2),\n",
    "        r2              = round(r2, 4),\n",
    "        current_date    = str(curr_date.date()),\n",
    "        predicted_date  = str((curr_date + timedelta(days=PREDICTION_HORIZON)).date()),\n",
    "        history         = history.to_dict('records'),\n",
    "        data_points     = len(ready),\n",
    "    )\n",
    "\n",
    "print('✓ Model functions ready')",
]))

# ── 9. Run training on all crops ──────────────────────────────────────────────
cells.append(cell([
    "# ── Train on ALL crops ──────────────────────────────────────────────────\n",
    "commodities = commodity_df['commodity'].unique()\n",
    "print(f'Training {len(commodities)} crop models on {DEVICE.upper()}...\\n')\n",
    "\n",
    "predictions = []\n",
    "failed      = []\n",
    "\n",
    "for commodity in tqdm(commodities, desc='Training crops'):\n",
    "    series  = commodity_df[commodity_df['commodity'] == commodity].copy()\n",
    "    feat_df = make_features(series)\n",
    "    model, result = train_model(commodity, feat_df)\n",
    "    if model is None:\n",
    "        failed.append(str(commodity))\n",
    "        continue\n",
    "    model.save_model(os.path.join(MODELS_DIR, f'{commodity}.json'))\n",
    "    predictions.append(result)\n",
    "\n",
    "sigs = [p['signal'] for p in predictions]\n",
    "print(f'\\n✅ Done!')\n",
    "print(f'   🟢 BUY  = {sigs.count(\"BUY\")}')\n",
    "print(f'   🔴 SELL = {sigs.count(\"SELL\")}')\n",
    "print(f'   🟡 HOLD = {sigs.count(\"HOLD\")}')\n",
    "if failed:\n",
    "    print(f'   ⚠  Skipped (insufficient data): {len(failed)} crops')",
]))

# ── 10. Save predictions.json ─────────────────────────────────────────────────
cells.append(cell([
    "# ── Save predictions.json ───────────────────────────────────────────────\n",
    "output = dict(\n",
    "    generated_at            = datetime.now().isoformat(),\n",
    "    prediction_horizon_days = PREDICTION_HORIZON,\n",
    "    device_used             = DEVICE,\n",
    "    total_crops             = len(predictions),\n",
    "    predictions             = sorted(predictions, key=lambda x: abs(x['pct_change']), reverse=True),\n",
    ")\n",
    "out_path = os.path.join(BASE_DIR, 'predictions.json')\n",
    "with open(out_path, 'w') as f:\n",
    "    json.dump(output, f, indent=2, default=str)\n",
    "print(f'📄 Saved → {out_path}  ({os.path.getsize(out_path)/1024:.0f} KB)')",
]))

# ── 11. Top signals preview ───────────────────────────────────────────────────
cells.append(cell([
    "# ── Preview top signals ─────────────────────────────────────────────────\n",
    "df_sig = pd.DataFrame([{\n",
    "    'Commodity': p['commodity'],\n",
    "    'Signal': p['signal'],\n",
    "    'Current ₹/qtl': p['current_price'],\n",
    "    'Predicted ₹/qtl': p['predicted_price'],\n",
    "    'Change %': p['pct_change'],\n",
    "    'R²': p['r2'],\n",
    "    'MAE': p['mae'],\n",
    "} for p in predictions])\n",
    "\n",
    "print('\\n🟢 TOP BUY SIGNALS')\n",
    "print(df_sig[df_sig['Signal']=='BUY'].sort_values('Change %', ascending=False).head(15).to_string(index=False))\n",
    "print('\\n🔴 TOP SELL SIGNALS')\n",
    "print(df_sig[df_sig['Signal']=='SELL'].sort_values('Change %').head(15).to_string(index=False))",
]))

# ── 12. Download output ───────────────────────────────────────────────────────
cells.append(cell([
    "# ── Download predictions.json to your machine ───────────────────────────\n",
    "from google.colab import files\n",
    "files.download(out_path)\n",
    "print('⬇  Download started — copy predictions.json to your project folder')\n",
    "print('   Then run: python app.py   to launch the dashboard')",
]))

# ── Build notebook JSON ───────────────────────────────────────────────────────
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
    },
    "cells": cells,
}

out = "CropSignal_GPU.ipynb"
with open(out, "w") as f:
    json.dump(notebook, f, indent=1)

print(f"✅  Created {out}  ({os.path.getsize(out)/1024:.0f} KB)")
print("   Open in Google Colab: https://colab.research.google.com → File → Upload notebook")
