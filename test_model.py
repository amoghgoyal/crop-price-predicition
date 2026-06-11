import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import r2_score, mean_absolute_error
from ml_pipeline import load_commodity_data, make_features, FEATURE_COLS

df = pd.read_csv('crop_data/Agmarknet_Price_Report.csv', chunksize=100000)
barley = []
for chunk in df:
    chunk.columns = chunk.columns.str.strip()
    c = chunk[chunk['Commodity'] == 'Barley (Jau)']
    if len(c): barley.append(c)
bdf = pd.concat(barley)

bdf['report_date'] = pd.to_datetime(bdf['Price Date'], format='%d %b %Y', errors='coerce')
bdf.rename(columns={'Modal Price (Rs./Quintal)': 'modal_price', 'Commodity': 'commodity'}, inplace=True)
bdf = bdf.groupby('report_date')['modal_price'].median().reset_index()
bdf['commodity'] = 'Barley (Jau)'

# Dummy external data
bdf['year'] = bdf['report_date'].dt.year
bdf['month'] = bdf['report_date'].dt.month
bdf['dayofweek']  = bdf['report_date'].dt.dayofweek
bdf['weekofyear'] = bdf['report_date'].dt.isocalendar().week
bdf['quarter']    = bdf['report_date'].dt.quarter
bdf['is_kharif']  = bdf['month'].isin([6, 7, 8, 9, 10]).astype(int)
bdf['is_rabi']    = bdf['month'].isin([11, 12, 1, 2, 3]).astype(int)

for lag in [1, 3, 7, 14, 21, 30]: bdf[f'price_lag_{lag}'] = bdf['modal_price'].shift(lag)
for w in [7, 14, 30]: bdf[f'price_roll_mean_{w}'] = bdf['modal_price'].rolling(w).mean()
for w in [7, 30]: bdf[f'price_roll_std_{w}'] = bdf['modal_price'].rolling(w).std()
bdf['price_roll_min_7'] = bdf['modal_price'].rolling(7).min()
bdf['price_roll_max_7'] = bdf['modal_price'].rolling(7).max()
bdf['price_change_7d'] = bdf['modal_price'] - bdf['modal_price'].shift(7)
bdf['price_pct_change_7d'] = bdf['price_change_7d'] / bdf['modal_price'].shift(7)
bdf['price_spread'] = bdf['price_roll_max_7'] - bdf['price_roll_min_7']
bdf['price_spread_lag7'] = bdf['price_spread'].shift(7)

# Mock missing features
for f in FEATURE_COLS:
    if f not in bdf.columns:
        bdf[f] = -999

bdf['target_abs'] = bdf['modal_price'].shift(-7)
bdf['target_diff'] = bdf['target_abs'] - bdf['modal_price']

bdf.dropna(subset=['target_abs', 'price_lag_30'], inplace=True)

split = int(len(bdf) * 0.8)
train, test = bdf.iloc[:split], bdf.iloc[split:]

X_tr, y_tr_abs, y_tr_diff = train[FEATURE_COLS], train['target_abs'], train['target_diff']
X_te, y_te_abs, y_te_diff = test[FEATURE_COLS], test['target_abs'], test['target_diff']

# 1. Absolute Model
m1 = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
m1.fit(X_tr, y_tr_abs)
p1 = m1.predict(X_te)
print("Abs Model R2:", r2_score(y_te_abs, p1))

# 2. Diff Model
m2 = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
m2.fit(X_tr, y_tr_diff)
p2_diff = m2.predict(X_te)
p2_abs = test['modal_price'].values + p2_diff
print("Diff Model R2 (vs abs):", r2_score(y_te_abs, p2_abs))
