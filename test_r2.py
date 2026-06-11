import json
with open('predictions.json') as f:
    d = json.load(f)
preds = d['predictions']
low_r2 = [p for p in preds if p['r2'] < 0.2]
low_r2 = sorted(low_r2, key=lambda x: x['r2'])
print(f'Total crops: {len(preds)}')
print(f'Crops with R2 < 0.2: {len(low_r2)}')
for p in low_r2[:15]:
    print(f"{p['commodity']:<20} R2: {p['r2']:>5.2f} | Points: {p['data_points']:>4} | MAE: {p['mae']:>4.0f}")
