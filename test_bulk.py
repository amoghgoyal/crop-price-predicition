import time, json
from app import load_predictions
from live_predictor import predict_live, _model_cache

data = load_predictions()
if not data:
    print("No predictions data")
    exit(1)

crops = [p['commodity'] for p in data['predictions']]

start = time.time()
print(f"Running live prediction for {len(crops)} crops...")

results = []
for crop in crops:
    res = predict_live(crop, predictions_data=data)
    if 'error' not in res:
        results.append(res)

print(f"Finished {len(results)} predictions in {time.time()-start:.2f} seconds")
print(f"Models in cache: {len(_model_cache)}")
