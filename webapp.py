from flask import Flask, render_template, request, jsonify
import os
import math
try:
    import pandas as pd
    import numpy as np
except Exception:
    pd = None
    np = None

app = Flask(__name__, static_folder='static', template_folder='templates')

# Load dataset and prepare crop mean feature vectors (if pandas available)
CSV_PATH = os.path.join(os.path.dirname(__file__), 'Crop_Recommendation.csv')
crop_means = {}
feature_cols = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall']
if pd is not None and os.path.exists(CSV_PATH):
    try:
        df = pd.read_csv(CSV_PATH)
        # ensure columns exist
        if set(feature_cols + ['Crop']).issubset(df.columns):
            grp = df.groupby('Crop')[feature_cols].mean()
            for crop, row in grp.iterrows():
                crop_means[crop] = row.values.astype(float)
    except Exception:
        crop_means = {}

# If pandas wasn't available or loading failed, try a lightweight csv fallback
if (not crop_means) and os.path.exists(CSV_PATH):
    try:
        import csv
        sums = {}
        counts = {}
        with open(CSV_PATH, newline='', encoding='utf-8') as fh:
            reader = csv.DictReader(fh)
            # detect header names mapping if slightly different
            for r in reader:
                crop = r.get('Crop') or r.get('crop')
                if not crop:
                    continue
                try:
                    vals = [float(r.get(col, 0)) for col in feature_cols]
                except Exception:
                    # skip rows with invalid numbers
                    continue
                if crop not in sums:
                    sums[crop] = np.array(vals) if np is not None else vals
                    counts[crop] = 1
                else:
                    if np is not None:
                        sums[crop] = sums[crop] + np.array(vals)
                    else:
                        sums[crop] = [s + v for s, v in zip(sums[crop], vals)]
                    counts[crop] += 1
        for crop, total in sums.items():
            c = counts.get(crop, 1)
            if np is not None:
                crop_means[crop] = (total / c).astype(float)
            else:
                crop_means[crop] = [float(x) / c for x in total]
    except Exception:
        # leave crop_means empty on failure
        pass

print(f"Loaded {len(crop_means)} crops from {CSV_PATH}")



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json or {}
    try:
        N = float(data.get('N', 0))
        P = float(data.get('P', 0))
        K = float(data.get('K', 0))
        temperature = float(data.get('temperature', 25))
        humidity = float(data.get('humidity', 50))
        ph = float(data.get('ph', 7))
        rainfall = float(data.get('rainfall', 100))
    except Exception:
        return jsonify({'error': 'Invalid numeric input'}), 400

    # server-side validation (ranges)
    errors = {}
    if N < 0:
        errors['N'] = 'N must be non-negative.'
    if P < 0:
        errors['P'] = 'P must be non-negative.'
    if K < 0:
        errors['K'] = 'K must be non-negative.'
    if temperature < -50 or temperature > 60:
        errors['temperature'] = 'Temperature must be between -50 and 60 °C.'
    if humidity < 0 or humidity > 100:
        errors['humidity'] = 'Humidity must be between 0 and 100%.'
    if ph < 0 or ph > 14:
        errors['ph'] = 'pH must be between 0 and 14.'
    if rainfall < 0:
        errors['rainfall'] = 'Rainfall must be non-negative.'

    if errors:
        return jsonify({'errors': errors}), 400

    # If crop_means is available, score all crops by distance to mean feature vector
    input_vec = np.array([N, P, K, temperature, humidity, ph, rainfall]) if np is not None else None

    scores = {}
    if crop_means and input_vec is not None:
        # compute inverse-distance score (higher is better). Add small epsilon to avoid div-by-zero.
        for crop, mean_vec in crop_means.items():
            # Euclidean distance
            dist = float(np.linalg.norm(input_vec - mean_vec))
            score = 1.0 / (dist + 1e-6)
            scores[crop] = float(score)
        # normalize scores to make them easier to read (optional)
        max_score = max(scores.values()) if scores else 1.0
        for k in list(scores.keys()):
            scores[k] = float(scores[k] / max_score)
        # pick top recommendation
        recommendation = max(scores, key=lambda k: scores[k])
    else:
        # fallback to the previous simple heuristic when dataset isn't available
        scores = {}
        scores['Rice'] = (N * 0.2 + P * 0.1 + K * 0.1) + (7 - abs(ph - 6.5)) * 0.3 + min(rainfall, 200) * 0.001
        scores['Wheat'] = (N * 0.3 + P * 0.2 + K * 0.1) + (7 - abs(ph - 7.0)) * 0.2 + (1 if temperature > 20 else 0) * 0.5
        scores['Maize'] = (N * 0.25 + P * 0.15 + K * 0.15) + (humidity / 100) * 0.2 + (1 if rainfall > 80 else 0) * 0.3
        recommendation = max(scores, key=lambda k: scores[k])

    # Return full scores (sorted) and top recommendation
    sorted_scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
    return jsonify({'recommendation': recommendation, 'scores': sorted_scores})


@app.route('/api/crops', methods=['GET'])
def api_crops():
    # return list of available crops
    return jsonify({'crops': sorted(list(crop_means.keys()))})


@app.route('/api/crop_mean', methods=['GET'])
def api_crop_mean():
    crop = request.args.get('crop')
    if not crop:
        return jsonify({'error': 'crop query parameter required'}), 400
    vec = crop_means.get(crop)
    if vec is None:
        return jsonify({'error': f'crop {crop} not found'}), 404
    # ensure list of floats
    try:
        mean_list = [float(x) for x in vec]
    except Exception:
        mean_list = list(vec)
    return jsonify({'crop': crop, 'mean': mean_list})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
