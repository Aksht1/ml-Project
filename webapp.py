from flask import Flask, render_template, request, jsonify

app = Flask(__name__, static_folder='static', template_folder='templates')


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

    # Simple heuristic-based recommendation (replace with model later)
    scores = {}
    scores['Rice'] = (N * 0.2 + P * 0.1 + K * 0.1) + (7 - abs(ph - 6.5)) * 0.3 + min(rainfall, 200) * 0.001
    scores['Wheat'] = (N * 0.3 + P * 0.2 + K * 0.1) + (7 - abs(ph - 7.0)) * 0.2 + (1 if temperature > 20 else 0) * 0.5
    scores['Maize'] = (N * 0.25 + P * 0.15 + K * 0.15) + (humidity / 100) * 0.2 + (1 if rainfall > 80 else 0) * 0.3

    recommendation = max(scores, key=lambda k: scores[k])

    return jsonify({'recommendation': recommendation, 'scores': scores})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
