from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import logging

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})  # Restrict to specific routes

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load the model
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    logging.error(f"Error loading model: {e}")
    model = None

# Define ethnicity mapping
ethnicity_map = {
    "White European": 5, "Asian": 6, "Middle Eastern": 8, "Black": 7,
    "South Asian": 10, "Hispanic": 0, "Others": 3, "Latino": 1,
    "Pacifica": 4, "Mixed": 9, "Native Indian": 2
}

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.json
        logging.info(f"Received data: {data}")

        # Validate incoming data
        required_fields = [f'q{i}' for i in range(1, 11)] + ['age', 'sex', 'ethnicity', 'jaundice', 'familyASD']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400

        # Prepare the input data
        input_data = [
            1 if data[f'q{i}'] == 'Yes' else 0 for i in range(1, 11)
        ]
        input_data.append(int(data['age']))
        input_data.append(1 if data['sex'] == 'Male' else 0)
        
        if data['ethnicity'] not in ethnicity_map:
            return jsonify({'error': 'Invalid ethnicity value'}), 400

        input_data.append(ethnicity_map[data['ethnicity']])
        input_data.append(1 if data['jaundice'] == 'Yes' else 0)
        input_data.append(1 if data['familyASD'] == 'Yes' else 0)

        # Convert to numpy array
        input_array = np.array(input_data).reshape(1, -1)
        logging.info(f"Model input: {input_array}")

        # Make prediction
        result = model.predict(input_array)
        prediction = "ASD" if result[0] == 1 else "NO ASD"
        logging.info(f"Prediction: {prediction}")

        return jsonify({'prediction': prediction})

    except Exception as e:
        logging.error(f"Error in prediction: {e}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
