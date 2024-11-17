
from flask import Blueprint, request, jsonify, current_app
import numpy as np

main = Blueprint('main', __name__)

@main.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200

@main.route('/predict', methods=['POST'])
def predict():
    try:
# Get data from request
        data = request.get_json(force=True)
        features = data['features']

# Validate input
        if len(features) != 4:
            return jsonify({'error': 'Expected 4 features'}), 400

# Convert to numpy array and reshape
        features = np.array(features).reshape(1, -1)

# Make prediction
        model = current_app.model
        metadata = current_app.metadata

        prediction = model.predict(features)
        prediction_proba = model.predict_proba(features)

# Get predicted class name
        predicted_class = metadata['target_names'][prediction[0]]

# Prepare response
        response = {
            'prediction': predicted_class,
            'prediction_probability': prediction_proba[0].tolist(),
            'features_names': metadata['feature_names'],
            'target_names': metadata['target_names'].tolist()
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 400
