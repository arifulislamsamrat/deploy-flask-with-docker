# Deploying Machine Learning Models with Flask and Docker

### Overview

In this hands-on lab, you'll learn how to deploy a machine learning model using Flask and Docker. We'll build a complete end-to-end solution that includes training a model, creating a REST API, and containerizing the application.

**Prerequisites**: Python 3.8+, Docker installed
**Learning Objectives**:

- Train a machine learning model using scikit-learn
- Create a Flask API for model predictions
- Containerize the application using Docker
- Test the deployed model with real requests

### Folder Structure

```markdown
ml-flask-docker/
│
├── app/
│   ├── __pycache__/
│   │   └── __init__.cpython-38.pyc
│   ├── templates/
│   ├── __init__.py
│   ├── model.py
│   └── routes.py
│
├── model/
│   ├── iris_model.pkl
│   └── model_metadata.pkl
│
├── tests/
│   ├── test_api.py
│   └── train_model.py
│
├── Dockerfile
├── requirements.txt
└── run.py
├── test_api.py
└── train_model.py

```

## Step 0: Setting Up the Prerequisites

```jsx

# update the system
sudo apt update

# upgrade the system
sudo apt upgrade

# install pip 
sudo apt install python-pip	#python 2
sudo apt install python3-pip	#python 3

```

## Step 1: Setting Up the Project Structure

```bash

# Create project directory
mkdir ml-flask-docker 
#change the directory
cd ml-flask-docker

# Create required directories
mkdir -p app/templates model tests

# Create required files
touch app/__init__.py app/model.py app/routes.py requirements.txt Dockerfile

```

## Step 2: Installing Required Dependencies

Create requirements.txt with the following content:

```python

flask==2.0.1
scikit-learn==1.0.2
numpy==1.21.0
pandas==1.3.0
gunicorn==20.1.0
werkzeug==2.0
```

Install the dependencies:

```bash
#must need pip installed
pip3 install -r requirements.txt

```

## Step 3: Creating the Machine Learning Model

Create a new file `train_model.py` with the following content:

```python

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Create model directory if it doesn't exist
os.makedirs('model', exist_ok=True)

# Save the model
with open('model/iris_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save feature names and target names
with open('model/model_metadata.pkl', 'wb') as f:
    pickle.dump({
        'feature_names': iris.feature_names,
        'target_names': iris.target_names
    }, f)

print(f"Model trained successfully! Test accuracy: {model.score(X_test, y_test):.4f}")

```

Run the training script:

```bash
# run on python 03
python3 train_model.py 

```

## Step 4: Creating the Flask Application

### 1. Update app/**init**.py:

```python

from flask import Flask
from app.model import load_model

def create_app():
    app = Flask(__name__)

# Load the model at startup
    app.model, app.metadata = load_model()

    from app.routes import main
    app.register_blueprint(main)

    return app

```

### 2. Update app/model.py:

```python

import pickle

def load_model():
    """Load the trained model and metadata"""
    with open('model/iris_model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('model/model_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)

    return model, metadata

```

### 3. Update app/routes.py:

```python

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

```

### 4. Create run.py in the root directory:

```python

from app import create_app

app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

```

## Step 5: Creating the Dockerfile

Create a Dockerfile with the following content:

```
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application
COPY . .

# Train the model
RUN python train_model.py

# Expose port
EXPOSE 5000

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "run:app"]

```

## Step 6: Building and Running the Docker Container

```bash

# Build the Docker image
docker build -t ml-flask-demo .

# Run the container
docker run -p 5000:5000 ml-flask-demo

```

## Step 7: Testing the Deployed Model

Create a new file `test_api.py`:

```python
import requests
import json

# Test data (features for Iris setosa)
test_data = {
    "features": [5.1, 3.5, 1.4, 0.2]# Example iris measurements
}

# Make prediction request
response = requests.post('http://localhost:5000/predict',
                        json=test_data)

# Print results
print("\nPrediction Results:")
print(json.dumps(response.json(), indent=2))

```

Run the test:

```bash
#cmd for python3 
python3 test_api.py
#cmd for python2
python test_api.py

```

**Expected output should look like:**

```json

Prediction Results:
{
  "features_names": [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)"
  ],
  "prediction": "setosa",
  "prediction_probability": [
    1.0,
    0.0,
    0.0
  ],
  "target_names": [
    "setosa",
    "versicolor",
    "virginica"
  ]
}

```

## Step 8: Using curl for Testing

You can also test using curl:

```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [5.1, 3.5, 1.4, 0.2]}'

```