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
