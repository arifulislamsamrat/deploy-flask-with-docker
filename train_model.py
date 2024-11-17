
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
