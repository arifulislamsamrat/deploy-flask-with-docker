
import pickle

def load_model():
    """Load the trained model and metadata"""
    with open('model/iris_model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('model/model_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)

    return model, metadata
