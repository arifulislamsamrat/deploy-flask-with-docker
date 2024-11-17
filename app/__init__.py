
from flask import Flask
from app.model import load_model

def create_app():
    app = Flask(__name__)

# Load the model at startup
    app.model, app.metadata = load_model()

    from app.routes import main
    app.register_blueprint(main)

    return app
