from flask import Flask, request
from flask_session import Session
from flask_caching import Cache
from flask_babel import Babel
from app.config import Config
import os
import logging

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Use shared Redis connection
    cache = Cache(app)
    app.cache = cache  # Store on app for access via current_app.cache
    Session(app)

    # Validate required Google Drive env vars at startup
    required_drive_vars = ['PC_DRIVE_FOLDER_ID', 'XY_DRIVE_FOLDER_ID', 'FINAL_DRIVE_FOLDER_ID']
    for var in required_drive_vars:
        if not os.getenv(var):
            logging.warning(f"WARNING: {var} is not set — uploads to Google Drive will fail.")

    def get_locale():
        return request.accept_languages.best_match(app.config['LANGUAGES'])
    babel = Babel(app, locale_selector=get_locale)

    # Blueprints
    from app.vase.routes import vaseBP
    from app.metadata.routes import metadata
    from app.experiment.routes import experimentBP
    from app.about_us.routes import aboutBP

    app.register_blueprint(vaseBP)
    app.register_blueprint(metadata)
    app.register_blueprint(experimentBP)
    app.register_blueprint(aboutBP)

    return app
