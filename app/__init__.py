#Flask app instance goes here
#initialize sqlite connection or sqlalchemy
import os
from flask import Flask
from db import init_app

app = Flask(__name__)

def create_app():
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'app.sqlite')
    )