# app/config.py
import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'supersecret123')
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or "postgresql://postgres:QBqSgklgHmaAgxNbSUdIAtGfElPRsgap@trolley.proxy.rlwy.net:41161/railway"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "jwtsecret123")
    SQLALCHEMY_ENGINE_OPTIONS = {
        "connect_args": {"sslmode": "require"}
    }