# app/routes/__init__.py
# Empty, needed so Python treats 'routes' as a package

from flask import Blueprint

default_bp = Blueprint('default', __name__)

@default_bp.route('/', methods=['GET'])
def index():
    return "✅ API server is running! Welcome to Face Attendance API 🚀"