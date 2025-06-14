from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager
from flask_migrate import Migrate
from flask_cors import CORS
from flask import Blueprint

db = SQLAlchemy()
bcrypt = Bcrypt()
jwt = JWTManager()
migrate = Migrate()

def create_app():
    app = Flask(__name__)
    app.config.from_object('app.config.Config')
    db.init_app(app)
    bcrypt.init_app(app)
    jwt.init_app(app)

    # --- IMPORTANT: explicit import of all your models here!
    from app.models import Student, Teacher, Course, Attendancelog, AttendanceSession

    migrate.init_app(app, db)

    CORS(app)

    from app.routes.auth_routes import auth_bp
    from app.routes.teacher_routes import teacher_bp
    from app.routes.student_routes import student_bp
    from app.routes import default_bp
    app.register_blueprint(default_bp)
    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    app.register_blueprint(teacher_bp, url_prefix='/api/teacher')
    app.register_blueprint(student_bp, url_prefix='/api/student')
    return app