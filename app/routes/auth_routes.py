# app/routes/auth_routes.py
from flask import Blueprint, request, jsonify
from app.models import Student, Teacher
from flask_jwt_extended import create_access_token

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/login/student', methods=['POST'])
def student_login():
    data = request.json
    email = data.get('email')
    password = data.get('password')
    student = Student.query.filter_by(email=email).first()
    if not student or not student.check_password(password):
        return jsonify({"success": False, "message": "Invalid credentials"}), 401
    token = create_access_token(identity=student.student_id, additional_claims={"role": "student"})
    return jsonify({"success": True, "token": token})

@auth_bp.route('/login/teacher', methods=['POST'])
def teacher_login():
    data = request.json
    teacher_id = str(data.get('teacher_id'))
    password = data.get('password')
    teacher = Teacher.query.filter_by(teacher_id=teacher_id).first()
    if not teacher or not teacher.check_password(password):
        return jsonify({"success": False, "message": "Invalid credentials"}), 401
    token = create_access_token(identity=teacher.teacher_id, additional_claims={"role": "teacher"})
    return jsonify({"success": True, "token": token})