from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity, get_jwt
from app.models import Student, Course, AttendanceSession, Attendancelog, db
import os
import cv2
import numpy as np
import json
import tempfile
from core.utils.config import Config
from enum import Enum
from dataclasses import asdict, is_dataclass
from app.ml_back.wifi_verification_system import WifiVerificationSystem
from app.ml_backend import verify_attendance_backend, register_face_backend

student_bp = Blueprint('student', __name__)
wifi_verification_system = WifiVerificationSystem()

def to_json_safe(obj):
    # Handles dataclass (including nested dataclasses) and other JSON edge cases
    if is_dataclass(obj):
        return to_json_safe(asdict(obj))
    elif isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, set):
        return [to_json_safe(v) for v in obj]
    elif isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    elif isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, (bytes, bytearray)):
        return obj.decode('utf-8', errors='ignore')
    elif isinstance(obj, bool):
        return bool(obj)
    elif isinstance(obj, (int, float, str)) or obj is None:
        return obj
    else:
        return str(obj)

def get_student_courses_with_attendance(student_obj):
    # Returns [{course_id, course_name, attendance: [session_id, status, ...]}]
    courses_info = []
    for course in student_obj.enrolled_courses:
        logs = Attendancelog.query.filter_by(student_id=student_obj.student_id, course_id=course.course_id).all()
        attendance_data = [
            {
                "session_id": log.session_id,
                "status": log.status,
                "verified": log.is_verified,
                "date": log.date.isoformat() if log.date else None,
                "verification_score": log.verification_score,
                "liveness_score": log.liveness_score
            }
            for log in logs
        ]
        courses_info.append({
            "course_id": course.course_id,
            "course_name": course.course_name,
            "attendance": attendance_data
        })
    return courses_info

@student_bp.route('/student/login', methods=['POST'])
def student_login():
    data = request.json
    email = data.get('email')
    password = data.get('password')
    student = Student.query.filter_by(email=email).first()
    if not student or not student.check_password(password):
        return jsonify({"success": False, "message": "Invalid credentials"}), 401

    # Create JWT access token on valid login
    access_token = create_access_token(identity=student.student_id, additional_claims={"role": "student"})

    registered_courses = [
        {"course_id": course.course_id, "course_name": course.course_name}
        for course in student.enrolled_courses
    ]

    if student.face_encoding:
        # Face already registered, proceed with normal login
        return jsonify({
            "success": True,
            "message": "Login successful",
            "access_token": access_token,
            "student_data": {
                "student_id": student.student_id,
                "name": student.name,
                "email": student.email,
                "registered_courses": registered_courses
            },
            "face_registered": True
        }), 200
    else:
        # Send 202 to request face capture and provide instructions for frontend UX
        return jsonify({
            "success": False,
            "message": "Face registration required",
            "access_token": access_token,  # frontend MUST use this for /student/register_face
            "face_registered": False,
            "student_id": student.student_id,
            "action_required": {
                "type": "capture_face",
                "instructions": "Please capture and register your face to access the system"
            }
        }), 202

@student_bp.route('/student/register_face', methods=['POST'])
@jwt_required()
def student_register_face():
    student_id = get_jwt_identity()
    claims = get_jwt()
    if claims.get('role') != 'student':
        return jsonify({"message": "Forbidden"}), 403

    file = request.files.get('image')
    if not file:
        return jsonify({"success": False, "message": "Image file required"}), 400

    # Save image temporarily
    with tempfile.NamedTemporaryFile(dir="/tmp", suffix=f"_regface_{student_id}.jpg", delete=False) as tmpfile:
        file.save(tmpfile.name)
        temp_path = tmpfile.name

    result = register_face_backend(student_id, temp_path)
    os.remove(temp_path)

    if not result.get("success"):
        return jsonify({
            "success": False,
            "message": result.get("message", "Failed to register face")
        })

    # Save face encoding to student record (ARRAY(Float), not used for verification)
    student = Student.query.filter_by(student_id=student_id).first()
    if not student:
        return jsonify({"success": False, "message": "Student not found"}), 404
    student.face_encoding = result["encoding"]
    db.session.commit()

    # No need to save processed image here, already handled in backend

    # Return login payload
    access_token = create_access_token(identity=student.student_id, additional_claims={"role": "student"})
    registered_courses = [
        {"course_id": course.course_id, "course_name": course.course_name}
        for course in student.enrolled_courses
    ]
    return jsonify({
        "success": True,
        "message": "Face encoding saved successfully. Login complete.",
        "access_token": access_token,
        "student_data": {
            "student_id": student.student_id,
            "name": student.name,
            "email": student.email,
            "registered_courses": registered_courses
        },
        "face_registered": True
    }), 200

@student_bp.route('/student/courses_with_attendance', methods=['GET'])
@jwt_required()
def courses_with_attendance():
    student_id = get_jwt_identity()
    claims = get_jwt()
    if claims.get('role') != 'student':
        return jsonify({"message": "Forbidden"}), 403
    student = Student.query.filter_by(student_id=student_id).first()
    if not student:
        return jsonify({"success": False, "message": "Student not found."}), 404
    return jsonify({
        "success": True,
        "courses": get_student_courses_with_attendance(student)
    })

@student_bp.route('/student/attend/<int:course_id>', methods=['POST'])
@jwt_required()
def student_attend_latest_session(course_id):
    student_id = get_jwt_identity()
    claims = get_jwt()
    if claims.get('role') != 'student':
        return jsonify({"message": "Forbidden"}), 403

    file = request.files.get('image')
    if not file:
        return jsonify({"success": False, "message": "Image file required"}), 400

    # 1. Get course and check registration
    course = Course.query.get(course_id)
    student = Student.query.get(student_id)
    if not course or not student or course not in student.enrolled_courses:
        return jsonify({"success": False, "message": "You are not registered in this course."}), 403

    # 2. Find latest (active or not) attendance session for the course
    session = AttendanceSession.query.filter_by(course_id=course_id).order_by(AttendanceSession.start_time.desc()).first()
    if not session:
        return jsonify({"success": False, "message": "No attendance session found for this course yet."}), 404

    if not session.is_active:
        return jsonify({"success": False, "message": "Session is closed."}), 403

    # ---- NEW: WiFi Verification ----
    wifi_ssid = request.form.get('wifi_ssid')
    student_ip = request.form.get('student_ip')
    wifi_result = wifi_verification_system.verify_wifi_connection(
        session_id=f"session_{session.start_time.strftime('%Y%m%d_%H%M%S')}_{course_id}",
        student_wifi_ssid=wifi_ssid,
        student_ip=student_ip
    )
    if not wifi_result.get("success"):
        return jsonify(wifi_result), 403

    # 3. Save image temporarily
    temp_path = f"/tmp/attend_{student_id}_{session.id}.jpg"
    file.save(temp_path)

    # 4. ML attendance verification
    from app.ml_backend import verify_attendance_backend
    result = verify_attendance_backend(student_id, temp_path)
    os.remove(temp_path)

    # 5. Log attendance if success
    if result and result.get("success"):
        log = Attendancelog.query.filter_by(
            student_id=student_id,
            course_id=course_id,
            session_id=session.id
        ).first()
        from datetime import datetime
        if log:
            log.verification_score = float(result.get("confidence_score")) if result.get("confidence_score") is not None else None
            log.liveness_score = float(result.get("liveness_score")) if result.get("liveness_score") is not None else None
            log.status = "present"
            log.verification_timestamp = datetime.utcnow()
        else:
            log = Attendancelog(
                student_id=student_id,
                course_id=course_id,
                session_id=session.id,
                teacher_id=course.teacher_id,
                verification_score=float(result.get("confidence_score")) if result.get("confidence_score") is not None else None,
                liveness_score=float(result.get("liveness_score")) if result.get("liveness_score") is not None else None,
                status="present",
                verification_timestamp=datetime.utcnow()
            )
            db.session.add(log)
        db.session.commit()

    return jsonify(to_json_safe(result))