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
from app.ml_backend import register_face_backend
from app.ml_back.wifi_verification_system import WifiVerificationSystem
from core.models.face_recognition import FaceRecognitionSystem

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

    access_token = create_access_token(identity=student.student_id, additional_claims={"role": "student"})

    registered_courses = [
        {"course_id": course.course_id, "course_name": course.course_name}
        for course in student.enrolled_courses
    ]

    # Updated: check face_encodings
    if student.face_encodings and len(student.face_encodings) > 0:
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
        return jsonify({
            "success": False,
            "message": "Face registration required",
            "access_token": access_token,
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

    # Save encoding as JSON array in face_encodings
    student = Student.query.filter_by(student_id=student_id).first()
    if not student:
        return jsonify({"success": False, "message": "Student not found"}), 404

    # Append new encoding to face_encodings array
    new_encoding = result["encoding"]
    if not new_encoding or len(new_encoding) != 128:
        return jsonify({"success": False, "message": "Invalid encoding data"}), 400

    if student.face_encodings and isinstance(student.face_encodings, list):
        # Avoid duplicates
        for enc in student.face_encodings:
            if np.allclose(enc, new_encoding, atol=1e-6):
                break
        else:
            student.face_encodings.append(new_encoding)
    else:
        student.face_encodings = [new_encoding]

    db.session.commit()

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

    # 2. Find latest session for the course
    session = AttendanceSession.query.filter_by(course_id=course_id).order_by(AttendanceSession.start_time.desc()).first()
    if not session:
        return jsonify({"success": False, "message": "No attendance session found for this course yet."}), 404
    if not session.is_active:
        return jsonify({"success": False, "message": "Session is closed."}), 403

    # ---- WiFi Verification ----
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

    # 4. Extract encoding from uploaded image and compare to DB encodings
    frs = FaceRecognitionSystem()
    try:
        img = cv2.imread(temp_path)
        encoding_result = frs.get_face_encoding_for_storage(img)
        os.remove(temp_path)
        if not encoding_result["success"]:
            return jsonify({"success": False, "message": encoding_result.get("message", "Failed to extract encoding")})

        uploaded_encoding = np.array(encoding_result["encoding"])
        if uploaded_encoding.shape != (128,):
            return jsonify({"success": False, "message": "Invalid encoding from uploaded image"})

        if not student.face_encodings or not isinstance(student.face_encodings, list) or len(student.face_encodings) == 0:
            return jsonify({"success": False, "message": "No face encodings registered for this student."})

        # Compare to each encoding in DB
        stored_encodings = [np.array(enc) for enc in student.face_encodings if isinstance(enc, list) and len(enc) == 128]
        if len(stored_encodings) == 0:
            return jsonify({"success": False, "message": "No valid face encodings for this student."})

        threshold = getattr(Config, 'FACE_RECOGNITION_THRESHOLD', 0.46)
        best_similarity = -1.0
        best_distance = 1.0
        for stored_vec in stored_encodings:
            similarity = frs._calculate_similarity(uploaded_encoding, stored_vec)
            distance = 1.0 - similarity
            if similarity > best_similarity:
                best_similarity = similarity
                best_distance = distance

        verified = best_distance <= threshold

        result = {
            "success": verified,
            "confidence_score": float(best_similarity),
            "distance": float(best_distance),
            "threshold_used": threshold,
            "encodings_compared": len(stored_encodings),
            "message": ("Attendance marked" if verified else "Face not recognized")
        }

        # 5. Log attendance if success
        if verified:
            log = Attendancelog.query.filter_by(
                student_id=student_id,
                course_id=course_id,
                session_id=session.id
            ).first()
            from datetime import datetime
            if log:
                log.verification_score = float(best_similarity)
                log.status = "present"
                log.verification_timestamp = datetime.utcnow()
            else:
                log = Attendancelog(
                    student_id=student_id,
                    course_id=course_id,
                    session_id=session.id,
                    teacher_id=course.teacher_id,
                    verification_score=float(best_similarity),
                    status="present",
                    verification_timestamp=datetime.utcnow()
                )
                db.session.add(log)
            db.session.commit()

        return jsonify(result)

    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"success": False, "message": f"Verification error: {str(e)}"})

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