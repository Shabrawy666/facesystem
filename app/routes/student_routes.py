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
from core.models.face_recognition import LivenessDetector
from sqlalchemy.orm.attributes import flag_modified

student_bp = Blueprint('student', __name__)
wifi_verification_system = WifiVerificationSystem()
liveness_detector = LivenessDetector()

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

    files = request.files.getlist('image')
    if not files or len(files) == 0:
        return jsonify({"success": False, "message": "At least one image file required"}), 400

    student = Student.query.filter_by(student_id=student_id).first()
    if not student:
        return jsonify({"success": False, "message": "Student not found"}), 404

    saved_encodings = 0
    failed_files = []
    for idx, file in enumerate(files):
        # Save image temporarily
        with tempfile.NamedTemporaryFile(dir="/tmp", suffix=f"_regface_{student_id}_{idx}.jpg", delete=False) as tmpfile:
            file.save(tmpfile.name)
            temp_path = tmpfile.name

        result = register_face_backend(student_id, temp_path)
        os.remove(temp_path)

        if not result.get("success"):
            failed_files.append(file.filename or f"image_{idx+1}")
            continue

        new_encoding = result["encoding"]
        if not new_encoding or len(new_encoding) != 128:
            failed_files.append(file.filename or f"image_{idx+1}")
            continue

        if student.face_encodings and isinstance(student.face_encodings, list):
            student.face_encodings.append(new_encoding)
        else:
            student.face_encodings = [new_encoding]
        saved_encodings += 1

    # Mark the JSON field as modified so SQLAlchemy will save it
    flag_modified(student, "face_encodings")
    db.session.commit()

    access_token = create_access_token(identity=student.student_id, additional_claims={"role": "student"})
    registered_courses = [
        {"course_id": course.course_id, "course_name": course.course_name}
        for course in student.enrolled_courses
    ]
    return jsonify({
        "success": saved_encodings > 0,
        "message": f"Saved {saved_encodings} encodings." + (f" Failed: {failed_files}" if failed_files else ""),
        "access_token": access_token,
        "student_data": {
            "student_id": student.student_id,
            "name": student.name,
            "email": student.email,
            "registered_courses": registered_courses
        },
        "face_registered": saved_encodings > 0,
        "encodings_count": len(student.face_encodings)
    }), 200 if saved_encodings > 0 else 400



@student_bp.route('/student/pre_attend_check/<int:course_id>', methods=['POST'])
@jwt_required()
def pre_attend_check(course_id):
    """Check if the student can mark attendance based on IP comparison."""
    student_id = get_jwt_identity()
    claims = get_jwt()
    if claims.get('role') != 'student':
        return jsonify({"message": "Forbidden"}), 403

    course = Course.query.get(course_id)
    student = Student.query.get(student_id)
    if not course or not student or course not in student.enrolled_courses:
        return jsonify({"success": False, "message": "You are not registered in this course."}), 403

    # Find latest session
    session = AttendanceSession.query.filter_by(course_id=course_id).order_by(AttendanceSession.start_time.desc()).first()
    if not session or not session.is_active:
        return jsonify({"success": False, "message": "No active attendance session found for this course."}), 404

    student_ip = request.form.get('student_ip') or request.remote_addr

    wifi_result = wifi_verification_system.verify_connection_strength(
        session_id=session.id,
        student_ip=student_ip
    )

    # If weak, log the attempt for teacher to see
    if not wifi_result.get("success"):
        # Log failed attempt (do not mark as present, just log the try)
        from datetime import datetime
        log = Attendancelog(
            student_id=student_id,
            course_id=course_id,
            session_id=session.id,
            teacher_id=course.teacher_id,
            status="attempted",
            verification_timestamp=datetime.utcnow(),
            connection_strength="weak"
        )
        db.session.add(log)
        db.session.commit()

    return jsonify(wifi_result), 200 if wifi_result.get("success") else 403


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

    student_ip = request.form.get('student_ip') or request.remote_addr

    # ---- IP-based Connection Strength Check ----
    wifi_result = wifi_verification_system.verify_connection_strength(
        session_id=session.id,
        student_ip=student_ip
    )
    if not wifi_result.get("success"):
        return jsonify(wifi_result), 403

    connection_strength = wifi_result.get("connection_strength", "unknown")

    # 3. Save image temporarily
    temp_path = f"/tmp/attend_{student_id}_{session.id}.jpg"
    file.save(temp_path)

    # 4. Extract encoding from uploaded image and compare to DB encodings
    frs = FaceRecognitionSystem()
    try:
        img = cv2.imread(temp_path)

        # --- Liveness Detection ---
        liveness_result = liveness_detector.analyze(img)
        liveness_score = liveness_result.get("score", 0)
        liveness_message = liveness_result.get("message", "")
        is_live = liveness_result.get("live", True)

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

        verified = bool(best_distance <= threshold)

        result = {
            "success": bool(verified),  # always native bool for JSON
            "confidence_score": float(best_similarity),
            "distance": float(best_distance),
            "threshold_used": float(threshold),
            "encodings_compared": int(len(stored_encodings)),
            "liveness_score": liveness_score,
            "liveness_message": liveness_message,
            "message": ("Attendance marked" if verified else "Face not recognized"),
        }

        from datetime import datetime
        log = Attendancelog.query.filter_by(
            student_id=student_id,
            course_id=course_id,
            session_id=session.id
        ).first()
        now = datetime.utcnow()

        if log:
            # Always update attempts_count and last_attempt
            log.attempts_count = (log.attempts_count or 1) + 1
            log.last_attempt = now

            if log.status == "present":
                db.session.commit()
                return jsonify({
                    "success": False,
                    "message": "Attendance already marked for this session.",
                    "attempts_count": log.attempts_count,
                    "last_attempt": log.last_attempt.isoformat() if log.last_attempt else None,
                    "verification_timestamp": log.verification_timestamp.isoformat() if log.verification_timestamp else None,
                }), 400

            # If not already present, update as usual
            log.verification_score = float(best_similarity)
            log.status = "present"
            log.verification_timestamp = now
            log.connection_strength = connection_strength
            log.liveness_score = liveness_score
        else:
            log = Attendancelog(
                student_id=student_id,
                course_id=course_id,
                session_id=session.id,
                teacher_id=course.teacher_id,
                verification_score=float(best_similarity),
                status="present",
                verification_timestamp=now,
                connection_strength=connection_strength,
                liveness_score=liveness_score,
                attempts_count=1,
                last_attempt=now,
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