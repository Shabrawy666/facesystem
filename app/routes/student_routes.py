from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity, get_jwt
from app.models import Student, Course, AttendanceSession, Attendancelog, db
import cv2
import numpy as np
from core.models.face_recognition import FaceRecognitionSystem
from core.session.wifi_verification import WifiVerificationSystem

wifi_verification_system = WifiVerificationSystem()

student_bp = Blueprint('student', __name__)

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

@student_bp.route('/login', methods=['POST'])
def student_login():
    data = request.json
    email = data.get('email')
    password = data.get('password')
    student = Student.query.filter_by(email=email).first()
    if not student or not student.check_password(password):
        return jsonify({"success": False, "message": "Invalid credentials"}), 401

    # Create JWT access token on valid login
    access_token = create_access_token(identity=student.student_id, additional_claims={"role": "student"})

    # Fetch registered courses for summary
    registered_courses = [
        {"course_id": course.course_id, "course_name": course.course_name}
        for course in student.enrolled_courses
    ]

    # Check if face encoding exists
    if student.face_encoding:
        # Face already registered, proceed with normal login
        return jsonify({
            "message": "Login successful",
            "access_token": access_token,
            "student_data": {
                "student_id": student.student_id,
                "name": student.name,
                "email": student.email,
                "registered_courses": registered_courses,
                "total_courses": len(registered_courses)
            },
            "face_registered": True
        }), 200
    else:
        # Send 202 to request face capture and provide instructions for frontend UX
        return jsonify({
            "message": "Face registration required",
            "access_token": access_token,  # frontend MUST use this for /student/register_face
            "face_registered": False,
            "student_id": student.student_id,
            "action_required": {
                "type": "capture_face",
                "instructions": "Please center your face in the frame",
                "requirements": {
                    "lighting": "Even lighting, no shadows",
                    "angle": "Face directly facing camera",
                    "distance": "About arm's length away",
                    "movement": "Maintain natural movement for liveness check",
                    "eyes": "Keep eyes open and blink naturally"
                }
            }
        }), 202

@student_bp.route('/register_face', methods=['POST'])
@jwt_required()
def student_register_face():
    # Get student identity from JWT
    student_id = get_jwt_identity()
    student = Student.query.filter_by(student_id=student_id).first()
    if not student:
        return jsonify({"success": False, "message": "Student not found."}), 404

    # Check that an image is provided
    file = request.files.get('image')
    if not file:
        return jsonify({"success": False, "message": "Image file required"}), 400

    nparr = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"success": False, "message": "Invalid image file"}), 400

    # Register face with ML code
    frs = FaceRecognitionSystem()
    encoding_result = frs.get_face_encoding_for_storage(img, student_id=student.student_id)
    if not encoding_result.get("success") or encoding_result.get("encoding") is None:
        return jsonify({
            "success": False,
            "message": encoding_result.get("message", "Failed to register face")
        })

    # Save face encoding to Student DB
    student.face_encoding = encoding_result["encoding"]
    db.session.commit()

    # Return the same payload as successful login
    token = create_access_token(identity=student.student_id, additional_claims={"role": "student"})
    return jsonify({
        "success": True,
        "message": "Face encoding saved successfully. Login complete.",
        "token": token,
        "name": student.name,
        "student_id": student.student_id,
        "email": student.email,
        "courses": get_student_courses_with_attendance(student)
    })

# Secure endpoint for getting all registered courses with attendance
@student_bp.route('/courses_with_attendance', methods=['GET'])
@jwt_required()
def courses_with_attendance():
    student_id = get_jwt_identity()
    student = Student.query.filter_by(student_id=student_id).first()
    if not student:
        return jsonify({"success": False, "message": "Student not found."}), 404
    return jsonify({"success": True, "courses": get_student_courses_with_attendance(student)})

@student_bp.route('/courses/<int:course_id>/attend', methods=['POST'])
@jwt_required()
def attend_latest_session(course_id):
    student_id = get_jwt_identity()
    student = Student.query.filter_by(student_id=student_id).first()
    if not student:
        return jsonify({"success": False, "message": "Student not found."}), 404

    course = Course.query.get(course_id)
    if not course:
        return jsonify({"success": False, "message": "Course not found."}), 404

    # Check if student is registered for course
    if not student in course.enrolled_students:
        return jsonify({"success": False, "message": "You are not registered in this course."}), 403

    # Find latest session for this course (latest start_time or highest ID)
    session = AttendanceSession.query \
        .filter_by(course_id=course_id) \
        .order_by(AttendanceSession.start_time.desc()) \
        .first()

    if not session:
        return jsonify({"success": False, "message": "No attendance session found for this course yet."}), 404

    if not session.is_active:
        return jsonify({"success": False, "message": "Session is closed."}), 403

    file = request.files.get('image')
    if not file:
        return jsonify({"success": False, "message": "Image file required"}), 400

    nparr = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"success": False, "message": "Invalid image file."}), 400

    # --- Calculate Connection Strength (WiFi/IP) ---
    student_wifi_data_str = request.form.get("student_wifi_data", "{}")
    try:
        student_wifi_data = json.loads(student_wifi_data_str)
    except Exception:
        student_wifi_data = {}
    student_ip = request.remote_addr

    wifi_result = wifi_verification_system.verify_wifi_connection(
        session_id=str(session.id),
        student_id=student_id,
        student_wifi_data=student_wifi_data,
        student_ip=student_ip
    )
    connection_strength = wifi_result.get("connection_strength", "unknown")
    strength_label = wifi_result.get("strength_label", None)

    # --- ML Verification ---
    frs = FaceRecognitionSystem()
    result = frs.verify_student(student_id, img)

    if not result or not getattr(result, "success", False):
        return jsonify({
            "success": False,
            "message": getattr(result, "error_message", "Verification failed"),
            "verification_type": getattr(result, "verification_type", "face"),
            "confidence_score": getattr(result, "confidence_score", 0.0),
            "liveness_score": (getattr(result, "data", {}) or {}).get("liveness_score", None)
        }), 401

    # --- Attendance Log ---
    log = Attendancelog.query.filter_by(
        student_id=student_id,
        course_id=course.course_id,
        session_id=session.id
    ).first()
    from datetime import datetime
    if not log:
        log = Attendancelog(
            student_id=student_id,
            course_id=course.course_id,
            session_id=session.id,
            teacher_id=course.teacher_id,
            connection_strength=connection_strength,
            status="present",
            verification_score=result.confidence_score,
            liveness_score=(result.data or {}).get("liveness_score", None),
            verification_method="face",
            verification_timestamp=datetime.utcnow(),
            last_attempt=datetime.utcnow()
        )
        db.session.add(log)
    else:
        log.status = "present"
        log.verification_score = result.confidence_score
        log.liveness_score = (result.data or {}).get("liveness_score", None)
        log.verification_method = "face"
        log.attempts_count += 1
        log.connection_strength = connection_strength
        log.verification_timestamp = datetime.utcnow()
        log.last_attempt = datetime.utcnow()

    db.session.commit()

    return jsonify({
        "success": True,
        "message": "Attendance marked successfully.",
        "verification_type": "face",
        "confidence_score": result.confidence_score,
        "liveness_score": (result.data or {}).get("liveness_score", None),
        "verification_time": result.verification_time,
        "connection_strength": connection_strength,
        "strength_label": strength_label,
        "attendance": {
            "student_id": student_id,
            "course_id": course.course_id,
            "session_id": session.id,
            "status": "present",
            "session_start_time": session.start_time.isoformat() if session.start_time else None
        }
    })