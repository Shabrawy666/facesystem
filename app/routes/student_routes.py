from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity, get_jwt
from app.models import Student, Course, AttendanceSession, Attendancelog, db, bytes_to_numpy_image
import cv2
import json
from core.models.face_recognition import FaceRecognitionSystem
from app.ml_backend import verify_attendance_from_db, register_face_backend, get_liveness_system
from core.session.wifi_verification import WifiVerificationSystem
from enum import Enum
from dataclasses import asdict, is_dataclass

def to_json_safe(obj):
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

student_bp = Blueprint('student', __name__)

wifi_verification_system = WifiVerificationSystem()

@student_bp.route('/student/courses', methods=['GET'])
@jwt_required()
def student_courses():
    student_id = get_jwt_identity()
    claims = get_jwt()
    if claims.get('role') != 'student':
        return jsonify({"message": "Forbidden"}), 403
    student = Student.query.filter_by(student_id=student_id).first()
    if student is None:
        return jsonify([]), 200
    return jsonify([{"course_id": c.course_id, "name": c.course_name} for c in student.enrolled_courses])

@student_bp.route('/student/open_sessions', methods=['GET'])
@jwt_required()
def student_open_sessions():
    student_id = get_jwt_identity()
    claims = get_jwt()
    if claims.get('role') != 'student':
        return jsonify({"message": "Forbidden"}), 403
    student = Student.query.filter_by(student_id=student_id).first()
    if student is None:
        return jsonify([]), 200
    open_sessions = []
    for course in student.enrolled_courses:
        sessions = AttendanceSession.query.filter_by(course_id=course.course_id, is_active=True).all()
        for s in sessions:
            open_sessions.append({
                "course_id": course.course_id,
                "course_name": course.course_name,
                "session_id": s.id
            })
    return jsonify(open_sessions)

@student_bp.route('/student/sessions/<int:session_id>/attend', methods=['POST'])
@jwt_required()
def student_attend(session_id):
    student_id = get_jwt_identity()
    claims = get_jwt()
    if claims.get('role') != 'student':
        return jsonify({"message": "Forbidden"}), 403
    file = request.files.get('image')
    if not file:
        return jsonify({"success": False, "message": "Image file required"}), 400

    # Look up session
    session = AttendanceSession.query.get(session_id)
    if not session or not session.is_active:
        return jsonify({"success": False, "message": "Session is not open"}), 403

    course = Course.query.get(session.course_id)
    student = Student.query.get(student_id)
    if not course or not student or not course in student.enrolled_courses:
        return jsonify({"success": False, "message": "Not allowed"}), 403

    # WiFi verification and safety
    student_wifi_data_str = request.form.get("student_wifi_data", "{}")
    try:
        student_wifi_data = json.loads(student_wifi_data_str)
    except Exception:
        student_wifi_data = {}
    student_ip = request.remote_addr

    wifi_result = wifi_verification_system.verify_wifi_connection(
        session_id=str(session_id),
        student_id=student_id,
        student_wifi_data=student_wifi_data,
        student_ip=student_ip
    )

    connection_strength = wifi_result.get("connection_strength")
    strength_label = wifi_result.get("strength_label")

    def serialize_value(val):
        if isinstance(val, set):
            return list(val)
        elif isinstance(val, Enum):
            return val.value
        elif isinstance(val, bytes):
            return val.decode('utf-8')
        return val

    safe_wifi_result = {k: serialize_value(v) for k, v in wifi_result.items()}

    # --- IN-MEMORY Numpy Image Decode ---
    img_bytes = file.read()
    img = bytes_to_numpy_image(img_bytes)
    if img is None:
        return jsonify({"success": False, "message": "Invalid image file"}), 400

    # RUN BOTH FACE RECOGNITION AND LIVENESS
    # (verify_attendance_from_db does both and returns all required scores)
    result = verify_attendance_from_db(student_id, captured_img=img)

    # Merge safe wifi_result into result (override only if not already in 'result')
    result["connection_strength"] = safe_wifi_result.get("connection_strength")
    result["strength_label"] = safe_wifi_result.get("strength_label")
    for k, v in safe_wifi_result.items():
        if k not in result:
            result[k] = v

    # Attendance is logged ONLY if both face and liveness succeed!
    if result and result.get("success"):
        log = Attendancelog.query.filter_by(
            student_id=student_id,
            course_id=course.course_id,
            session_id=session_id
        ).first()
        if log:
            log.verification_score = float(result.get("confidence_score")) if result.get("confidence_score") is not None else None
            log.liveness_score = float(result.get("liveness_score")) if result.get("liveness_score") is not None else None
            log.status = "present"
            log.connection_strength = strength_label if strength_label else 'none'
        else:
            log = Attendancelog.create_attendance(
                student_id=student_id,
                course_id=course.course_id,
                session_id=session_id,
                teacher_id=course.teacher_id,
                verification_score=float(result.get("confidence_score")) if result.get("confidence_score") is not None else None,
                liveness_score=float(result.get("liveness_score")) if result.get("liveness_score") is not None else None,
                status="present",
                connection_strength=strength_label if strength_label else 'none'
            )
            db.session.add(log)
        db.session.commit()

    return jsonify(to_json_safe(result))


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

    img_bytes = file.read()
    img = bytes_to_numpy_image(img_bytes)
    if img is None:
        return jsonify({"success": False, "message": "Invalid image file"}), 400

    # -- ENFORCE ANTI-SPOOF/LIVENESS FOR REGISTRATION --
    liveness_result = get_liveness_system().analyze(img)
    liveness_score = float(liveness_result.get("score", 0.0))
    is_live = bool(liveness_result.get("live", True))
    if not is_live or liveness_score < 0.8:  # threshold as in attend
        return jsonify({
            "success": False,
            "message": "Anti-spoofing check failed: Live person not detected. Please provide a real/live photo.",
            "liveness_score": liveness_score
        }), 400

    # ML backend: handles saving cropped face to DB and creating encoding
    reg_result = register_face_backend(student_id=student_id, image=img)
    if not reg_result.get("success"):
        return jsonify({
            "success": False,
            "message": reg_result.get("message", "Failed to register face")
        })

    # Save face encoding and processed face_image to Student DB
    student = Student.query.filter_by(student_id=student_id).first()
    if not student:
        return jsonify({"success": False, "message": "Student not found"}), 404

    student.face_encoding = reg_result["encoding"]
    # Save the **preprocessed* image to DB as binary (cropped clean face)
    preprocessed = reg_result.get("preprocessed", img)
    _, buf = cv2.imencode('.jpg', preprocessed)
    student.face_image = buf.tobytes()
    db.session.commit()

    return jsonify({"success": True, "message": "Face successfully registered and saved."})