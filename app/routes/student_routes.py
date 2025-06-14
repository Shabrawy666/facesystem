from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity, get_jwt
from app.models import Student, Course, AttendanceSession, Attendancelog, db
import os
import cv2
import json
from core.models.face_recognition import FaceRecognitionSystem
from app.ml_backend import verify_attendance_backend, register_face_backend
from core.session.wifi_verification import WifiVerificationSystem
from enum import Enum
from dataclasses import asdict, is_dataclass

def to_json_safe(obj):
    # Handle dataclass (including nested dataclasses)
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

# Singleton instance at module level
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

    # SAFETY: Remove/set all non-JSON-serializable/enum/set fields!
    def serialize_value(val):
        if isinstance(val, set):
            return list(val)
        elif isinstance(val, Enum):
            return val.value
        elif isinstance(val, bytes):
            return val.decode('utf-8')
        return val

    safe_wifi_result = {k: serialize_value(v) for k, v in wifi_result.items()}

    # Save image temporarily
    temp_path = f"/tmp/uploaded_{student_id}_{session_id}.jpg"
    file.save(temp_path)
    result = verify_attendance_backend(student_id, temp_path)
    os.remove(temp_path)

    # Merge safe wifi_result into result (override only if not already in 'result')
    result["connection_strength"] = safe_wifi_result.get("connection_strength")
    result["strength_label"] = safe_wifi_result.get("strength_label")
    for k, v in safe_wifi_result.items():
        if k not in result:
            result[k] = v

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

        import pprint
        pprint.pprint(result)

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
        print("No image uploaded")
        return jsonify({"success": False, "message": "Image file required"}), 400

    # Save image temporarily
    temp_path = f"/tmp/regface_{student_id}.jpg"
    file.save(temp_path)
    print("Got image file, student_id:", student_id)
    print("Saved to temp:", temp_path)

    img = cv2.imread(temp_path)
    if img is None:
        print("Failed to load image with cv2")
        os.remove(temp_path)
        return jsonify({"success": False, "message": "Invalid image file"}), 400
    else:
        print("Loaded img.shape:", img.shape)

    print("About to run face recognition and liveness...")
    frs = FaceRecognitionSystem()
    encoding_result = frs.get_face_encoding_for_storage(img, student_id=student_id)
    print("Encoding result:", encoding_result)

    if not encoding_result.get("success") or encoding_result.get("encoding") is None:
        os.remove(temp_path)
        print("Face encoding failed:", encoding_result.get("message"))
        return jsonify({
            "success": False,
            "message": encoding_result.get("message", "Failed to register face")
        })

    student = Student.query.filter_by(student_id=student_id).first()
    print("Student DB lookup result:", student)
    if not student:
        os.remove(temp_path)
        print("No student found in DB")
        return jsonify({"success": False, "message": "Student not found"}), 404

    student.face_encoding = encoding_result["encoding"]  # ...
    try:
        db.session.commit()
        print("DB commit successful!")
    except Exception as e:
        print("DB commit error:", str(e))
        raise

    processed = encoding_result.get("preprocessed", img)
    images_dir = os.path.join(os.getcwd(), 'stored_images')
    os.makedirs(images_dir, exist_ok=True)
    image_path = os.path.join(images_dir, f"{student_id}.jpg")
    print("Saving processed face to:", image_path)
    success = cv2.imwrite(image_path, processed)
    print("cv2.imwrite success:", success)

    os.remove(temp_path)
    print("Registration success for:", student_id)
    return jsonify({"success": True, "message": "Face successfully registered and saved."})