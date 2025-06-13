# app/routes/teacher_routes.py
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity, get_jwt
from app.models import Course, AttendanceSession, Attendancelog, db

teacher_bp = Blueprint('teacher', __name__)

@teacher_bp.route('/teacher/courses', methods=['GET'])
@jwt_required()
def list_courses():
    teacher_id = get_jwt_identity()
    claims = get_jwt()
    if claims.get('role') != 'teacher':
        return jsonify({"message": "Forbidden"}), 403
    courses = Course.query.filter_by(teacher_id=teacher_id).all()
    return jsonify([{"course_id": c.course_id, "course_name": c.course_name} for c in courses])

@teacher_bp.route('/teacher/courses/<int:course_id>/sessions', methods=['GET'])
@jwt_required()
def list_sessions(course_id):
    teacher_id = get_jwt_identity()
    claims = get_jwt()
    if claims.get('role') != 'teacher':
        return jsonify({"message": "Forbidden"}), 403
    course = Course.query.filter_by(course_id=course_id, teacher_id=teacher_id).first()
    if not course:
        return jsonify({"message": "Access denied"}), 403
    sessions = AttendanceSession.query.filter_by(course_id=course_id).all()
    return jsonify([{"session_id": s.id, "number": s.session_number, "status": s.status} for s in sessions])

@teacher_bp.route('/teacher/courses/<int:course_id>/sessions/start', methods=['POST'])
@jwt_required()
def start_session(course_id):
    teacher_id = get_jwt_identity()
    claims = get_jwt()
    if claims.get('role') != 'teacher':
        return jsonify({"message": "Forbidden"}), 403
    course = Course.query.filter_by(course_id=course_id, teacher_id=teacher_id).first()
    if not course:
        return jsonify({"message": "Not your course"}), 403

    from datetime import datetime
    sn = 1 + AttendanceSession.query.filter_by(course_id=course_id).count()
    # NEW: get SSID (required)
    wifi_ssid = (request.json or {}).get('wifi_ssid') or request.form.get('wifi_ssid')
    if not wifi_ssid:
        return jsonify({"message": "WiFi SSID required"}), 400

    session = AttendanceSession(
        session_number=sn,
        teacher_id=teacher_id,
        course_id=course_id,
        ip_address=request.remote_addr,
        wifi_ssid=wifi_ssid,  # <<< ensure this field is in your model!
        start_time=datetime.utcnow(),
        is_active=True
    )
    db.session.add(session)
    db.session.commit()
    return jsonify({"success": True, "session_id": session.id})

@teacher_bp.route('/teacher/courses/<int:course_id>/sessions/end', methods=['POST'])
@jwt_required()
def end_session(course_id):
    teacher_id = get_jwt_identity()
    claims = get_jwt()
    if claims.get('role') != 'teacher':
        return jsonify({"message": "Forbidden"}), 403
    session_id = request.json.get('session_id')
    session = AttendanceSession.query.filter_by(id=session_id, course_id=course_id, teacher_id=teacher_id).first()
    if not session:
        return jsonify({"message": "Not found"}), 403
    from datetime import datetime
    session.end_time = datetime.utcnow()
    session.is_active = False
    session.status = "completed"
    db.session.commit()
    return jsonify({"success": True})

@teacher_bp.route('/teacher/sessions/<int:session_id>/attendance', methods=['GET'])
@jwt_required()
def view_attendance(session_id):
    teacher_id = get_jwt_identity()
    claims = get_jwt()
    if claims.get('role') != 'teacher':
        return jsonify({"message": "Forbidden"}), 403
    session = AttendanceSession.query.filter_by(id=session_id, teacher_id=teacher_id).first()
    if not session:
        return jsonify({"message": "Access denied"}), 403
    logs = Attendancelog.query.filter_by(session_id=session_id).all()
    return jsonify([
        {
            "student_id": log.student_id,
            "status": log.status,
            "verified": log.is_verified,
            "verification_score": log.verification_score
        }
        for log in logs
    ])

@teacher_bp.route('/teacher/sessions/<int:session_id>/attendance', methods=['PUT'])
@jwt_required()
def edit_attendance(session_id):
    teacher_id = get_jwt_identity()
    claims = get_jwt()
    if claims.get('role') != 'teacher':
        return jsonify({"message": "Forbidden"}), 403
    session = AttendanceSession.query.filter_by(id=session_id, teacher_id=teacher_id).first()
    if not session:
        return jsonify({"message": "Access denied"}), 403
    data = request.json.get('attendance', [])
    for entry in data:
        log = Attendancelog.query.filter_by(session_id=session_id, student_id=entry.get('student_id')).first()
        if log:
            log.status = entry.get('status', log.status)
    db.session.commit()
    return jsonify({"success": True})