from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity, get_jwt
from app.models import Teacher, Course, Student, AttendanceSession, Attendancelog, db
from datetime import datetime

teacher_bp = Blueprint('teacher', __name__)

# --- Teacher Login ---
@teacher_bp.route('/login', methods=['POST'])
def teacher_login():
    data = request.json
    teacher_id = str(data.get('teacher_id'))
    password = data.get('password')
    teacher = Teacher.query.filter_by(teacher_id=teacher_id).first()
    if not teacher or not teacher.check_password(password):
        return jsonify({"success": False, "message": "Invalid credentials"}), 401

    courses = [
        {"course_id": course.course_id, "course_name": course.course_name}
        for course in teacher.courses
    ]
    token = create_access_token(identity=teacher.teacher_id, additional_claims={"role": "teacher"})
    return jsonify({
        "success": True,
        "token": token,
        "teacher_id": teacher.teacher_id,
        "name": teacher.name,
        "courses": courses
    })


def get_course_with_attendance(course):
    # For each session in the course, provide student-by-student attendance/absence
    sessions_data = []
    for session in course.attendance_sessions:
        session_logs = Attendancelog.query.filter_by(course_id=course.course_id, session_id=session.id).all()
        student_attendance = [
            {
                "student_id": log.student_id,
                "name": log.student.name if log.student else "",
                "status": log.status,
                "verified": log.is_verified,
                "verification_score": log.verification_score,
                "liveness_score": log.liveness_score,
                "verification_time": log.verification_timestamp.isoformat() if log.verification_timestamp else None
            }
            for log in session_logs
        ]
        sessions_data.append({
            "session_id": session.id,
            "session_number": session.session_number,
            "start_time": session.start_time.isoformat() if session.start_time else None,
            "end_time": session.end_time.isoformat() if session.end_time else None,
            "status": session.status,
            "student_attendance": student_attendance
        })
    students = [
        {"student_id": s.student_id, "name": s.name, "email": s.email}
        for s in course.enrolled_students
    ]
    return {
        "course_id": course.course_id,
        "course_name": course.course_name,
        "students": students,
        "sessions": sessions_data
    }

# --- View Course Details and Attendance ---
@teacher_bp.route('/courses/<int:course_id>/details', methods=['GET'])
@jwt_required()
def view_course_details(course_id):
    teacher_id = get_jwt_identity()
    teacher = Teacher.query.filter_by(teacher_id=teacher_id).first()
    course = Course.query.filter_by(course_id=course_id, teacher_id=teacher_id).first()
    if not course:
        return jsonify({"success": False, "message": "Access denied (Not your course)"}), 403
    return jsonify({"success": True, "course": get_course_with_attendance(course)})

# --- Start Session ---
@teacher_bp.route('/courses/<int:course_id>/sessions/start', methods=['POST'])
@jwt_required()
def start_course_session(course_id):
    teacher_id = get_jwt_identity()
    teacher = Teacher.query.filter_by(teacher_id=teacher_id).first()
    course = Course.query.filter_by(course_id=course_id, teacher_id=teacher_id).first()
    if not course:
        return jsonify({"success": False, "message": "Access denied (Not your course)"}), 403
    
    # Prevent multiple active sessions for the same course
    open_session = AttendanceSession.query.filter_by(course_id=course_id, is_active=True).first()
    if open_session:
        return jsonify({"success": False, "message": "A session is already open for this course."}), 400

    wifi_ssid = (request.json or {}).get('wifi_ssid') or request.form.get('wifi_ssid')
    if not wifi_ssid:
        return jsonify({"success": False, "message": "WiFi SSID required"}), 400

    session_number = AttendanceSession.query.filter_by(course_id=course_id).count() + 1
    new_session = AttendanceSession(
        session_number=session_number,
        teacher_id=teacher_id,
        course_id=course_id,
        ip_address=request.remote_addr,
        wifi_ssid=wifi_ssid,
        start_time=datetime.utcnow(),
        is_active=True,
        status="ongoing"
    )
    db.session.add(new_session)
    db.session.commit()
    return jsonify({"success": True, "session_id": new_session.id, "start_time": new_session.start_time.isoformat()})

# --- End Session ---
@teacher_bp.route('/courses/<int:course_id>/sessions/end', methods=['POST'])
@jwt_required()
def end_course_session(course_id):
    teacher_id = get_jwt_identity()
    course = Course.query.filter_by(course_id=course_id, teacher_id=teacher_id).first()
    if not course:
        return jsonify({"success": False, "message": "Access denied (Not your course)"}), 403

    open_session = AttendanceSession.query.filter_by(course_id=course_id, is_active=True).first()
    if not open_session:
        return jsonify({"success": False, "message": "No active session for this course."}), 400

    open_session.end_time = datetime.utcnow()
    open_session.is_active = False
    open_session.status = "completed"

    # Mark all course students who have no attendance log this session as absent
    registered_students = [s.student_id for s in course.enrolled_students]
    present_student_ids = [
        log.student_id for log in Attendancelog.query.filter_by(
            session_id=open_session.id,
            course_id=course_id
        ).all()
    ]
    for student_id in registered_students:
        if student_id not in present_student_ids:
            absent_log = Attendancelog(
                student_id=student_id,
                course_id=course_id,
                session_id=open_session.id,
                teacher_id=teacher_id,
                connection_strength='none',
                status="absent",
                verification_method="manual"
            )
            db.session.add(absent_log)
    db.session.commit()
    return jsonify({
        "success": True,
        "session_id": open_session.id,
        "end_time": open_session.end_time.isoformat()
    })

# --- View Session Attendance ---
@teacher_bp.route('/courses/<int:course_id>/sessions/<int:session_id>/attendance', methods=['GET'])
@jwt_required()
def view_session_attendance(course_id, session_id):
    teacher_id = get_jwt_identity()
    course = Course.query.filter_by(course_id=course_id, teacher_id=teacher_id).first()
    if not course:
        return jsonify({"success": False, "message": "Access denied (Not your course)"}), 403
    session = AttendanceSession.query.filter_by(id=session_id, course_id=course_id).first()
    if not session:
        return jsonify({"success": False, "message": "Session not found."}), 404

    logs = Attendancelog.query.filter_by(session_id=session_id).all()
    return jsonify([{
        "student_id": log.student_id,
        "status": log.status,
        "verified": log.is_verified,
        "verification_score": log.verification_score,
        "liveness_score": log.liveness_score,
        "last_attempt": log.last_attempt.isoformat() if log.last_attempt else None
    } for log in logs])

# --- Edit Attendance Manually (present/absent) ---
@teacher_bp.route('/courses/<int:course_id>/sessions/<int:session_id>/attendance/edit', methods=['POST'])
@jwt_required()
def edit_attendance(course_id, session_id):
    teacher_id = get_jwt_identity()
    # Only allow teacher to edit attendance for his own courses/sessions
    course = Course.query.filter_by(course_id=course_id, teacher_id=teacher_id).first()
    if not course:
        return jsonify({"success": False, "message": "Access denied (You don't teach this course)."}), 403
    session = AttendanceSession.query.filter_by(id=session_id, course_id=course_id, teacher_id=teacher_id).first()
    if not session:
        return jsonify({"success": False, "message": "Session not found."}), 404

    data = request.json
    student_id = data.get('student_id')
    new_status = data.get('status')
    if new_status not in ["present", "absent"]:
        return jsonify({"success": False, "message": "Status must be 'present' or 'absent'."}), 400

    # Find attendance log
    log = Attendancelog.query.filter_by(
        session_id=session_id,
        course_id=course_id,
        student_id=student_id
    ).first()
    if not log:
        return jsonify({"success": False, "message": "Attendance record not found for this student."}), 404

    log.status = new_status
    db.session.commit()

    return jsonify({
        "success": True,
        "student_id": student_id,
        "course_id": course_id,
        "session_id": session_id,
        "new_status": new_status
    })

# --- Teacher Profile Endpoint ---
@teacher_bp.route('/profile', methods=['GET'])
@jwt_required()
def teacher_profile():
    teacher_id = get_jwt_identity()
    teacher = Teacher.query.filter_by(teacher_id=teacher_id).first()
    if not teacher:
        return jsonify({"success": False, "message": "Teacher not found."}), 404
    
    courses_data = []
    for course in teacher.courses:
        sessions_data = []
        for session in course.attendance_sessions:
            sessions_data.append({
                "session_id": session.id,
                "session_number": session.session_number,
                "start_time": session.start_time.isoformat() if session.start_time else None,
                "end_time": session.end_time.isoformat() if session.end_time else None,
                "status": session.status
            })
        courses_data.append({
            "course_id": course.course_id,
            "course_name": course.course_name,
            "sessions": sessions_data
        })

    profile_data = {
        "teacher_id": teacher.teacher_id,
        "name": teacher.name,
        "email": getattr(teacher, "email", None),
        "courses": courses_data
    }

    return jsonify({"success": True, "profile": profile_data})