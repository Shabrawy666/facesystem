from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity, get_jwt
from app.models import Teacher, Student, Course, AttendanceSession, Attendancelog, db
from app.ml_back.wifi_verification_system import WifiVerificationSystem

teacher_bp = Blueprint('teacher', __name__)
wifi_verification_system = WifiVerificationSystem()

def is_teacher_of_course(teacher_id, course_id):
    return Course.query.filter_by(course_id=course_id, teacher_id=teacher_id).first() is not None

# ---- LOGIN ----
@teacher_bp.route('/teacher/login', methods=['POST'])
def teacher_login():
    data = request.json
    teacher_id = str(data.get('teacher_id'))
    password = data.get('password')
    teacher = Teacher.query.filter_by(teacher_id=teacher_id).first()
    if not teacher or not teacher.check_password(password):
        return jsonify({"success": False, "message": "Invalid credentials"}), 401

    token = create_access_token(identity=teacher.teacher_id, additional_claims={"role": "teacher"})
    courses = Course.query.filter_by(teacher_id=teacher.teacher_id).all()
    courses_list = [{"course_id": c.course_id, "course_name": c.course_name} for c in courses]
    return jsonify({
        "success": True,
        "token": token,
        "teacher_id": teacher.teacher_id,
        "name": teacher.name,
        "courses": courses_list
    })

# ---- PROFILE ----
@teacher_bp.route('/teacher/profile', methods=['GET'])
@jwt_required()
def teacher_profile():
    teacher_id = get_jwt_identity()
    teacher = Teacher.query.get(teacher_id)
    courses = Course.query.filter_by(teacher_id=teacher_id).all()
    course_objs = []
    for c in courses:
        sessions = AttendanceSession.query.filter_by(course_id=c.course_id).all()
        session_objs = [
            {
                "session_id": s.id,
                "session_number": s.session_number,
                "is_active": s.is_active,
                "status": s.status,
                "start_time": s.start_time.isoformat() if s.start_time else None,
                "end_time": s.end_time.isoformat() if s.end_time else None,
            } for s in sessions
        ]
        course_objs.append({
            "course_id": c.course_id,
            "course_name": c.course_name,
            "sessions": session_objs
        })
    return jsonify({
        "teacher_id": teacher.teacher_id,
        "name": teacher.name,
        "courses": course_objs
    })

# ---- COURSE DETAILS (STUDENTS & ATTENDANCE) ----
@teacher_bp.route('/teacher/course/<int:course_id>/details', methods=['GET'])
@jwt_required()
def course_details(course_id):
    teacher_id = get_jwt_identity()
    if not is_teacher_of_course(teacher_id, course_id):
        return jsonify({"message": "Access denied"}), 403
    course = Course.query.get(course_id)
    students = course.enrolled_students
    sessions = AttendanceSession.query.filter_by(course_id=course_id).order_by(AttendanceSession.session_number).all()

    session_list = [
        {
            "session_id": s.id,
            "session_number": s.session_number,
            "is_active": s.is_active,
            "start_time": s.start_time.isoformat() if s.start_time else None,
            "end_time": s.end_time.isoformat() if s.end_time else None,
        } for s in sessions
    ]

    student_list = []
    for student in students:
        logs = Attendancelog.query.filter_by(student_id=student.student_id, course_id=course_id).all()
        attendance = [
            {
                "session_id": log.session_id,
                "status": log.status,
                "verified": log.is_verified,
                "verification_score": log.verification_score
            } for log in logs
        ]
        student_list.append({
            "student_id": student.student_id,
            "name": student.name,
            "attendance": attendance
        })

    return jsonify({
        "course_id": course.course_id,
        "course_name": course.course_name,
        "sessions": session_list,
        "students": student_list
    })

@teacher_bp.route('/teacher/course/<int:course_id>/sessions/start', methods=['POST'])
@jwt_required()
def start_session(course_id):
    teacher_id = get_jwt_identity()
    if not is_teacher_of_course(teacher_id, course_id):
        return jsonify({"message": "Not your course"}), 403
    # Check if an open session exists
    if AttendanceSession.query.filter_by(course_id=course_id, is_active=True).first():
        return jsonify({"message": "A session for this course is already open"}), 400

    from datetime import datetime
    sn = 1 + AttendanceSession.query.filter_by(course_id=course_id).count()
    # REMOVE THE REQUIREMENT FOR WIFI_SSID
    # wifi_ssid = (request.json or {}).get('wifi_ssid') or request.form.get('wifi_ssid')
    # if not wifi_ssid:
    #     return jsonify({"message": "WiFi SSID required"}), 400

    session = AttendanceSession(
        session_number=sn,
        teacher_id=teacher_id,
        course_id=course_id,
        start_time=datetime.utcnow(),
        is_active=True,
        ip_address=request.remote_addr,
        teacher_ip=request.remote_addr,  # This is your new column!
        status="active"
    )
    db.session.add(session)
    db.session.commit()

    return jsonify({"success": True, "session_id": session.id})

# ---- END SESSION ----
@teacher_bp.route('/teacher/course/<int:course_id>/sessions/end', methods=['POST'])
@jwt_required()
def end_session(course_id):
    teacher_id = get_jwt_identity()
    if not is_teacher_of_course(teacher_id, course_id):
        return jsonify({"message": "Not your course"}), 403
    session = AttendanceSession.query.filter_by(course_id=course_id, is_active=True, teacher_id=teacher_id).first()
    if not session:
        return jsonify({"message": "No open session found"}), 404
    from datetime import datetime
    session.end_time = datetime.utcnow()
    session.is_active = False
    session.status = "completed"
    db.session.commit()

    # ---- NEW: End the persistent WifiSession ----
    wifi_verification_system.end_session(
        session_id=f"session_{session.start_time.strftime('%Y%m%d_%H%M%S')}_{course_id}",
        teacher_id=teacher_id
    )

    return jsonify({"success": True})


# ---- VIEW ATTENDANCE (FOR SESSION OF A COURSE) ----
@teacher_bp.route('/teacher/course/<int:course_id>/sessions/<int:session_id>/attendance', methods=['GET'])
@jwt_required()
def session_attendance(course_id, session_id):
    teacher_id = get_jwt_identity()
    if not is_teacher_of_course(teacher_id, course_id):
        return jsonify({"message": "Access denied"}), 403
    session = AttendanceSession.query.filter_by(id=session_id, course_id=course_id, teacher_id=teacher_id).first()
    if not session:
        return jsonify({"message": "Session not found"}), 404
    logs = Attendancelog.query.filter_by(session_id=session_id, course_id=course_id).all()
    students_in_course = [s.student_id for s in Course.query.get(course_id).enrolled_students]
    attendance_dict = {log.student_id: log for log in logs}
    attendance_list = []
    for sid in students_in_course:
        log = attendance_dict.get(sid)
        if log:
            attendance_list.append({
                "student_id": log.student_id,
                "name": Student.query.get(log.student_id).name if Student.query.get(log.student_id) else "",
                "status": log.status,
                "verified": log.is_verified,
                "verification_score": log.verification_score,
                "connection_strength": log.connection_strength
            })
        else:
            attendance_list.append({
                "student_id": sid,
                "name": Student.query.get(sid).name if Student.query.get(sid) else "",
                "status": "absent",
                "verified": False,
                "verification_score": None,
                "connection_strength": None
            })
    return jsonify(attendance_list)


# ---- MANUAL EDIT ATTENDANCE (requires course and session) ----
@teacher_bp.route('/teacher/course/<int:course_id>/sessions/<int:session_id>/attendance', methods=['PUT'])
@jwt_required()
def edit_attendance(course_id, session_id):
    teacher_id = get_jwt_identity()
    if not is_teacher_of_course(teacher_id, course_id):
        return jsonify({"message": "Access denied"}), 403
    session = AttendanceSession.query.filter_by(id=session_id, course_id=course_id, teacher_id=teacher_id).first()
    if not session:
        return jsonify({"message": "Session not found"}), 404
    data = request.json.get('attendance', [])
    for entry in data:
        sid = entry.get('student_id')
        status = entry.get('status')
        if not sid or not status or status not in ["present", "absent"]:
            continue
        log = Attendancelog.query.filter_by(session_id=session_id, course_id=course_id, student_id=sid).first()
        if log:
            log.status = status
        else:
            # If flipping to present, create log
            if status == "present":
                newlog = Attendancelog(
                    session_id=session_id,
                    course_id=course_id,
                    student_id=sid,
                    teacher_id=teacher_id,
                    status="present"
                )
                db.session.add(newlog)
    db.session.commit()
    return jsonify({"success": True})