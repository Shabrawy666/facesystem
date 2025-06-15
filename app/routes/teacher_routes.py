from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity, get_jwt
from app.models import Teacher, Student, Course, AttendanceSession, Attendancelog, db
from app.ml_back.wifi_verification_system import WifiVerificationSystem
from datetime import timedelta

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

    token = create_access_token(identity=teacher.teacher_id, additional_claims={"role": "teacher"}, expires_delta=timedelta(days=7))
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

# ---- COURSE DETAILS ----
@teacher_bp.route('/teacher/course/<int:course_id>/summary', methods=['GET'])
@jwt_required()
def course_summary(course_id):
    teacher_id = get_jwt_identity()
    if not is_teacher_of_course(teacher_id, course_id):
        return jsonify({"message": "Access denied"}), 403
    course = Course.query.get(course_id)
    if not course:
        return jsonify({"message": "Course not found"}), 404
    registered_students_count = course.enrolled_students.count()
    return jsonify({
        "course_id": course.course_id,
        "course_name": course.course_name,
        "registered_students_count": registered_students_count
    })
# --- SESSION START ---
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
        teacher_ip=request.remote_addr, 
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

    # Set end time and status
    from datetime import datetime
    session.end_time = datetime.utcnow()
    session.is_active = False
    session.status = "completed"

    # Mark all students who haven't attended as absent
    course = Course.query.get(course_id)
    now = datetime.utcnow()
    for student in course.enrolled_students:
        # Only add an Attendancelog if it doesn't already exist
        log_exists = Attendancelog.query.filter_by(
            student_id=student.student_id,
            course_id=course_id,
            session_id=session.id
        ).first()
        if log_exists is None:
            absent_log = Attendancelog(
                student_id=student.student_id,
                course_id=course_id,
                session_id=session.id,
                teacher_id=teacher_id,
                connection_strength="unknown",   
                status="absent",
                date=now.date(),
                time=now.time(),
                verification_method="none",
                verification_timestamp=now,
                attempts_count=1,
                last_attempt=now,
                verification_details={}
            )
            db.session.add(absent_log)

    db.session.commit()
    return jsonify({"success": True})


# ---- VIEW ATTENDANCE (FOR SESSION OF A COURSE) ----
@teacher_bp.route('/teacher/course/<int:course_id>/sessions/<int:session_id>/details', methods=['GET'])
@jwt_required()
def session_details(course_id, session_id):
    teacher_id = get_jwt_identity()
    if not is_teacher_of_course(teacher_id, course_id):
        return jsonify({"message": "Access denied"}), 403
    session = AttendanceSession.query.filter_by(id=session_id, course_id=course_id, teacher_id=teacher_id).first()
    if not session:
        return jsonify({"message": "Session not found"}), 404
    course = Course.query.get(course_id)
    student_objs = course.enrolled_students.all()
    logs_dict = {log.student_id: log for log in Attendancelog.query.filter_by(session_id=session_id, course_id=course_id).all()}
    attendance = []
    total_present = total_absent = 0
    for student in student_objs:
        log = logs_dict.get(student.student_id)
        status = log.status if log else "absent"
        if status == "present":
            total_present += 1
        else:
            total_absent += 1
        attendance.append({
            "student_id": student.student_id,
            "name": student.name,
            "status": status
        })

    summary = {
        "total_registered": len(student_objs),
        "total_present": total_present,
        "total_absent": total_absent
    }
    return jsonify({
        "session_id": session_id,
        "course_id": course_id,
        "attendance": attendance,
        "summary": summary
    })

# ---- MANUAL EDIT ATTENDANCE (requires course and session) ----
@teacher_bp.route('/teacher/course/<int:course_id>/sessions/<int:session_id>/attendance/toggle', methods=['PUT'])
@jwt_required()
def toggle_attendance(course_id, session_id):
    teacher_id = get_jwt_identity()
    if not is_teacher_of_course(teacher_id, course_id):
        return jsonify({"message": "Access denied"}), 403

    session = AttendanceSession.query.filter_by(id=session_id, course_id=course_id, teacher_id=teacher_id).first()
    if not session:
        return jsonify({"message": "Session not found"}), 404

    student_id = request.json.get('student_id')
    if not student_id:
        return jsonify({"message": "student_id is required"}), 400

    # Find the student in this course
    course = Course.query.get(course_id)
    student = next((s for s in course.enrolled_students if s.student_id == student_id), None)
    if not student:
        return jsonify({"message": "Student not in this course"}), 404

    log = Attendancelog.query.filter_by(session_id=session_id, course_id=course_id, student_id=student_id).first()
    if log:
        # Toggle
        old_status = log.status
        log.status = "absent" if log.status == "present" else "present"
    else:
        # If no log yet, means absent; create as present
        log = Attendancelog(session_id=session_id, course_id=course_id, student_id=student_id,
                            teacher_id=teacher_id, status="present")
        db.session.add(log)
        old_status = "absent"
    db.session.commit()
    return jsonify({"success": True, "from_status": old_status, "to_status": log.status})

# --- VIEW ATTENDANCE FOR ACTIVE SESSION ---
@teacher_bp.route('/teacher/course/<int:course_id>/current_session/attendance', methods=['GET'])
@jwt_required()
def current_session_attendance(course_id):
    teacher_id = get_jwt_identity()
    if not is_teacher_of_course(teacher_id, course_id):
        return jsonify({"message": "Access denied"}), 403

    # Find active session for this course
    session = AttendanceSession.query.filter_by(
        course_id=course_id, is_active=True, teacher_id=teacher_id
    ).order_by(AttendanceSession.start_time.desc()).first()

    if not session:
        return jsonify({"message": "No active session found for this course"}), 404

    course = Course.query.get(course_id)
    students = course.enrolled_students.all()
    logs = Attendancelog.query.filter_by(session_id=session.id, course_id=course_id).all()
    attendance_dict = {log.student_id: log for log in logs}

    present = []
    absent = []

    for student in students:
        log = attendance_dict.get(student.student_id)
        if log and log.status == "present":
            present.append({
                "student_id": student.student_id,
                "name": student.name,
                "status": "present",
                "mark_time": log.verification_timestamp.isoformat() if log.verification_timestamp else None
            })
        else:
            absent.append({
                "student_id": student.student_id,
                "name": student.name,
                "status": "absent"
            })

    summary = {
        "session_id": session.id,
        "course_id": course_id,
        "total_students": len(students),
        "num_present": len(present),
        "num_absent": len(absent),
        "start_time": session.start_time.isoformat() if session.start_time else None
    }

    return jsonify({
        "summary": summary,
        "present": present,
        "absent": absent
    })

@teacher_bp.route('/teacher/course/<int:course_id>/sessions', methods=['GET'])
@jwt_required()
def get_all_sessions(course_id):
    teacher_id = get_jwt_identity()
    if not is_teacher_of_course(teacher_id, course_id):
        return jsonify({"message": "Access denied"}), 403

    sessions = AttendanceSession.query.filter_by(course_id=course_id).order_by(AttendanceSession.session_number).all()
    course = Course.query.get(course_id)
    students = {s.student_id: s for s in course.enrolled_students.all()}

    sessions_with_attendance = []
    for s in sessions:
        logs = Attendancelog.query.filter_by(session_id=s.id, course_id=course_id).all()
        present_ids = [log.student_id for log in logs if log.status == "present"]
        absent_ids = set(students.keys()) - set(present_ids)

        present_students = [
            {
                "student_id": sid,
                "name": students[sid].name
            }
            for sid in present_ids if sid in students
        ]
        absent_students = [
            {
                "student_id": sid,
                "name": students[sid].name
            }
            for sid in absent_ids if sid in students
        ]

        sessions_with_attendance.append({
            "session_id": s.id,
            "session_number": s.session_number,
            "is_active": s.is_active,
            "status": s.status,
            "start_time": s.start_time.isoformat() if s.start_time else None,
            "end_time": s.end_time.isoformat() if s.end_time else None,
            "present_students": present_students,
            "absent_students": absent_students,
            "total_present": len(present_students),
            "total_absent": len(absent_students),
        })

    return jsonify({
        "course_id": course_id,
        "sessions": sessions_with_attendance
    })