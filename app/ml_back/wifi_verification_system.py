from app.models import AttendanceSession, db

class WifiVerificationSystem:
    def get_teacher_ip_for_session(self, session_id):
        """
        Retrieve the teacher_ip from AttendanceSession for the given session_id.
        """
        session = AttendanceSession.query.filter_by(id=session_id, is_active=True).first()
        if session:
            return session.teacher_ip
        return None

    def verify_connection_strength(self, session_id, student_ip):
        """
        Compares student IP to teacher IP for the session.
        """
        teacher_ip = self.get_teacher_ip_for_session(session_id)
        if not teacher_ip:
            return {
                "success": False,
                "message": "No active teaching session found",
                "connection_strength": "none",
                "strength_label": "none"
            }
        if student_ip == teacher_ip:
            return {
                "success": True,
                "message": "Connection is strong, IP match",
                "connection_strength": "strong",
                "strength_label": "strong"
            }
        else:
            return {
                "success": False,
                "message": "Connection is weak, IP mismatch",
                "connection_strength": "weak",
                "strength_label": "weak",
                "teacher_ip": teacher_ip
            }