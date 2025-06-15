from app.models import WifiSession, db

class WifiVerificationSystem:

    def create_session(self, teacher_id, hall_id, wifi_ssid, teacher_ip, session_id):
        # End any existing active session for this hall/teacher/hall
        WifiSession.query.filter_by(hall_id=hall_id, is_active=True).update({"is_active": False})
        wifi_session = WifiSession(
            session_id=session_id,
            teacher_id=teacher_id,
            hall_id=hall_id,
            wifi_ssid=wifi_ssid,
            teacher_ip=teacher_ip,
            is_active=True
        )
        db.session.add(wifi_session)
        db.session.commit()

    def get_teacher_ip_for_session(self, session_id):
        wifi_session = WifiSession.query.filter_by(session_id=session_id, is_active=True).first()
        if wifi_session:
            return wifi_session.teacher_ip
        return None

    def verify_connection_strength(self, session_id, student_ip):
        """Compares student IP to teacher IP for the session."""
        teacher_ip = self.get_teacher_ip_for_session(session_id)
        if not teacher_ip:
            return {"success": False, "message": "No active teaching session found", "connection_strength": "none", "strength_label": "none"}
        if student_ip == teacher_ip:
            return {"success": True, "message": "Connection is strong, IP match", "connection_strength": "strong", "strength_label": "strong"}
        else:
            return {"success": False, "message": "Connection is weak, IP mismatch", "connection_strength": "weak", "strength_label": "weak", "teacher_ip": teacher_ip}
    
    def end_session(self, session_id, teacher_id):
        session = WifiSession.query.filter_by(session_id=session_id, teacher_id=teacher_id, is_active=True).first()
        if session:
            session.is_active = False
            db.session.commit()