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

    def verify_wifi_connection(self, session_id, student_wifi_ssid, student_ip):
        wifi_session = WifiSession.query.filter_by(session_id=session_id, is_active=True).first()
        if not wifi_session:
            return {"success": False, "message": "No active teaching session found", "connection_strength": "none", "strength_label": "none"}
        # Example WiFi check: SSID match
        if wifi_session.wifi_ssid != student_wifi_ssid:
            return {"success": False, "message": "SSID mismatch", "connection_strength": "weak", "strength_label": "weak"}
        return {"success": True, "message": "WiFi verification successful", "connection_strength": "strong", "strength_label": "strong"}

    def end_session(self, session_id, teacher_id):
        session = WifiSession.query.filter_by(session_id=session_id, teacher_id=teacher_id, is_active=True).first()
        if session:
            session.is_active = False
            db.session.commit()