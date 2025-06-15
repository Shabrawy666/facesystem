import os
import cv2
import numpy as np

from core.models.face_recognition import FaceRecognitionSystem
from core.models.liveness_detection import LivenessDetector
from core.models.image_processor import ImagePreprocessor
from app.models import WifiSession, db

try:
    from core.utils.config import Config
    STORED_IMAGES_DIR = Config.STORED_IMAGES_DIR
except Exception:
    # Fallback: Default to local project 'stored_images'
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    STORED_IMAGES_DIR = os.path.join(SCRIPT_DIR, '..', 'stored_images')

# --- Singleton System Loader ---
_system = None
_liveness = None

def get_face_system():
    """
    Returns a singleton instance of FaceRecognitionSystem.
    """
    global _system
    if _system is None:
        _system = FaceRecognitionSystem()
    return _system

def get_liveness_system():
    """
    Returns a singleton instance of LivenessDetector.
    """
    global _liveness
    if _liveness is None:
        _liveness = LivenessDetector()
    return _liveness

# --- Face Verification for Attendance ---
def verify_attendance_backend(student_id: str, file_path: str):
    """
    Main backend API for verifying attendance using student's uploaded image.
    - Checks liveness and face match.
    - Returns verification dict result.
    """
    image = cv2.imread(file_path)
    if image is None:
        return {"success": False, "message": "Failed to load image"}

    frs = get_face_system()
    result = frs.verify_student(student_id, image)
    data = getattr(result, "data", {}) or {}
    return {
        "success": getattr(result, "success", False),
        "confidence_score": getattr(result, "confidence_score", 0.0),
        "message": getattr(result, "error_message", "Verification successful") if not getattr(result, "success", False) else "Verification successful",
        "liveness_score": data.get("liveness_score"),
        "distance": data.get("distance"),
        "threshold_used": data.get("threshold_used"),
        "verification_time": getattr(result, "verification_time", None),
        # Optionally include other details for debugging/frontend
        "encodings_compared": data.get("encodings_compared"),
    }

# --- Student Face Registration (API) ---
def register_face_backend(student_id: str, file_path: str):
    """
    Register a new face: processes and encodes the submitted image, saves encoding to DB and the cropped image file.
    Returns: {success, encoding, message}
    """
    img = cv2.imread(file_path)
    if img is None:
        return {"success": False, "message": "Failed to load image"}

    frs = get_face_system()
    encoding_result = frs.get_face_encoding_for_storage(img, student_id=student_id)
    if not encoding_result.get("success") or encoding_result.get("encoding") is None:
        return {
            "success": False,
            "message": encoding_result.get("message", "Failed to register face")
        }

    processed = encoding_result.get("preprocessed", img)
    # Save cropped/processed face image
    os.makedirs(STORED_IMAGES_DIR, exist_ok=True)
    image_path = os.path.join(STORED_IMAGES_DIR, f"{student_id}.jpg")
    cv2.imwrite(image_path, processed)

    return {
        "success": True,
        "encoding": encoding_result.get("encoding"),
        "message": "Face registered and saved.",
        "face_quality": encoding_result.get("quality_score"),
        "image_path": image_path
    }

# --- Liveness Detection (standalone, for testing) ---
def run_liveness_detection(file_path: str):
    img = cv2.imread(file_path)
    if img is None:
        return {"success": False, "message": "Invalid image"}
    liveness = get_liveness_system()
    result = liveness.analyze(img)
    return dict(result)

# --- Image Preprocessing (standalone, for diagnostics/debug) ---
def preprocess_face_image(file_path: str):
    img = cv2.imread(file_path)
    if img is None:
        return None
    return ImagePreprocessor.preprocess_image(img)

# --- Batch Verification (for admin/testing, e.g., for analytics) ---
def batch_verify_images(image_data_list):
    """
    image_data_list: [{"student_id": ..., "file_path": ...}, ...]
    Returns: list of result dicts (verdict, scores)
    """
    frs = get_face_system()
    results = []
    for entry in image_data_list:
        student_id = entry.get("student_id")
        path = entry.get("file_path")
        img = cv2.imread(path)
        if not student_id or img is None:
            results.append({"student_id": student_id, "success": False, "message": "Missing student or image"})
            continue
        res = frs.verify_student(student_id, img)
        data = getattr(res, 'data', {}) or {}
        results.append({
            "student_id": student_id,
            "success": getattr(res, "success", False),
            "confidence_score": getattr(res, 'confidence_score', 0.0),
            "liveness_score": data.get("liveness_score"),
            "message": getattr(res, 'error_message', "Verified") if not getattr(res, "success", False) else "Verified"
        })
    return results

# --- System Info ---
def get_system_info():
    """
    Returns system info (can be implemented in FaceRecognitionSystem).
    """
    sys = get_face_system()
    if hasattr(sys, "get_performance_metrics"):
        return sys.get_performance_metrics()
    return {}

__all__ = [
    "verify_attendance_backend",
    "register_face_backend",
    "run_liveness_detection",
    "preprocess_face_image",
    "batch_verify_images",
    "get_system_info",
]

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