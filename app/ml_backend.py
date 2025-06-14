import os
import cv2
import numpy as np

from core.models.face_recognition import FaceRecognitionSystem
from core.models.liveness_detection import LivenessDetector
from core.models.image_processor import ImagePreprocessor

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
    global _system
    if _system is None:
        _system = FaceRecognitionSystem()
    return _system

def get_liveness_system():
    global _liveness
    if _liveness is None:
        _liveness = LivenessDetector()
    return _liveness

# --- Face Verification for Attendance ---
def verify_attendance_backend(student_id: str, file_path: str):
    """
    Main backend API for verifying attendance using student's uploaded image.
    - Checks liveness and face match.
    - Returns verification/dict result.
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
        "verification_time": getattr(result, "verification_time", None)
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
    return get_face_system().get_system_info()

__all__ = [
    "verify_attendance_backend",
    "register_face_backend",
    "run_liveness_detection",
    "preprocess_face_image",
    "batch_verify_images",
    "get_system_info",
    ]