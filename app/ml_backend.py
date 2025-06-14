import os
import numpy as np

from core.models.face_recognition import FaceRecognitionSystem
from core.models.liveness_detection import LivenessDetector
from core.models.image_processor import ImagePreprocessor
try:
    from core.utils.config import Config
    STORED_IMAGES_DIR = Config.STORED_IMAGES_DIR
except Exception:
    STORED_IMAGES_DIR = os.path.join(os.getcwd(), 'stored_images')

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

def verify_attendance_backend(student_id: str, image: np.ndarray):
    """
    Main backend API for verifying attendance using uploaded image (np.ndarray).
    """
    if image is None:
        return {"success": False, "message": "No image array supplied."}
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

def register_face_backend(student_id: str, image: np.ndarray):
    """
    Register a new face: processes and encodes image, saves encoding and processed cropped profile image to disk.
    """
    if image is None:
        return {"success": False, "message": "No image array supplied."}
    frs = get_face_system()
    encoding_result = frs.get_face_encoding_for_storage(image, student_id=student_id)
    if not encoding_result.get("success") or encoding_result.get("encoding") is None:
        return {
            "success": False,
            "message": encoding_result.get("message", "Failed to register face")
        }

    processed = encoding_result.get("preprocessed", image)
    os.makedirs(STORED_IMAGES_DIR, exist_ok=True)
    image_path = os.path.join(STORED_IMAGES_DIR, f"{student_id}.jpg")
    import cv2
    cv2.imwrite(image_path, processed)

    return {
        "success": True,
        "encoding": encoding_result.get("encoding"),
        "message": "Face registered and saved.",
        "face_quality": encoding_result.get("quality_score"),
        "image_path": image_path
    }

def run_liveness_detection(image: np.ndarray):
    if image is None:
        return {"success": False, "message": "No image array supplied."}
    liveness = get_liveness_system()
    result = liveness.analyze(image)
    return dict(result)

def preprocess_face_image(image: np.ndarray):
    if image is None:
        return None
    return ImagePreprocessor.preprocess_image(image)

def batch_verify_images(image_data_list):
    """
    image_data_list: [{"student_id": ..., "image": ...}, ...] where image is np.ndarray
    Returns: list of result dicts (verdict, scores)
    """
    frs = get_face_system()
    results = []
    for entry in image_data_list:
        student_id = entry.get("student_id")
        img = entry.get("image")
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

def get_system_info():
    return get_face_system().get_system_info()