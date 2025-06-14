import os
import cv2
import numpy as np
from sqlalchemy.orm.exc import NoResultFound
from app.models import Student, bytes_to_numpy_image  # Use imported helper from models.py

from core.models.face_recognition import FaceRecognitionSystem
from core.models.liveness_detection import LivenessDetector
from core.models.image_processor import ImagePreprocessor
try:
    from core.utils.config import Config
    STORED_IMAGES_DIR = Config.STORED_IMAGES_DIR
except Exception:
    # Fallback if Config is missing or misconfigured
    STORED_IMAGES_DIR = os.path.join(os.getcwd(), 'stored_images')

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

def verify_attendance_backend(student_id: str, image: np.ndarray = None, file_path: str = None):
    """
    Main backend API for verifying attendance using student's uploaded image.
    - Checks liveness and face match.
    - Returns verification/dict result.
    """
    if image is not None:   # Web upload or DB-to-numpy
        img = image
    elif file_path is not None:
        img = cv2.imread(file_path)
    else:
        return {"success": False, "message": "No image data provided."}
    if img is None:
        return {"success": False, "message": "Failed to load image"}

    frs = get_face_system()
    result = frs.verify_student(student_id, img)
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

def verify_attendance_from_db(student_id: str, captured_img: np.ndarray):
    """
    Verifies a submitted image (from web) against the student's reference face image stored in the database,
    and only returns success if BOTH face recognition and anti-spoof liveness check PASS.
    """
    from app.ml_backend import get_liveness_system  # Avoid circular import if in ml_backend.py

    try:
        student = Student.query.filter_by(student_id=student_id).one()
    except NoResultFound:
        return {"success": False, "message": "Student not found."}

    if not student.face_image:
        return {"success": False, "message": "No stored face image in DB for this student."}

    reference_img = bytes_to_numpy_image(student.face_image)
    if reference_img is None:
        return {"success": False, "message": "Stored facial image is corrupted or unreadable."}

    # 1. FACE RECOGNITION
    frs = get_face_system()
    face_result = frs.verify_student(student_id, captured_img)
    face_data = getattr(face_result, "data", {}) or {}

    face_success = bool(getattr(face_result, "success", False))
    confidence_score = float(getattr(face_result, "confidence_score", 0.0))
    distance = float(face_data.get("distance")) if face_data.get("distance") is not None else None
    threshold = float(face_data.get("threshold_used")) if face_data.get("threshold_used") is not None else None

    # 2. LIVENESS/ANTI-SPOOF DETECTION: Always enforced!
    liveness = get_liveness_system()
    liveness_result = liveness.analyze(captured_img)
    liveness_score = float(liveness_result.get("score", 0.0))
    is_live = bool(liveness_result.get("live", True))
    liveness_pass = is_live and liveness_score > 0.8  # You can adjust this threshold!

    # 3. Final result is only success if BOTH pass:
    overall_success = face_success and liveness_pass

    return {
        "success": overall_success,
        "face_match_pass": face_success,
        "confidence_score": confidence_score,
        "distance": distance,
        "threshold": threshold,
        "liveness_score": liveness_score,
        "liveness_pass": liveness_pass,
        "model_used": face_data.get("model_used", "Facenet"),
        "message": (
            "Verification and liveness passed." if overall_success else
            "Verification failed: "
            + ("" if face_success else "Face match failed. ")
            + ("" if liveness_pass else "Anti-spoof/liveness failed (not a live face).")
        )
    }

# --- Student Face Registration (API) ---
def register_face_backend(student_id: str, image: np.ndarray = None, file_path: str = None):
    """
    Register a new face: processes and encodes the submitted image, saves encoding and cropped face to DB.
    Returns: {success, encoding, message}
    """
    if image is not None:
        img = image
    elif file_path is not None:
        img = cv2.imread(file_path)
    else:
        return {"success": False, "message": "No image data provided."}
    if img is None:
        return {"success": False, "message": "Failed to load image"}

    frs = get_face_system()
    encoding_result = frs.get_face_encoding_for_storage(img, student_id=student_id)
    if not encoding_result.get("success") or encoding_result.get("encoding") is None:
        return {
            "success": False,
            "message": encoding_result.get("message", "Failed to register face")
        }

    # Save processed/cropped face image as JPEG to DB (replace disk saving!)
    processed = encoding_result.get("preprocessed", img)
    try:
        student = Student.query.filter_by(student_id=student_id).one()
    except NoResultFound:
        return {
            "success": False,
            "message": "Student not found in DB"
        }
    # Encode processed (cropped/enhanced face) as JPEG for DB storage
    success, face_buf = cv2.imencode('.jpg', processed)
    if not success:
        return {
            "success": False,
            "message": "Failed to encode cropped face for DB storage."
        }
    student.face_image = face_buf.tobytes()
    from app import db
    db.session.commit()

    # (optional) Also store/overwrite face_encoding in DB for quick retrieval, if using single encoding.
    student.face_encoding = encoding_result.get("encoding")
    db.session.commit()

    return {
        "success": True,
        "encoding": encoding_result.get("encoding"),
        "message": "Face registered and saved in DB.",
        "face_quality": encoding_result.get("quality_score")
    }

# --- Liveness Detection (standalone, for testing) ---
def run_liveness_detection(student_id: str = None, image: np.ndarray = None, file_path: str = None):
    '''
    You can test liveness for a given student (from DB) or on an uploaded/test image.
    '''
    if student_id:
        student = Student.query.filter_by(student_id=student_id).first()
        if not student or not student.face_image:
            return {"success": False, "message": "Student face image not found in DB."}
        img = bytes_to_numpy_image(student.face_image)
    elif image is not None:
        img = image
    elif file_path is not None:
        img = cv2.imread(file_path)
    else:
        return {"success": False, "message": "No student_id or image data provided."}
    if img is None:
        return {"success": False, "message": "Failed to load/process image"}
    liveness = get_liveness_system()
    result = liveness.analyze(img)
    return dict(result)

# --- Image Preprocessing (standalone, for diagnostics/debug) ---
def preprocess_face_image(student_id: str = None, image: np.ndarray = None, file_path: str = None):
    if student_id:
        student = Student.query.filter_by(student_id=student_id).first()
        if not student or not student.face_image:
            return {"success": False, "message": "Student face image not found in DB."}
        img = bytes_to_numpy_image(student.face_image)
    elif image is not None:
        img = image
    elif file_path is not None:
        img = cv2.imread(file_path)
    else:
        return {"success": False, "message": "No student_id or image data provided."}
    if img is None:
        return {"success": False, "message": "Failed to load/process image"}
    return ImagePreprocessor.preprocess_image(img)

# --- Batch Verification (for admin/testing, e.g., for analytics) ---
def batch_verify_images(image_data_list):
    """
    image_data_list: [{"student_id": ..., "image": ...}, ...]
    "image" can be a numpy array (from upload or DB).
    Returns: list of result dicts (verdict, scores)
    """
    frs = get_face_system()
    results = []
    for entry in image_data_list:
        student_id = entry.get("student_id")
        file_path = entry.get("file_path")
        img = entry.get("image")
        if not img and file_path:
            img = cv2.imread(file_path)
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
    "verify_attendance_from_db",
    "register_face_backend",
    "run_liveness_detection",
    "preprocess_face_image",
    "batch_verify_images",
    "get_system_info",
]