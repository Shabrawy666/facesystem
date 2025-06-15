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

# --- Student Face Registration (API) ---
def register_face_backend(student_id: str, file_path: str):
    """
    Register a new face: processes and encodes the submitted image, returns encoding to be saved in DB.
    """
    img = cv2.imread(file_path)
    if img is None:
        return {"success": False, "message": "Failed to load image"}
    
    print("REGISTER: img shape", img.shape, "sum", np.sum(img))
    frs = get_face_system()
    encoding_result = frs.get_face_encoding_for_storage(img, student_id=student_id)
    if not encoding_result.get("success") or encoding_result.get("encoding") is None:
        return {
            "success": False,
            "message": encoding_result.get("message", "Failed to register face")
        }
    print("REGISTER: encoding sum", np.sum(encoding_result["encoding"]))

    processed = encoding_result.get("preprocessed", img)
    # Save cropped/processed face image for diagnostics or admin review
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

# --- Multi-image Registration (API utility) ---
def register_faces_backend(student_id: str, file_paths: list):
    frs = get_face_system()
    encodings = []
    messages = []
    for idx, path in enumerate(file_paths):
        img = cv2.imread(path)
        if img is None:
            messages.append(f"Image {idx+1}: Failed to load image")
            continue
        print(f"REGISTER {idx+1}: img shape", img.shape, "sum", np.sum(img))
        encoding_result = frs.get_face_encoding_for_storage(img, student_id=student_id)
        if encoding_result.get("success") and encoding_result.get("encoding") is not None:
            encodings.append(encoding_result["encoding"])
            print(f"REGISTER {idx+1}: encoding sum", np.sum(encoding_result["encoding"]))
        else:
            messages.append(f"Image {idx+1}: {encoding_result.get('message', 'Failed to register face')}")
    if encodings:
        return {
            "success": True,
            "encodings": encodings,
            "message": "Face encodings registered.",
            "details": messages
        }
    else:
        return {
            "success": False,
            "encodings": [],
            "message": "No valid face encodings found.",
            "details": messages
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

# --- System Info ---
def get_system_info():
    sys = get_face_system()
    if hasattr(sys, "get_performance_metrics"):
        return sys.get_performance_metrics()
    return {}

__all__ = [
    "register_face_backend",
    "register_faces_backend",
    "run_liveness_detection",
    "preprocess_face_image",
    "get_system_info",
]