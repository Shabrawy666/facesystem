import os

class Config:
    """System configuration settings"""

    # Face Recognition Settings
    FACE_DETECTION_CONFIDENCE = 0.9
    FACE_RECOGNITION_THRESHOLD = 0.4
    IMAGE_SIZE = (224, 224)

    # Always use project root, not process cwd
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Image Storage Paths (safe, reliable, always in project)
    TEMP_IMAGE_DIR = os.path.join(PROJECT_ROOT, "temp_images")
    STORED_IMAGES_DIR = os.path.join(PROJECT_ROOT, "stored_images")

    # Logging Settings
    LOG_FILE = os.path.join(PROJECT_ROOT, "facial_recognition.log")
    LOG_LEVEL = "INFO"