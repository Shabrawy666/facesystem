import os

class Config:
    """System configuration settings"""

    # Face Recognition Settings
    FACE_DETECTION_CONFIDENCE = 0.9
    FACE_RECOGNITION_THRESHOLD = 0.4
    IMAGE_SIZE = (224, 224)

    # Project root for paths
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Use env var override if present, default to Fly.io mount or project folder
    STORED_IMAGES_DIR = os.environ.get('STORED_IMAGES_DIR', '/app/stored_images')
    TEMP_IMAGE_DIR = os.path.join(PROJECT_ROOT, "temp_images")

    # Logging
    LOG_FILE = os.path.join(PROJECT_ROOT, "facial_recognition.log")
    LOG_LEVEL = "INFO"