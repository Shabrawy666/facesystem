import os
from dataclasses import dataclass
from typing import Tuple

class Config:
        """System configuration settings"""
        # Face Recognition Settings
        FACE_DETECTION_CONFIDENCE = 0.9
        FACE_RECOGNITION_THRESHOLD = 0.4
        IMAGE_SIZE = (224, 224)

        # Image Storage Paths (ensure these are relative to workdir)
        TEMP_IMAGE_DIR = os.path.join(os.getcwd(), "temp_images")
        STORED_IMAGES_DIR = os.path.join(os.getcwd(), "stored_images")

        # Logging Settings
        LOG_FILE = os.path.join(os.getcwd(), "facial_recognition.log")
        LOG_LEVEL = "INFO"
