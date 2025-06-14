import os
import numpy as np
import cv2
import sys

# Detect project root and build the path to Silent-Face-Anti-Spoofing
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
SFS_PATH = os.path.join(PROJECT_ROOT, 'Silent-Face-Anti-Spoofing-master')
SRC_PATH = os.path.join(SFS_PATH, 'src')

# Add only src/ to sys.path so imports work regardless of cwd
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from anti_spoof_predict import AntiSpoofPredict
from generate_patches import CropImage
from utility import parse_model_name

class LivenessDetector:
    """Using Silent-Face-Anti-Spoofing model"""

    def __init__(self):
        device_id = -1  # Use CPU

        self.model_dir = os.path.join(SFS_PATH, "resources", "anti_spoof_models")
        self.model_files = [
            os.path.join(self.model_dir, "2.7_80x80_MiniFASNetV2.pth"),
            os.path.join(self.model_dir, "4_0_0_80x80_MiniFASNetV1SE.pth")
        ]

        for model_file in self.model_files:
            if not os.path.exists(model_file):
                print(f"❌ Model file not found: {model_file}")
            else:
                print(f"✅ Model file found: {model_file}")

        self.model = AntiSpoofPredict(device_id)
        self.image_cropper = CropImage()

    def analyze(self, image: np.ndarray) -> dict:
        """Analyze using pre-trained anti-spoofing model (stateless, cwd-independent)"""
        try:
            image_bbox = self.model.get_bbox(image)
            prediction = np.zeros((1, 3))

            for model_file in self.model_files:
                if not os.path.exists(model_file):
                    print(f"Skipping missing model: {model_file}")
                    continue

                model_name = os.path.basename(model_file)
                h_input, w_input, model_type, scale = parse_model_name(model_name)

                param = {
                    "org_img": image,
                    "bbox": image_bbox,
                    "scale": scale,
                    "out_w": w_input,
                    "out_h": h_input,
                    "crop": True,
                }
                if scale is None:
                    param["crop"] = False

                img = self.image_cropper.crop(**param)
                prediction += self.model.predict(img, model_file)

            label = np.argmax(prediction)
            value = prediction[0][label]/2
            is_live = label == 1  # 1 = real, 0 = fake
            confidence = float(value)
            return {
                "live": is_live,
                "score": confidence,
                "explanation": f"Model prediction: {'Real' if is_live else 'Fake'} ({confidence:.3f})",
                "message": "Live person detected" if is_live else "Spoof detected"
            }
        except Exception as e:
            print(f"Liveness detection error: {str(e)}")
            return {
                "live": True,  # Fail open for business flow
                "score": 0.5,
                "explanation": f"Model error: {str(e)}",
                "message": "Liveness check skipped"
            }