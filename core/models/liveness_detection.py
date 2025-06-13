import sys
import os
import cv2
import numpy as np
import importlib.util

# Dynamically add the Silent-Face-Anti-Spoofing-master/src folder to sys.path
here = os.path.dirname(os.path.abspath(__file__))
# Was: project_root = os.path.abspath(os.path.join(here, '../../..'))
# Should be just one up:
project_root = os.path.abspath(os.path.join(here, '..', '..'))
silent_face_path = os.path.join(project_root, "Silent-Face-Anti-Spoofing-master")
src_path = os.path.join(silent_face_path, "src")

def import_from_src_by_file(module_filename, class_name):
    module_path = os.path.join(src_path, module_filename)
    spec = importlib.util.spec_from_file_location(module_filename[:-3], module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)

AntiSpoofPredict = import_from_src_by_file("anti_spoof_predict.py", "AntiSpoofPredict")
CropImage = import_from_src_by_file("generate_patches.py", "CropImage")
parse_model_name = import_from_src_by_file("utility.py", "parse_model_name")

class LivenessDetector:
    """Using Silent-Face-Anti-Spoofing model"""

    def __init__(self):
        self.silent_face_path = silent_face_path
        self.original_cwd = os.getcwd()
        os.chdir(self.silent_face_path)

        device_id = -1  # Use CPU (-1) for compatibility

        # Full paths to model files
        self.model_dir = os.path.join(self.silent_face_path, "resources", "anti_spoof_models")
        self.model_files = [
            os.path.join(self.model_dir, "2.7_80x80_MiniFASNetV2.pth"),
            os.path.join(self.model_dir, "4_0_0_80x80_MiniFASNetV1SE.pth")
        ]

        # Check if model files exist
        for model_file in self.model_files:
            if not os.path.exists(model_file):
                print(f"❌ Model file not found: {model_file}")
            else:
                print(f"✅ Model file found: {model_file}")

        self.model = AntiSpoofPredict(device_id)
        self.image_cropper = CropImage()
        os.chdir(self.original_cwd)

    def analyze(self, image: np.ndarray) -> dict:
        """Analyze using pre-trained anti-spoofing model"""
        try:
            current_dir = os.getcwd()
            os.chdir(self.silent_face_path)

            # Prepare image
            image_bbox = self.model.get_bbox(image)
            prediction = np.zeros((1, 3))

            # Get prediction using FULL PATHS
            for model_file in self.model_files:
                if not os.path.exists(model_file):
                    print(f"Skipping missing model: {model_file}")
                    continue

                model_name = os.path.basename(model_file)  # Get just the filename for parse_model_name
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

                # Use FULL PATH for prediction
                prediction += self.model.predict(img, model_file)  # Use full path instead of just filename

            os.chdir(current_dir)

            # Calculate final result
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
            # Change back to original directory in case of error
            os.chdir(self.original_cwd)
            print(f"Liveness detection error: {str(e)}")
            return {
                "live": True,  # Fail open
                "score": 0.5,
                "explanation": f"Model error: {str(e)}",
                "message": "Liveness check skipped"
            }