import sys
import os
import cv2
import numpy as np
import importlib.util

# Dynamically add the Silent-Face-Anti-Spoofing-master/src folder to sys.path
here = os.path.dirname(os.path.abspath(__file__))
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

# Optional: import Student for DB-based helper
try:
    from app.models import Student, bytes_to_numpy_image
    from sqlalchemy.orm.exc import NoResultFound
except ImportError:
    Student, bytes_to_numpy_image = None, None

class LivenessDetector:
    """
    Silent-Face-Anti-Spoofing based liveness detector.
    Usage:
        - Analyze method always expects a numpy BGR image (as loaded with cv2).
        - Use in your backend by supplying face images decoded from DB blobs.
    """

    def __init__(self):
        self.silent_face_path = silent_face_path
        self.original_cwd = os.getcwd()
        os.chdir(self.silent_face_path)

        device_id = -1  # Use CPU (-1) for compatibility

        self.model_dir = os.path.join(self.silent_face_path, "resources", "anti_spoof_models")
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
        os.chdir(self.original_cwd)

    def analyze(self, image: np.ndarray) -> dict:
        """
        Analyze a face image (as a numpy array) using anti-spoofing models.
        Intended for use on DB-decoded images or web-uploaded arrays.
        Returns a dict with keys: live (bool), score, explanation, message.
        """
        try:
            current_dir = os.getcwd()
            os.chdir(self.silent_face_path)

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

            os.chdir(current_dir)

            label = np.argmax(prediction)
            value = prediction[0][label] / 2

            is_live = label == 1  # 1 = real, 0 = fake
            confidence = float(value)

            return {
                "live": is_live,
                "score": confidence,
                "explanation": f"Model prediction: {'Real' if is_live else 'Fake'} ({confidence:.3f})",
                "message": "Live person detected" if is_live else "Spoof detected"
            }

        except Exception as e:
            os.chdir(self.original_cwd)
            print(f"Liveness detection error: {str(e)}")
            return {
                "live": True,  # Fail open
                "score": 0.5,
                "explanation": f"Model error: {str(e)}",
                "message": "Liveness check skipped"
            }

    # ----------- Optional: Helper to analyze directly from DB face image (by student_id) --------------
    @classmethod
    def analyze_student_db_image(cls, student_id: str):
        """
        Optionally, use this to check the stored student DB image's liveness directly.
        Usage: LivenessDetector.analyze_student_db_image(student_id)
        """
        if not Student or not bytes_to_numpy_image:
            return {
                "live": True,
                "score": 0.5,
                "explanation": "Not running in the app context (no DB access).",
                "message": "Liveness check skipped"
            }
        try:
            student = Student.query.filter_by(student_id=student_id).first()
        except Exception:
            return {
                "live": True,
                "score": 0.5,
                "explanation": "Could not query database.",
                "message": "Liveness check skipped"
            }
        if not student or not student.face_image:
            return {
                "live": True,
                "score": 0.5,
                "explanation": "No stored image.",
                "message": "Liveness check failed"
            }
        img = bytes_to_numpy_image(student.face_image)
        detector = cls()
        return detector.analyze(img)