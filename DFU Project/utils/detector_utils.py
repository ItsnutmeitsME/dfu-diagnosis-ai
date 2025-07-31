from inference_sdk import InferenceHTTPClient
from PIL import Image
import os

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="zWcf3wYl1bELz9rj7A8Y"
)

def detect_and_crop(image_path, output_folder):
    result = client.infer(image_path, model_id="dfu-detection-b7oab/1")
    image = Image.open(image_path).convert("RGB")
    predictions = result["predictions"]

    cropped_paths = []
    for i, pred in enumerate(predictions):
        cx, cy = pred["x"], pred["y"]
        w, h = pred["width"], pred["height"]
        left = int(cx - w / 2)
        top = int(cy - h / 2)
        right = int(cx + w / 2)
        bottom = int(cy + h / 2)
        crop = image.crop((left, top, right, bottom))
        crop_path = os.path.join(output_folder, f"crop_{i}.jpg")
        crop.save(crop_path)
        cropped_paths.append(crop_path)
    return cropped_paths

