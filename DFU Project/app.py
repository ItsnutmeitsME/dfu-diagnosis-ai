from flask import Flask, render_template, request
import os
from utils.detector_utils import detect_and_crop
from utils.classifier_utils import load_model, preprocess_image, predict_class
from utils.gradcam_utils import generate_gradcam
from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
CROP_FOLDER = 'static/crops'
OUTPUT_FOLDER = 'static/outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CROP_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

model = load_model()

@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    if request.method == "POST":
        file = request.files["image"]
        image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(image_path)

        crop_paths = detect_and_crop(image_path, output_folder=CROP_FOLDER)

        for i, crop_path in enumerate(crop_paths):
            img = Image.open(crop_path).convert("RGB")
            input_tensor = preprocess_image(img)
            predicted_class, probs = predict_class(model, input_tensor)

            cam_path = os.path.join(OUTPUT_FOLDER, f"gradcam_{i}.jpg")
            generate_gradcam(model, input_tensor, cam_path)

            results.append({
                "crop": crop_path,
                "prediction": predicted_class,
                "gradcam": cam_path,
                "probs": probs
            })

    return render_template("index.html", results=results)

if __name__ == "__main__":
    app.run(debug=True)

