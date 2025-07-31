import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

# --------------------------
# Configuration
# --------------------------
MODEL_PATH = "best_mobilenetv2_dfu_5cats.pth"         # Path to trained model weights
IMAGE_PATH = "../../wagner_classification/test/2/182.jpg"        # Path to the input image
NUM_CLASSES = 6                      # Total number of classes in the model
CLASS_NAMES = ['0', '1', '2', '3', '4', '5']  # Modify as per your dataset

# --------------------------
# Define Your Model
# --------------------------
# Example using MobileNetV2; change this to your actual model if different
from torchvision.models import mobilenet_v2

def get_model(num_classes):
    model = mobilenet_v2(pretrained=False)
    model.classifier[1] = torch.nn.Linear(1280, num_classes)
    return model

# --------------------------
# Preprocess Image
# --------------------------
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet mean
                             [0.229, 0.224, 0.225])  # ImageNet std
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # Add batch dimension

# --------------------------
# Inference Function
# --------------------------
def run_inference(model, image_tensor, class_names):
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1).squeeze()
        predicted_class = torch.argmax(probs).item()
    return predicted_class, probs

# --------------------------
# Main Logic
# --------------------------
if __name__ == "__main__":
    # Load model
    model = get_model(NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))

    # Load and preprocess image
    image_tensor = preprocess_image(IMAGE_PATH)

    # Run inference
    predicted_idx, softmax_probs = run_inference(model, image_tensor, CLASS_NAMES)

    # Display results
    print(f"\n🔍 Predicted Class: {CLASS_NAMES[predicted_idx]} (Index: {predicted_idx})")
    print("📊 Class Probabilities:")
    for idx, prob in enumerate(softmax_probs):
        print(f"  {CLASS_NAMES[idx]}: {prob:.4f}")
