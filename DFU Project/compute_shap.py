import torch,sys
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import shap
import numpy as np

# Load your trained model
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = torch.nn.Linear(model.last_channel, 6)  # Replace with correct number of classes

# Load the saved state dict
state_dict = torch.load("best_mobilenetv2_dfu_5cats.pth")
model.load_state_dict(state_dict)
model.eval()

# Define preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load an image and preprocess
img = Image.open("../../wagner_classification/test/2/288.jpg").convert("RGB")
img_tensor = transform(img).unsqueeze(0)
img_tensor.requires_grad_()

def predict(x):
    with torch.no_grad():
        return model(x)

# Use GradientExplainer for image models
explainer = shap.GradientExplainer(model, img_tensor)
shap_values = explainer.shap_values(img_tensor)

# Convert tensor to numpy image
img_for_plot = img_tensor.squeeze().permute(1, 2, 0).detach().numpy()
img_for_plot = img_for_plot * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]  # unnormalize
img_for_plot = np.clip(img_for_plot, 0, 1)

# Get prediction from model
with torch.no_grad():
    logits = model(img_tensor)
    predicted_class = logits.argmax(dim=1).item()  # e.g., 0 to 5

print(predicted_class)
# Extract SHAP values for the predicted class
shap_val = shap_values[..., predicted_class]  # shape: (3, 224, 224)
shap_val = shap_val[0]
# print(shap_val.shape)
shap_val = np.transpose(shap_val, (1, 2, 0))

# Ensure it's a float32 NumPy array
shap_val = shap_val.astype(np.float32)

# Prepare image for plot
# If img_for_plot is (3, 224, 224), transpose it too
if img_for_plot.shape[0] == 3:
    img_plot_ready = np.transpose(img_for_plot, (1, 2, 0))  # (224, 224, 3)
else:
    img_plot_ready = img_for_plot  # assume already correct

img_plot_ready = img_plot_ready.astype(np.float32)

# Normalize image if necessary
if img_plot_ready.max() > 1.0:
    img_plot_ready /= 255.0

img_plot_ready = img_plot_ready.astype(np.float32)
'''
print("SHAP shape:", shap_val.shape, "dtype:", shap_val.dtype)
print("IMG shape:", img_plot_ready.shape, "dtype:", img_plot_ready.dtype)
print("Max img:", img_plot_ready.max(), "Min img:", img_plot_ready.min())
'''
import matplotlib.pyplot as plt
# Plot SHAP values
# shap.image_plot(shap_values, np.array([img_for_plot]))
shap.image_plot([shap_val], img_plot_ready)
plt.savefig("shap_explanation_2_288.png")  # Save the figure
