import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import numpy as np
from torchvision import models

# 🔁 Helper: GradCAM class
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []

        self.hook_handles.append(self.target_layer.register_forward_hook(self.save_activation))
        self.hook_handles.append(self.target_layer.register_backward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

    def generate_cam(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output)

        self.model.zero_grad()
        target = output[0, class_idx]
        target.backward()

        gradients = self.gradients[0].cpu().detach().numpy()
        activations = self.activations[0].cpu().detach().numpy()

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_tensor.shape[2], input_tensor.shape[3]))
        cam -= np.min(cam)
        cam /= np.max(cam)
        return cam, class_idx.item()

# 🎯 Your trained model
# Rebuild the architecture
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = torch.nn.Linear(model.last_channel, 6)  # Replace with correct number of classes

# Load the saved state dict
state_dict = torch.load("best_mobilenetv2_dfu_5cats.pth")
model.load_state_dict(state_dict)

# model = torch.load("best_mobilenetv2_dfu_5cats.pth")
model.eval()
model.cuda()

# 👀 Use the last conv layer for Grad-CAM
target_layer = dict([*model.named_modules()])["features.18"]  # For MobileNetV2

# 🧪 Test image
from torchvision import transforms
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

img = Image.open("../../wagner_classification/test/4/208.jpg").convert("RGB")
input_tensor = transform(img).unsqueeze(0).cuda()

# 🔍 Grad-CAM generation
grad_cam = GradCAM(model, target_layer)
cam, pred_class = grad_cam.generate_cam(input_tensor)

# 📊 Overlay CAM on image
img_np = np.array(img)
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
overlay = cv2.addWeighted(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR), 0.6, heatmap, 0.4, 0)

cv2.imwrite("gradcam_overlay_4_208.jpg", overlay)
print("Predicted class:", pred_class)
