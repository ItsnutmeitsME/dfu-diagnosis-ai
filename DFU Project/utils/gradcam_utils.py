import torch
import numpy as np
import cv2

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
        for h in self.hook_handles:
            h.remove()

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
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        cam -= cam.min()
        cam /= cam.max()
        return cam

def generate_gradcam(model, input_tensor, output_path):
    target_layer = dict([*model.named_modules()])["features.18"]
    grad_cam = GradCAM(model, target_layer)
    cam = grad_cam.generate_cam(input_tensor)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    input_np = input_tensor.squeeze().permute(1, 2, 0).numpy()
    input_np = (input_np * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
    input_np = np.clip(input_np * 255.0, 0, 255).astype(np.uint8)
    overlay = cv2.addWeighted(cv2.cvtColor(input_np, cv2.COLOR_RGB2BGR), 0.6, heatmap, 0.4, 0)
    cv2.imwrite(output_path, overlay)

