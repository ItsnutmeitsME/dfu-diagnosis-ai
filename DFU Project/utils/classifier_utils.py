import torch
from torchvision import transforms, models
from PIL import Image

CLASS_NAMES = ['0', '1', '2', '3', '4', '5']

def load_model(model_path="models/classifier.pth"):
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = torch.nn.Linear(model.last_channel, 6)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def predict_class(model, tensor):
    with torch.no_grad():
        output = model(tensor)
        probs = torch.nn.functional.softmax(output, dim=1).squeeze().tolist()
        pred_class = torch.argmax(output).item()
    return CLASS_NAMES[pred_class], [round(p, 4) for p in probs]

