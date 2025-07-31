# dfu-diagnosis-ai
A real-time, AI-powered system for detecting and classifying diabetic foot ulcers (DFUs) using medical images. Combines object detection with severity grading via the Wagner scale, deployed via Flask with integrated Grad-CAM visualizations for clinical interpretability.
# Diabetic Foot Ulcer Detection & Classification (DFU-AI)

This project is an end-to-end AI pipeline for real-time detection and classification of Diabetic Foot Ulcers (DFUs) using deep learning. It leverages object detection to localize ulcer regions, followed by classification into Wagner grades (0–5) to assess severity. The model is deployed via a web-based Flask application and includes interpretability features using Grad-CAM.

---

## Key Features

- **Two-Stage Pipeline**: Combines RF-DETR for ulcer detection and MobileNetV2 for Wagner grade classification.
- **Custom Annotated Dataset**: Expert-verified annotations from multiple clinical sources.
- **Transfer Learning**: Utilizes pre-trained ImageNet models for effective learning on limited medical data.
- **Explainability**: Integrated Grad-CAM visualizations highlight key regions influencing predictions.
- **Web Deployment**: Lightweight, real-time Flask interface for clinicians and field workers.
- **Ethically Developed**: Fully anonymized, secure data pipeline with patient privacy at its core.

---

## Model Architecture

### 1. Object Detection
- **Model**: RF-DETR (Transformer-based)
- **Task**: Detect and localize ulcer regions in foot images.

### 2. Image Classification
- **Model**: MobileNetV2 (Fine-tuned)
- **Task**: Classify cropped ulcer images into Wagner Grades 0 to 5.

---

## Performance

- **Detection mAP**: ~69.6%
- **Classification Accuracy**: ~75% on validation set
- **F1 Scores**: Balanced performance across most classes, especially Grades 2–4.
- **Explainability**: Grad-CAM heatmaps focused on clinically relevant ulcer regions.
<img width="451" height="387" alt="image" src="https://github.com/user-attachments/assets/17eee65c-9c44-4336-9319-9184f3f3d84d" />
<img width="302" height="231" alt="image" src="https://github.com/user-attachments/assets/e1e3942b-2131-4a09-bfd8-940f22771f62" />
<img width="435" height="245" alt="image" src="https://github.com/user-attachments/assets/8948abea-66aa-4802-95a4-a51d186a9116" />
<img width="472" height="264" alt="image" src="https://github.com/user-attachments/assets/a5a6ca76-31c3-4d9e-ba01-c59642abc2dc" />

---

## Deployment

A Flask web app allows:
1. Upload of foot images.
2. Automatic detection of DFUs.
3. Classification using Wagner scale.
4. Visualization with Grad-CAM overlays.
