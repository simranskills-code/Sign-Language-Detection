import torch
import numpy as np
import cv2
from PIL import Image
import sys
import os

# Add YOLOv5 repo to path
YOLO_PATH = os.path.join(os.getcwd(), "yolov5")
if YOLO_PATH not in sys.path:
    sys.path.append(YOLO_PATH)

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.augmentations import letterbox
from yolov5.utils.general import non_max_suppression
from yolov5.utils.torch_utils import select_device

# Manually set model path (inside yolov5 folder)
MODEL_PATH = os.path.join("yolov5", "runs", "train", "exp2", "weights", "best.pt")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found at {MODEL_PATH}")

# Load model
device = select_device("cpu")
model = DetectMultiBackend(MODEL_PATH, device=device)
stride, names = model.stride, model.names
print("✅ Model classes:", names)

def detect_signs(image_pil):
    img0 = np.array(image_pil)
    img0 = cv2.cvtColor(img0, cv2.COLOR_RGB2BGR)

    # Resize with padding (use 320 since model trained with that size)
    img = letterbox(img0, 320, stride=stride, auto=True)[0]
    img = img.transpose((2, 0, 1))[::-1]  # BGR to RGB, CHW
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float() / 255.0
    img = img.unsqueeze(0)

    # Inference
    pred = model(img)
    pred = non_max_suppression(pred, conf_thres=0.1, iou_thres=0.45)

    detected_classes = set()

    for det in pred:
        if len(det):
            for *xyxy, conf, cls in det:
                class_name = names[int(cls)]
                detected_classes.add(class_name)
                print(f"🔍 Detected: {class_name} with confidence {conf:.2f}")

    return list(detected_classes)
