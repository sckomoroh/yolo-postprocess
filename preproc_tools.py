import numpy as np
import cv2
from PIL import Image

def preprocess_image(img: np.ndarray, size: tuple[int, int]):
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    return img

def preprocess_yolo_cls_with_pil(img_bgr: np.ndarray, size=(224, 224)) -> np.ndarray:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    pil_img = pil_img.resize(size, resample=Image.Resampling.LANCZOS)

    img = np.array(pil_img).astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)  # HWC → CHW
    img = np.expand_dims(img, axis=0)
    return img
