import numpy as np

import onnx

from onnx_engine import OnnxEngine

from common_tools import load_image
from preproc_tools import preprocess_image, preprocess_yolo_cls_with_pil
from postproc_tools import classification_process, detect_process, segment_process, pose_process


# Infer section

models_map = {
    "detec":{
        "model": "models/yolov8n.onnx",
        "labels": "models/labels-dtc.txt",
    },
    "pose":{
        "model": "models/yolov8n-pose.onnx",
        "labels": "models/labels-pose.txt",
    },
    "segment":{
        "model": "models/yolov8n-seg.onnx",
        "labels": "models/labels-seg.txt",
    },
    "classification":{
        "model": "models/yolov8n-cls.onnx",
        "labels": "models/labels-cls.txt",
    }
}

engine = OnnxEngine(models_map["detec"]["model"])
engine.read_names(models_map["detec"]["labels"])
engine.dump_inputs()
engine.dump_outputs()

img = load_image('data/input.jpeg')

# classification only (see the output of the dump inputs)
# input = preprocess_image(img, (224, 224))

input = preprocess_image(img, (640, 640))

tensor = { "images" : input}

output_batch = engine.infer(tensor)

# classification_process(batch=output_batch, engine=engine)
# pose_process(batch=output_batch, orig_img=img, conf_thres=0.5)
# segment_process(batch=output_batch, engine=engine, orig_img=img, conf_thres=0.5)
detect_process(batch=output_batch, engine=engine, orig_img=img, conf_thres=0.5)

pass
