import numpy as np

import onnx

from onnx_engine import OnnxEngine

from common_tools import load_image
from preproc_tools import preprocess_image, preprocess_yolo_cls_with_pil
from postproc_tools import classification_process, detect_process, segment_process, pose_process

# debug section
model = onnx.load("models/yolov8n-cls.onnx")
for node in model.graph.node[:10]:
    print(node.op_type, node.name)

# Infer section
engine = OnnxEngine('models/yolov8n-cls.onnx')
engine.read_names('models/labels-cls.txt')
engine.dump_inputs()
engine.dump_outputs()

img = load_image('data/input.jpeg')
input = preprocess_image(img, (224, 224))
# input = preprocess_yolo_cls_with_pil(img, (224, 224))

tensor = { "images" : input}

output_batch = engine.infer(tensor)

classification_process(output_batch, engine)
pass
