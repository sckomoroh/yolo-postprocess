from yolo_onnx_engine import YoloOnnxEngine


engine = YoloOnnxEngine()
engine.init_model(YoloOnnxEngine.SEGMENT)
result = engine.infer(imagepath="data/input.jpeg", vis=True)

pass