import numpy as np
import cv2

from onnx_engine import OnnxEngine

from common_tools import load_image
from preproc_tools import preprocess_image, preprocess_yolo_cls_with_pil


class YoloOnnxEngine:
    DETECT = 0
    CLASSIFY = 1
    SEGMENT = 2
    POSE = 3

    MIN_CONF = 0.25

    def __init__(self):
        self.model_map = {
            self.DETECT: {
                "model": "models/yolov8n.onnx",
                "labels": "models/labels-dtc.txt",
                "process": self.detec_postprocess,
                "visualize": self.visualize_detection,
            },
            self.POSE: {
                "model": "models/yolov8n-pose.onnx",
                "labels": "models/labels-pose.txt",
                "process": self.pose_postprocess,
                "visualize": self.visualize_pose,
            },
            self.SEGMENT: {
                "model": "models/yolov8n-seg.onnx",
                "labels": "models/labels-seg.txt",
                "process": self.segment_postprocess,
                "visualize": self.visualize_segmentation,
            },
            self.CLASSIFY: {
                "model": "models/yolov8n-cls.onnx",
                "labels": "models/labels-cls.txt",
                "process": self.classify_postprocess,
                "visualize": self.visualize_classification,
            },
        }

        self.engine = None

    def init_model(self, model_type: int):
        if model_type not in range(self.DETECT, self.POSE + 1):
            raise ValueError(
                "Invalid model type. Must be one of DETECT, CLASSIFY, SEGMENT, POSE."
            )

        self.model_info = self.model_map[model_type]
        self.engine = OnnxEngine(self.model_info["model"])
        self.engine.read_names(self.model_info["labels"])

    def infer(self, imagepath: str, vis: bool = False):
        tensor = self.preprocess(imagepath=imagepath)

        model_input = self.engine.get_inputs()[0]
        input_name = model_input.name

        input_tensor = {input_name: tensor}
        output_batch = self.engine.infer(tensor=input_tensor)

        result = self.model_info["process"](output_batch=output_batch)

        if vis == True:
            vis_img = self.model_info["visualize"](imagepath=imagepath, results=result)
            cv2.imshow("prediction", vis_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows

        return result

    def preprocess(self, imagepath: str):
        image = load_image(filename=imagepath)

        self.image_shape = image.shape[:2]

        model_input = self.engine.get_inputs()[0]

        image = preprocess_image(img=image, size=model_input.shape[2:4])

        return image

    def detec_postprocess(self, output_batch: np.ndarray):
        tensor = output_batch[0][0]
        tensor = np.transpose(tensor)

        model_input = self.engine.get_inputs()[0]
        input_width, input_height = model_input.shape[2:4]

        scale_x = self.image_shape[1] / input_width
        scale_y = self.image_shape[0] / input_height

        result = []

        for prediction in tensor:
            bbox = prediction[:4]
            confidences = prediction[4:]
            class_index = np.argmax(confidences)
            confidence = confidences[class_index]
            if confidence > self.MIN_CONF:
                cx, cy, w, h = bbox
                x1 = int((cx - w / 2) * scale_x)
                y1 = int((cy - h / 2) * scale_y)
                x2 = int((cx + w / 2) * scale_x)
                y2 = int((cy + h / 2) * scale_y)

                predict_item = {
                    "bbox": [x1, y1, x2, y2],
                    "class_index": class_index,
                    "confidence": confidence,
                    "class_name": self.engine.get_class_name(index=class_index),
                }

                result.append(predict_item)

        boxes = [p["bbox"] for p in result]
        scores = [p["confidence"] for p in result]

        indices = cv2.dnn.NMSBoxes(
            boxes, scores, score_threshold=self.MIN_CONF, nms_threshold=0.45
        )
        result = [result[i] for i in indices.flatten()]

        return {"detect": result}

    def classify_postprocess(self, output_batch: np.ndarray):
        tensor = output_batch[0][0]
        index = np.argmax(tensor)
        confidence = tensor[index]
        class_name = self.engine.get_class_name(index=index)

        return {
            "classification": {
                "index": index,
                "confidence": confidence,
                "class_name": class_name,
            }
        }

    def segment_postprocess(self, output_batch: np.ndarray):
        prediction_conf = output_batch[0][0]
        tensor_kpmap = output_batch[1][0]

        prediction_conf = np.transpose(prediction_conf)

        model_input = self.engine.get_inputs()[0]
        input_width, input_height = model_input.shape[2:4]

        scale_x = self.image_shape[1] / input_width
        scale_y = self.image_shape[0] / input_height

        result = []

        for predic in prediction_conf:
            conf_values = predic[4:84]
            index = np.argmax(conf_values)

            conf = conf_values[index]
            if conf < self.MIN_CONF:
                continue

            cx, cy, w, h = predic[0], predic[1], predic[2], predic[3]
            x1 = int((cx - w / 2) * scale_x)
            y1 = int((cy - h / 2) * scale_y)
            x2 = int((cx + w / 2) * scale_x)
            y2 = int((cy + h / 2) * scale_y)
            predict_item = {
                "class": self.engine.get_class_name(index),
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
                "mask": predic[84:],
                "class_index": index,
            }

            result.append(predict_item)

            x1, y1, x2, y2 = predict_item["bbox"]
            w = x2 - x1
            h = y2 - y1

        boxes = [p["bbox"] for p in result]
        scores = [p["confidence"] for p in result]

        indices = cv2.dnn.NMSBoxes(
            boxes, scores, score_threshold=self.MIN_CONF, nms_threshold=0.45
        )
        result = [result[i] for i in indices.flatten()]

        for det in result:
            x1, y1, x2, y2 = det["bbox"]
            coeffs = det["mask"]  # shape: (32,)

            mask = np.tensordot(
                coeffs, tensor_kpmap, axes=([0], [0])
            )  # (32,) @ (32, 160, 160)

            mask = 1 / (1 + np.exp(-mask))

            mask = cv2.resize(
                mask,
                (self.image_shape[1], self.image_shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

            det["mask"] = mask

        return {"segment": result}

    def pose_postprocess(self, output_batch: np.ndarray):
        tensor = output_batch[0][0]

        tensor = np.transpose(tensor)

        model_input = self.engine.get_inputs()[0]
        input_width, input_height = model_input.shape[2:4]

        scale_x = self.image_shape[1] / input_width
        scale_y = self.image_shape[0] / input_height

        result = []

        for prediction in tensor:
            conf = prediction[4]
            if conf < self.MIN_CONF:
                continue

            cx, cy, w, h = prediction[:4]

            x1 = int((cx - w / 2) * scale_x)
            y1 = int((cy - h / 2) * scale_y)
            x2 = int((cx + w / 2) * scale_x)
            y2 = int((cy + h / 2) * scale_y)

            kpts = prediction[5:].reshape(-1, 3)  # 17 x (x, y, score)
            kpts[:, 0] = kpts[:, 0] * scale_x
            kpts[:, 1] = kpts[:, 1] * scale_y

            mask = kpts[:, 2] < 0.1
            kpts[mask] = np.nan

            predict_item = {
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "keypoints": kpts,
            }

            result.append(predict_item)

        boxes = [p["bbox"] for p in result]
        scores = [p["confidence"] for p in result]

        indices = cv2.dnn.NMSBoxes(
            boxes, scores, score_threshold=self.MIN_CONF, nms_threshold=0.45
        )

        result = [result[i] for i in indices]

        return {"pose": result}

    def visualize_classification(self, imagepath: str, results: dict):
        image = load_image(filename=imagepath)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        info = results["classification"]
        label = f'{info["class_name"]}: {info["confidence"]:.2f}'

        cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        return image

    def visualize_detection(self, imagepath: str, results: dict):
        image = load_image(filename=imagepath)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        for info in results["detect"]:
            x1, y1, x2, y2 = info["bbox"]
            label = f'{info["class_name"]}: {info["confidence"]:.2f}'

            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
            )

        return image

    def visualize_segmentation(self, imagepath: str, results: dict):
        image = load_image(filename=imagepath)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        for seg in results["segment"]:
            mask = seg["mask"]
            x1, y1, x2, y2 = seg["bbox"]

            mask_crop = mask[y1:y2, x1:x2]
            mask_bin = (mask_crop > 0.5).astype(np.uint8)
            mask_rgb = np.zeros_like(image[y1:y2, x1:x2])

            # I'm too lazy to describe colors for all classes
            if seg["class"] == "person":
                color = (0, 255, 0)
            elif seg["class"] == "tie":
                color = (255, 0, 0)
            else:
                color = (0, 0, 255)

            mask_rgb[mask_bin == 1] = color
            image[y1:y2, x1:x2] = cv2.addWeighted(image[y1:y2, x1:x2], 0.7, mask_rgb, 0.3, 0)

            label = f'{seg["class"]}: {seg["confidence"]:.2f}'
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )

        return image

    def visualize_pose(self, imagepath: str, results: dict):
        image = load_image(filename=imagepath)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        for pose in results["pose"]:
            x1, y1, x2, y2 = pose["bbox"]
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

            kpts = pose["keypoints"]
            for x, y, score in kpts:
                if not np.isnan(x) and not np.isnan(y):
                    cv2.circle(image, (int(x), int(y)), 3, (0, 0, 255), -1)

        return image
