import numpy as np
import cv2
import matplotlib.pyplot as plt

from onnx_engine import OnnxEngine

def classification_process(batch: np.ndarray, engine: OnnxEngine):
    tensor = batch[0][0]
    top5_indices = tensor.argsort()[-5:][::-1]
    msg = ''
    for top5_index in top5_indices:
        name = engine.get_class_name(top5_index)
        msg += f"{name} {tensor[top5_index]:.2f}. "
    
    print (msg)

def detect_process(batch: np.ndarray, engine: OnnxEngine, orig_img: np.ndarray, conf_thres: float = 0.5):
    tensor = batch[0][0]
    
    scale_x = orig_img.shape[1] / 640
    scale_y = orig_img.shape[0] / 640

    predictions = np.transpose(tensor, (1, 0))  
    result = []

    boxes_cv = []
    confidences = []

    for predic in predictions:
        conf_values = predic[4:]
        index = np.argmax(conf_values)

        conf = conf_values[index]
        if conf > conf_thres:
            cx, cy, w, h = predic[0], predic[1], predic[2], predic[3]
            x1 = int((cx - w / 2) * scale_x)
            y1 = int((cy - h / 2) * scale_y)
            x2 = int((cx + w / 2) * scale_x)
            y2 = int((cy + h / 2) * scale_y)            
            predict_item = {
                "class": engine.get_class_name(index),
                "conf": conf,
                "bbox": [x1, y1, x2, y2]
            }
            result.append(predict_item)

            x1, y1, x2, y2 = predict_item["bbox"]
            w = x2 - x1
            h = y2 - y1
            boxes_cv.append([x1, y1, w, h])
            confidences.append(float(predict_item["conf"]))

    # To avoid the multidetection on single object
    indices = cv2.dnn.NMSBoxes(boxes_cv, confidences, score_threshold=conf_thres, nms_threshold=0.45)
    result = [result[i] for i in indices]

    for predict in result:
        print(f"Predicted class: {predict['class']} Confidience: {predict['conf']:.2f}")
        bbox = predict["bbox"]
        x1, y1, x2, y2 = bbox#int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(orig_img, f"{predict['class']} {predict['conf']:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Detection", orig_img)
    cv2.imwrite("Detection.png", orig_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def visualize_proto_build_up(mask_protos, mask_coeffs, bbox=None, steps=8, image_size=(640, 640)):
    fig, axs = plt.subplots(2, steps // 2 + 1, figsize=(20, 8))
    axs = axs.flatten()
    
    combined = np.zeros((160, 160), dtype=np.float32)

    for i in range(steps):
        contribution = mask_coeffs[i] * mask_protos[i]
        combined += contribution

        mask_contrib = 1 / (1 + np.exp(-contribution))  # sigmoid
        mask_resized = cv2.resize(mask_contrib, image_size)

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            # mask_display = np.ones(image_size, dtype=np.float32)
            mask_display = np.ones((image_size[1], image_size[0]), dtype=np.float32)
            mask_display[y1:y2, x1:x2] = mask_resized[y1:y2, x1:x2]
            
        else:
            mask_display = mask_resized

        x1, y1, x2, y2 = bbox
        axs[i].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                   edgecolor='lime', linewidth=1, fill=False))        
        axs[i].imshow(mask_display, cmap='gray', vmin=0, vmax=1.0)
        axs[i].set_title(f"proto[{i}] × coeff[{mask_coeffs[i]:.2f}]")
        axs[i].axis('off')

    # Финальная маска
    full_mask = np.tensordot(mask_coeffs, mask_protos, axes=([0], [0]))
    full_prob = 1 / (1 + np.exp(full_mask))
    # full_prob = 1 / (1 + np.exp(-full_mask))
    full_prob_resized = cv2.resize(full_prob, image_size)

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        final_mask_display = np.ones((image_size[1], image_size[0]), dtype=np.float32)
        final_mask_display[y1:y2, x1:x2] = full_prob_resized[y1:y2, x1:x2]
    else:
        final_mask_display = full_prob_resized

    axs[-1].imshow(final_mask_display, cmap='gray', vmin=0, vmax=1)
    axs[-1].set_title("Final mask (sigmoid)")
    axs[-1].axis('off')

    plt.tight_layout()
    plt.show()

def segment_process(batch: np.ndarray, engine: OnnxEngine, orig_img: np.ndarray, conf_thres: float = 0.5):
    tensor = batch[0][0]
    mask_protos = batch[1][0]  

    scale_x = orig_img.shape[1] / 640
    scale_y = orig_img.shape[0] / 640

    predictions = np.transpose(tensor, (1, 0))  
    result = []

    boxes_cv = []
    confidences = []

    for predic in predictions:
        conf_values = predic[4:84]
        index = np.argmax(conf_values)

        conf = conf_values[index]
        if conf > conf_thres:
            cx, cy, w, h = predic[0], predic[1], predic[2], predic[3]
            x1 = int((cx - w / 2) * scale_x)
            y1 = int((cy - h / 2) * scale_y)
            x2 = int((cx + w / 2) * scale_x)
            y2 = int((cy + h / 2) * scale_y)            
            predict_item = {
                "class": engine.get_class_name(index),
                "conf": conf,
                "bbox": [x1, y1, x2, y2],
                "mask_coeffs": predic[84:]
            }
            result.append(predict_item)

            x1, y1, x2, y2 = predict_item["bbox"]
            w = x2 - x1
            h = y2 - y1
            boxes_cv.append([x1, y1, w, h])
            confidences.append(float(predict_item["conf"]))

    # To avoid the multidetection on single object
    indices = cv2.dnn.NMSBoxes(boxes_cv, confidences, score_threshold=conf_thres, nms_threshold=0.45)
    result = [result[i] for i in indices]

    for det in result:
        x1, y1, x2, y2 = det["bbox"]
        coeffs = det["mask_coeffs"]  # shape: (32,)
        
        visualize_proto_build_up(mask_protos, coeffs, det["bbox"], steps=8, image_size=(orig_img.shape[1], orig_img.shape[0]))
        # visualize_proto_build_up(mask_protos, coeffs, steps=8)

        # mask = sum_i (coeffs[i] * proto[i]) → shape: (160, 160)
        mask = np.tensordot(coeffs, mask_protos, axes=([0], [0]))  # (32,) @ (32, 160, 160)

        # Sigmoid → [0, 1]
        mask = 1 / (1 + np.exp(-mask))

        # Resize to original image size
        mask = cv2.resize(mask, (orig_img.shape[1], orig_img.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Crop to bbox
        mask_crop = mask[y1:y2, x1:x2]

        # Optionally threshold the mask
        mask_bin = (mask_crop > 0.5).astype(np.uint8)

        # (Optional) Draw filled mask on image
        if det["class"] == "person":
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)
        mask_rgb = np.zeros_like(orig_img[y1:y2, x1:x2])
        mask_rgb[mask_bin == 1] = color
        orig_img[y1:y2, x1:x2] = cv2.addWeighted(orig_img[y1:y2, x1:x2], 0.7, mask_rgb, 0.3, 0)

        # Draw bbox and label
        cv2.rectangle(orig_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(orig_img, f"{det['class']} {det['conf']:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
    cv2.imshow("Detection", orig_img)
    cv2.imwrite("Detection.png", orig_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def pose_process(batch: np.ndarray, orig_img: np.ndarray, conf_thres: float = 0.5):
    tensor = batch[0][0]


    scale_x = orig_img.shape[1] / 640
    scale_y = orig_img.shape[0] / 640

    predictions = np.transpose(tensor, (1, 0))  
    result = []
    for index, predict in enumerate(predictions):
        conf = predict[4]
        if conf < conf_thres:
            continue
            
        cx, cy, w, h = predict[:4]

        x1 = int((cx - w / 2) * scale_x)
        y1 = int((cy - h / 2) * scale_y)
        x2 = int((cx + w / 2) * scale_x)
        y2 = int((cy + h / 2) * scale_y)    

        kpts = predict[5:].reshape(-1, 3)  # 17 x (x, y, score)   
        kpts[:, 0] = kpts[:, 0] * scale_x
        kpts[:, 1] = kpts[:, 1] * scale_y

        predict_item = {
            'bbox': [x1, y1, x2, y2],
            'conf': conf,
            'keypoints': kpts
        } 
        result.append(predict_item)     


    boxes = [p['bbox'] for p in result]
    scores = [p['conf'] for p in result]

    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=conf_thres, nms_threshold=0.45)

    result = [result[i] for i in indices]

    skeleton = [
        (5, 6), (5, 7), (7, 9),     # левая рука
        (6, 8), (8, 10),            # правая рука
        (11, 12), (5, 11), (6, 12), # торс
        (11, 13), (13, 15),         # левая нога
        (12, 14), (14, 16),         # правая нога
        (0, 1), (0, 2),             # нос-глаза
        (1, 3), (2, 4)              # глаза-уши
    ]

    for det in result:
        x1, y1, x2, y2 = map(int, det['bbox'])
        conf = det['conf']
        keypoints = det['keypoints']  

        cv2.rectangle(orig_img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
        cv2.putText(orig_img, f"{conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        for (x, y, score) in keypoints:
            if score > conf_thres:
                cv2.circle(orig_img, (int(x), int(y)), radius=3, color=(0, 0, 255), thickness=-1)

        for i, j in skeleton:
            if keypoints[i][2] > conf_thres and keypoints[j][2] > conf_thres:
                pt1 = (int(keypoints[i][0]), int(keypoints[i][1]))
                pt2 = (int(keypoints[j][0]), int(keypoints[j][1]))
                cv2.line(orig_img, pt1, pt2, color=(255, 0, 128), thickness=2)

    cv2.imshow("Detection", orig_img)
    cv2.imwrite("Detection.png", orig_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

