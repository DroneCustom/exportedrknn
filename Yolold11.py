import os
import cv2
import sys
import argparse
import numpy as np
from rknnlite.api import RKNNLite

# Configuration parameters
OBJ_THRESH = 0.25
NMS_THRESH = 0.45
IMG_SIZE = (640, 640)  # (width, height)
CLASSES = (
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", 
    "train", "truck", "boat", "traffic light", "fire hydrant", 
    "stop sign", "parking meter", "bench", "bird", "cat", 
    "dog", "horse", "sheep", "cow", "elephant", 
    # Add more classes as needed
)

def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with object threshold."""
    box_confidences = box_confidences.reshape(-1)
    candidate, class_num = box_class_probs.shape
    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)
    
    _class_pos = np.where(class_max_score * box_confidences >= OBJ_THRESH)
    
    scores = (class_max_score * box_confidences)[_class_pos]
    boxes = boxes[_class_pos]
    classes = classes[_class_pos]
    
    return boxes, classes, scores

def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes."""
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    
    areas = w * h
    order = scores.argsort()[::-1]
    
    keep = []
    
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])
        
        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        
        inter = w1 * h1
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]

    return np.array(keep)

def post_process(input_data):
    """Post-process the output from the model."""
    boxes, scores, classes_conf = [], [], []
    
    default_branch=3
    pair_per_branch = len(input_data) // default_branch
    
    for i in range(default_branch):
        # Process each branch output here (you might need to adjust this based on your model output)
        # Assuming input_data has the correct format and dimensions.
        # This is a placeholder for actual processing logic.
        
        # Example processing logic:
        # boxes.append(process_boxes(input_data[pair_per_branch * i]))
        # classes_conf.append(input_data[pair_per_branch * i + 1])
        # scores.append(np.ones_like(input_data[pair_per_branch * i + 1][:, :1, :, :], dtype=np.float32))
        
        pass  # Replace with actual processing logic

    # Flatten and concatenate results from all branches
    # Implement your flattening logic here

    return boxes, classes_conf, scores

def draw(image, boxes, scores, classes):
    """Draw bounding boxes and labels on the image."""
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = [int(_b) for _b in box]
        
        cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, f'{CLASSES[cl]} {score:.2f}', (left, top - 6), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def main(model_path):
    rknn = RKNNLite()
    
    print('Loading model...')
    rknn.load_rknee(model_path)
    
    print('Initializing runtime...')
    rknn.init_runtime()

    cap = cv2.VideoCapture(0)  # Open webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        img_resized = cv2.resize(frame, IMG_SIZE)  # Resize to model's expected input size
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        
        input_data = img_rgb.transpose((2, 0, 1))  # Change to (channels, height, width)
        input_data = input_data[np.newaxis, :, :, :] / 255.0  # Add batch dimension and normalize
        
        outputs = rknn.inference(inputs=[input_data])  # Run inference
        
        boxes, classes_confidence_scores = post_process(outputs)  # Process outputs
        
        if boxes is not None:
            draw(frame.copy(), boxes, classes_confidence_scores)  # Draw results on frame
        
        cv2.imshow('Live Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    rknn.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLO Object Detection with RKNN')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the YOLO model in .rknn format')
    
    args = parser.parse_args()
    
    main(args.model_path)
