import os
import cv2
import sys
import argparse
import numpy as np
from rknnlite.api import RKNNLite

# Configuration parameters
OBJ_THRESH = 0.25
NMS_THRESH = 0.45
IMG_SIZE = (640, 640)
CLASSES = ("person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", 
           "truck", "boat", "traffic light", "fire hydrant", "stop sign", 
           "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", 
           "cow", "elephant", "bear", "zebra", "giraffe", "backpack", 
           "umbrella", "handbag", "tie", "suitcase", "frisbee", 
           "skis", "snowboard", "sports ball", "kite", 
           "baseball bat", "baseball glove", "skateboard",
           "surfboard", "tennis racket", "bottle",
           # Add more classes as needed
           )

def filter_boxes(boxes, box_confidences, box_class_probs):
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
    # Assuming input_data is structured correctly for YOLO model output.
    # Implement your post-processing logic here based on your model's output format.
    
    # Example of dummy processing:
    boxes, classes, scores = [], [], []  # Replace with actual processing logic.
    
    return np.array(boxes), np.array(classes), np.array(scores)

def draw(image, boxes, scores, classes):
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = [int(_b) for _b in box]
        cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, f'{CLASSES[cl]} {score:.2f}', (left, top - 6), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

def main(model_path):
    rknn = RKNNLite()
    
    # Load RKNN model
    print('Loading model...')
    rknn.load_rknee(model_path)
    
    # Initialize runtime environment
    print('Initializing runtime...')
    rknn.init_runtime()

    # Open webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_resized = cv2.resize(frame, IMG_SIZE)
        input_data = img_resized.astype(np.float32) / 255.0
        
        # Run inference
        outputs = rknn.inference(inputs=[input_data])
        
        # Post-process outputs to get bounding boxes and class predictions
        boxes, classes, scores = post_process(outputs)

        # Draw results on the frame
        draw(frame, boxes, scores, classes)

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