from flask import Flask, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
from sort.sort import Sort
from util import get_car
from fast_plate_ocr import ONNXPlateRecognizer
import os

app = Flask(__name__)

# Initialize models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('./license_plate_detector.pt')
mot_tracker = Sort()
ocr_model = ONNXPlateRecognizer('global-plates-mobile-vit-v2-model')

# Directories for uploaded and processed images
UPLOAD_FOLDER = './uploads'
PROCESSED_FOLDER = './processed'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)


@app.route('/recognize', methods=['POST'])
def recognize_license_plate():
    # Check if the request contains an image file
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(image_path)

    # Read the image
    frame = cv2.imread(image_path)

    # Detect vehicles
    detections = coco_model(frame)[0]
    detections_ = []
    vehicles = [2, 3, 5, 7]  # COCO vehicle classes
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicles:
            detections_.append([x1, y1, x2, y2, score])
    track_ids = mot_tracker.update(np.asarray(detections_))

    # Detect license plates
    license_plates = license_plate_detector(frame)[0]
    recognized_plates = []
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate

        # Assign license plate to car
        xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

        if car_id != -1:
            # Crop license plate
            license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]
            result = ocr_model.run(license_plate_crop)

            # Clean the result to remove non-alphanumeric characters
            cleaned_result = [''.join(char for char in item if char.isalnum()) for item in result]
            recognized_plates.extend(cleaned_result)

    # Save processed image (example: draw bounding boxes or annotations)
    processed_image_path = os.path.join(PROCESSED_FOLDER, f'processed_{file.filename}')
    cv2.imwrite(processed_image_path, frame)

    # Return the result
    return jsonify({
        'processed_image': processed_image_path,
        'recognized_plates': recognized_plates
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
