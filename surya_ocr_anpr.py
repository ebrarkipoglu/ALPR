import sys
from ultralytics import YOLO
import cv2
import os
from PIL import Image
from surya.ocr import run_ocr
from surya.model.detection.model import load_model as load_detection_model, load_processor as load_detection_processor
from surya.model.recognition.model import load_model as load_recognition_model
from surya.model.recognition.processor import load_processor as load_recognition_processor


def get_image_text(image, language=["en"]):

    plate_crop_thresh_pil = Image.fromarray(image)

    det_processor, det_model = load_detection_processor(), load_detection_model()
    rec_model, rec_processor = load_recognition_model(), load_recognition_processor()

    # Run the OCR process on the image
    predictions = run_ocr([plate_crop_thresh_pil], [language], det_model, det_processor, rec_model, rec_processor)[0]

    # Extract text from the OCR predictions
    text = [line.text for line in list(predictions)[0][1]]

    return text



def load_image(image_path):
    return cv2.imread(image_path)

def load_model(model_path):
    return YOLO(model_path)

def detect_vehicles(model, image, class_id=2, confidence_threshold=0.5):
    results = model(image)
    detections = []

    for result in results:
        for detection in result.boxes:
            xyxy = detection.xyxy
            conf = detection.conf
            cls = detection.cls
            x1, y1, x2, y2 = xyxy[0].tolist()
            confidence = conf[0].item()
            detected_class_id = cls[0].item()

            # Consider detections with the specified class ID and confidence above the threshold
            if detected_class_id == class_id and confidence > confidence_threshold:
                # print(f"Vehicle detected: x1={x1}, y1={y1}, x2={x2}, y2={y2}, conf={confidence:.4f}")

                # Crop the vehicle image
                vehicle_img = image[int(y1):int(y2), int(x1):int(x2)].copy()
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': confidence,
                    'vehicle_img': vehicle_img
                })

    return detections

def extract_plate_text(model, vehicle_img, vehicle_bbox):
    """Extract text from detected license plates using EasyOCR."""
    results = model(vehicle_img)

    vehicle_data = {
        'vehicle_bbox': vehicle_bbox,
        'plates': []
    }

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Crop the plate region from the image
            plate_crop = vehicle_img[y1:y2, x1:x2].copy()

            # Preprocess the cropped plate image
            plate_crop_gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)

            threshold_value = 128

            # Apply thresholding
            _, plate_crop_thresh = cv2.threshold(plate_crop_gray, threshold_value, 255, cv2.THRESH_BINARY_INV)

            plate_text = get_image_text(plate_crop_gray)
            # Store plate data
            vehicle_data['plates'].append({
                'plate_bbox': (x1, y1, x2, y2),
                'plate_text': plate_text
            })

            # print("Licence plate:", plate_text)
            break

    return vehicle_data

def draw_and_save(image_path, vehicle_data_list, save_dir):

    image = load_image(image_path)
    save_path = os.path.join(save_dir, f"output_image_surya.jpg")

    for vehicle in vehicle_data_list:
        vehicle_bbox = vehicle['vehicle_bbox']
        for plate in vehicle['plates']:
            plate_bbox = plate['plate_bbox']
            plate_text = plate['plate_text']

            full_img_x1 = int(vehicle_bbox[0] + plate_bbox[0])
            full_img_y1 = int(vehicle_bbox[1] + plate_bbox[1])
            full_img_x2 = int(vehicle_bbox[0] + plate_bbox[2])
            full_img_y2 = int(vehicle_bbox[1] + plate_bbox[3])


            plate_text_str = ''.join(plate_text)

            # Draw rectangle around the plate
            cv2.rectangle(image, (full_img_x1, full_img_y1), (full_img_x2, full_img_y2),
                          (0, 255, 0), 2)
            cv2.putText(image, plate_text_str, (full_img_x1, full_img_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2)

    # Save the modified image
    cv2.imwrite(save_path, image)


def img_detect_save(input_file_):
    current_dir = os.path.dirname(__file__)
    input_file = input_file_
    car_detect_model_ = os.path.join(current_dir, 'yolov8n.pt')
    license_plate_model_path = os.path.join(current_dir, 'license_plate_model.pt')
    output_dir = os.path.join(current_dir, 'output_images')

    car_model = load_model(car_detect_model_)
    license_plate_model = load_model(license_plate_model_path)
    img = load_image(input_file)

    vehicle_detections = detect_vehicles(car_model, img)
    detected_vehicles_data = []  # List to store all vehicle data

    for i, detection in enumerate(vehicle_detections):
        vehicle_img = detection['vehicle_img']

        if detection['confidence'] >= 0.8:
            vehicle_data = extract_plate_text(license_plate_model, vehicle_img, detection['bbox'])
            detected_vehicles_data.append(vehicle_data)

    draw_and_save(input_file, detected_vehicles_data, output_dir)



def main():
    if len(sys.argv) != 2:
        print("Usage: python script_name.py image_path")
    else:
        input_file = sys.argv[1]
        img_detect_save(input_file)

if __name__ == "__main__":
    main()
