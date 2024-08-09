import sys
from ultralytics import YOLO
import cv2
import easyocr
import os


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

            if detected_class_id == class_id and confidence > confidence_threshold:
                # print(f"Vehicle detected: x1={x1}, y1={y1}, x2={x2}, y2={y2}, conf={confidence:.4f}")

                vehicle_img = image[int(y1):int(y2), int(x1):int(x2)].copy()
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': confidence,
                    'vehicle_img': vehicle_img
                })

    return detections

def extract_plate_text(model, vehicle_img, vehicle_bbox):
    results = model(vehicle_img)
    reader = easyocr.Reader(['en'])
    vehicle_data = {
        'vehicle_bbox': vehicle_bbox,
        'plates': []
    }

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            plate_crop = vehicle_img[y1:y2, x1:x2].copy()

            plate_crop_gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            _, plate_crop_thresh = cv2.threshold(plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

            text = reader.readtext(plate_crop_thresh)
            plate_text = ''.join(item[1] for item in text).upper()

            vehicle_data['plates'].append({
                'plate_bbox': (x1, y1, x2, y2),
                'plate_text': plate_text
            })

            # print("Licence plate:", plate_text)

    return vehicle_data

def draw_and_save(image_path, vehicle_data_list, save_dir, frame_num):
    image = load_image(image_path)
    save_path = os.path.join(save_dir, f"{frame_num}.jpg")

    for vehicle in vehicle_data_list:
        vehicle_bbox = vehicle['vehicle_bbox']
        for plate in vehicle['plates']:
            plate_bbox = plate['plate_bbox']
            plate_text = plate['plate_text']

            full_img_x1 = int(vehicle_bbox[0] + plate_bbox[0])
            full_img_y1 = int(vehicle_bbox[1] + plate_bbox[1])
            full_img_x2 = int(vehicle_bbox[0] + plate_bbox[2])
            full_img_y2 = int(vehicle_bbox[1] + plate_bbox[3])

            cv2.rectangle(image, (full_img_x1, full_img_y1), (full_img_x2, full_img_y2), (0, 255, 0), 2)
            cv2.putText(image, plate_text, (full_img_x1, full_img_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imwrite(save_path, image)

def extract_frames_from_video(video_path, output_folder, interval):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Video could not be opened!")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    current_frame = 0
    frame_interval = int(fps * interval)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame % frame_interval == 0:
            frame_number = current_frame // frame_interval
            frame_filename = os.path.join(output_folder, f"{frame_number}.png")
            cv2.imwrite(frame_filename, frame)
            # print(f"Frame {frame_number} kaydedildi.")

        current_frame += 1

    cap.release()
    print("Video successfully split into frames!")

def frame_folder_detect_save(image_folder, detect_result_folder):
    current_dir = os.path.dirname(__file__)
    car_detect_model_ = os.path.join(current_dir, 'yolov8n.pt')
    license_plate_model_path = os.path.join(current_dir, 'license_plate_model.pt')
    car_model = load_model(car_detect_model_)
    license_plate_model = load_model(license_plate_model_path)

    if not os.path.exists(detect_result_folder):
        os.makedirs(detect_result_folder)

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort(key=lambda x: int(x.split('.')[0]))

    frame_count = 0
    for image in images:
        img_path = os.path.join(image_folder, image)
        curr_img = load_image(img_path)
        vehicle_detections = detect_vehicles(car_model, curr_img)
        detected_vehicles_data = []

        for i, detection in enumerate(vehicle_detections):
            vehicle_img = detection['vehicle_img']
            if detection['confidence'] >= 0.8:
                vehicle_data = extract_plate_text(license_plate_model, vehicle_img, detection['bbox'])
                detected_vehicles_data.append(vehicle_data)

        frame_count += 1
        draw_and_save(img_path, detected_vehicles_data, detect_result_folder, frame_count)

def create_video_from_images(image_folder, video_name, fps=30, size=(1920, 1080)):
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    images.sort(key=lambda x: int(x.split('.')[0]))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_name, fourcc, fps, size)

    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        frame = cv2.resize(frame, size)
        video.write(frame)

    video.release()
    print(f"The video '{video_name}' was successfully created!")

def main(video_path):
    extracted_frames_folder = "extracted_frames"
    result_frames_folder = "result_frames_folder"

    extract_frames_from_video(video_path, extracted_frames_folder, interval=1)
    frame_folder_detect_save(extracted_frames_folder, result_frames_folder)
    create_video_from_images(result_frames_folder, "detected_traffic.mp4", fps=1, size=(1920, 1080))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py image_path")
    else:
        video_path = sys.argv[1]
        main(video_path)
