from ultralytics import YOLO
from sort.sort import *
import numpy as np
import cv2
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))

def read_video(video_path):
    return cv2.VideoCapture(video_path)

def load_model():
    return YOLO("YOLOv8s.pt")

def detect_vehicles(frame, model):
    results = model(frame) 
    detections = []

    for result in results:
        for r in result.boxes:  
            x1, y1, x2, y2 = r.xyxy[0].tolist()  
            conf = r.conf[0].item()  
            cls = int(r.cls[0].item())  

            if cls in [2, 5]:  
                detections.append([x1, y1, x2, y2, conf]) 

    return detections

def count_vehicles(tracks, counting_points, tracked_vehicles, car_tolerance=80, bus_tolerance=25):
    car_count = 0
    bus_count = 0

    for track in tracks:
        x1_box, y1_box, x2_box, y2_box, object_id = track[:5]
        centroid_x, centroid_y = int((x1_box + x2_box) / 2), int((y1_box + y2_box) / 2)

        for i in range(len(counting_points) - 1):
            start_point = counting_points[i]
            end_point = counting_points[i + 1]

            dx = end_point[0] - start_point[0]
            dy = end_point[1] - start_point[1]

            m = dy / dx
            c = start_point[1] - m * start_point[0]
            y_on_line = m * centroid_x + c

            if (start_point[0] - bus_tolerance <= centroid_x <= end_point[0] + bus_tolerance and
                min(start_point[1], end_point[1]) - bus_tolerance <= centroid_y <= max(start_point[1], end_point[1]) + bus_tolerance):
                if object_id not in tracked_vehicles:
                    tracked_vehicles[object_id] = True
                    bus_count += 1

            elif (min(start_point[1], end_point[1]) - car_tolerance <= centroid_y <= max(start_point[1], end_point[1]) + car_tolerance and
                    abs(centroid_y - y_on_line) < car_tolerance):
                if object_id not in tracked_vehicles:
                    tracked_vehicles[object_id] = True
                    car_count += 1

    return car_count, bus_count

def visualize_results(frame, tracks, vehicle_counts, counting_points):
    line_color = (0, 255, 0)
    
    pts = np.array(counting_points, np.int32)
    pts = pts.reshape((-1, 1, 2))
    
    cv2.polylines(frame, [pts], isClosed=False, color=line_color, thickness=10)
    
    for track in tracks:
        x1, y1, x2, y2, object_id = track[:5]
        color = (0, 255, 0) 
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)
        cv2.putText(frame, f"ID: {int(object_id)}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    car_count = vehicle_counts['cars']
    bus_count = vehicle_counts['buses']

    cv2.putText(frame, f"Cars: {car_count}, Buses: {bus_count}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, line_color, 3)

    return frame

def main(input_video_path, output_video_path):
    cap = read_video(input_video_path)
    model = load_model()
    mot_tracker = Sort()
    
    counting_points = [
        (-80, 505), # The gate 1
        (400, 610), # The gate 2
        (680, 665), # The gate 3
        (920, 715), # The gate 4
        (1160, 770), # The gate 5
        (1400, 830), # The gate 6
        (1640, 880), # The gate 7
        (1880, 940), # The gate 8
    ]

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (frame_width, frame_height))
    
    vehicle_counts = {'cars': 0, 'buses': 0}
    tracked_vehicles = {}
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        detections = detect_vehicles(frame, model)
        
        if len(detections) > 0:
            dets = np.array(detections)
            tracks = mot_tracker.update(dets)
        else:
            tracks = np.empty((0, 5))
        
        car_count, bus_count = count_vehicles(tracks, counting_points, tracked_vehicles)
        
        vehicle_counts['cars'] += car_count
        vehicle_counts['buses'] += bus_count
        
        annotated_frame = visualize_results(frame, tracks, vehicle_counts, counting_points)
        
        out.write(annotated_frame)

        resized_frame = cv2.resize(annotated_frame, (960, 540))
        
        cv2.imshow('Vehicle Counting', resized_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    input_video = os.path.join(ROOT, 'assets', 'toll_gate.mp4')
    output_video = os.path.join(ROOT, 'assets', 'toll_gate_result.mp4')
    main(input_video, output_video)
