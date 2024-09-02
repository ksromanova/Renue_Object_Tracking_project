import numpy as np
import cv2
import torch
import sys
import yaml
from collections import defaultdict
from ultralytics import YOLO
import time

def initialize_video(path_video, output_video=None):
    cap = cv2.VideoCapture(path_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if output_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))
    else:
        out = None
    return cap, out, fps, frame_width, frame_height

def process_frame(frame, model, track_history, device, tracker_config_path):
    start_process_time = time.perf_counter()  # Начало замера времени

    frame_np = frame.copy()
    frame = frame.transpose(2, 0, 1)
    frame = torch.from_numpy(frame).float().div(255.0).to(device).unsqueeze(0)

    # Использование ByteTrack трекера с параметрами из файла
    results = model.track(frame_np, persist=True, verbose=False, tracker=tracker_config_path)

    process_time = time.perf_counter() - start_process_time  # Конец замера времени

    track_ids = results[0].boxes.id.cpu().numpy().astype(int) if results[0].boxes.id is not None else []
    boxes = results[0].boxes.xywh.cpu().numpy() if results[0].boxes.xywh is not None else []

    annotated_frame = results[0].plot()

    for box, track_id in zip(boxes, track_ids):
        x, y, w, h = box
        track = track_history[track_id]
        track.append((float(x), float(y)))
        if len(track) > 90:
            track.pop(0)
        
        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)

    return annotated_frame, track_ids, boxes, process_time

def main(path_video, path_model, output_video=None, mot_results_file=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")

    # Настройки трекера ByteTrack
    tracker_config = {
        'tracker_type': 'bytetrack',
        'track_high_thresh': 0.40,
        'track_low_thresh': 0.1,
        'new_track_thresh': 0.50,
        'track_buffer': 10,
        'match_thresh': 0.9,
        'fuse_score': True
    }

    # Сохранение параметров трекера в YAML файл
    tracker_config_path = "tracker_config.yaml"
    with open(tracker_config_path, 'w') as file:
        yaml.dump(tracker_config, file)

    # Инициализация модели
    model = YOLO(path_model, verbose=False).to(device)

    cap, out, fps, frame_width, frame_height = initialize_video(path_video, output_video)

    track_history = defaultdict(lambda: [])
    frame_count = 0
    mot_results = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processing_times = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.resize(frame, (frame_width, frame_height))
        annotated_frame, track_ids, boxes, process_time = process_frame(frame, model, track_history, device, tracker_config_path)

        processing_times.append(process_time)  # Сохранение времени обработки кадра

        if out:  # Сохранение видео, если задан output_video
            out.write(annotated_frame)

        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            x_min = x - w / 2
            y_min = y - h / 2
            mot_results.append((frame_count + 1, track_id, x_min, y_min, w, h, 1, 2, 1.0))

        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        frame_count += 1

    if out:
        out.release()
    cap.release()

    if mot_results_file:
        with open(mot_results_file, 'w') as f:
            for result in mot_results:
                f.write(f"{result[0]},{result[1]},{result[2]},{result[3]},{result[4]},{result[5]},{result[6]},{result[7]},{result[8]}\n")

    avg_processing_time = np.mean(processing_times)
    print(f"\nСреднее время обработки кадра: {avg_processing_time:.4f} секунд")

    if output_video:
        print(f"\nВидео с трекингом сохранено в {output_video}")
    if mot_results_file:
        print(f"Результаты трекинга сохранены в {mot_results_file}")

if __name__ == "__main__":
    if len(sys.argv) == 3:  # Только входное видео и модель
        path_video = sys.argv[1]
        path_model = sys.argv[2]
        main(path_video, path_model)
    elif len(sys.argv) == 5:  # Полный набор аргументов
        path_video = sys.argv[1]
        output_video = sys.argv[2]
        path_model = sys.argv[3]
        mot_results_file = sys.argv[4]
        main(path_video, path_model, output_video, mot_results_file)
    else:
        print("""
Usage:
    1. Для измерения скорости обработки видео без сохранения результата:
       python script.py <path_to_video> <path_to_model>

    2. Для обработки видео с сохранением результата:
       python script.py <path_to_video> <output_video> <path_to_model> <mot_results_file>

    Где:
        <path_to_video>     - путь к входному видеофайлу
        <path_to_model>     - путь к файлу с моделью YOLO
        <output_video>      - (опционально) путь для сохранения обработанного видео
        <mot_results_file>  - (опционально) путь для сохранения результатов трекинга
        """)
        sys.exit(1)
