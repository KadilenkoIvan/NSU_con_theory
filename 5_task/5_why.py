from ultralytics import YOLO
import multiprocessing
import time
import cv2


def fun_thread_read(path_video: str, frame_queue: multiprocessing.Queue, event_stop: multiprocessing.Event):
    cap = cv2.VideoCapture(path_video)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame!")
            break
        frame_queue.put(frame)
        time.sleep(0.0001)
    event_stop.set()
        

def fun_thread_safe_predict(frame_queue: multiprocessing.Queue, event_stop: multiprocessing.Event):
    local_model = YOLO(model="yolov8n.pt")
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            results = local_model.predict(source=frame, device='cpu')
        if event_stop.is_set():
            print(f'Process {multiprocessing.current_process()} final!')
            break


def main():
    YOLO(model="yolov8n.pt")
    threads = []
    frame_queue = multiprocessing.Queue()
    event_stop = multiprocessing.Event()
    video_path = "./test_crop_crop.mp4"
    thread_read = multiprocessing.Process(target=fun_thread_read, args=(video_path, frame_queue, event_stop, ))
    thread_read.start()
    start_t = time.monotonic()
    for _ in range(5):
        threads.append(multiprocessing.Process(target=fun_thread_safe_predict, args=(frame_queue, event_stop,)))

    for thr in threads:
        thr.start()

    for thr in threads:
        thr.join()

    thread_read.join()
    end_t = time.monotonic()
    print(f'Time: {end_t - start_t}')


if __name__ == "__main__":
    main()