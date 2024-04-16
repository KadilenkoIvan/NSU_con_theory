from ultralytics import YOLO
import multiprocessing
import argparse
import time
import cv2

def fun_thread_read(path_video: str, frame_queue: multiprocessing.Queue, event_stop: multiprocessing.Event):
    cap = cv2.VideoCapture(path_video)
    id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame!")
            break
        frame_queue.put((id, frame))
        id += 1
        time.sleep(0.0001)
    event_stop.set()
        

def fun_thread_safe_predict(frame_queue: multiprocessing.Queue, event_stop: multiprocessing.Event, output_queue: multiprocessing.Queue):
    local_model = YOLO(model="yolov8n-pose.pt")
    while True:
        try:
            id, frame = frame_queue.get(timeout=1)
            output_queue.put((id, local_model.predict(source=frame, device='cpu')[0].plot()))
        except:
            if event_stop.is_set():
                print(f'Thread {multiprocessing.current_process()} final!')
                break


def main(input_video_path, num_of_threads, output_video_path):
    YOLO(model="yolov8n-pose.pt")
    threads = []
    frame_queue = multiprocessing.Queue(1000)
    output_queue = multiprocessing.Queue()
    event_stop = multiprocessing.Event()
    thread_read = multiprocessing.Process(target=fun_thread_read, args=(input_video_path, frame_queue, event_stop, ))
    thread_read.start()
    start_t = time.monotonic()
    for _ in range(num_of_threads):
        threads.append(multiprocessing.Process(target=fun_thread_safe_predict, args=(frame_queue, event_stop, output_queue, )))

    for thr in threads:
        thr.start()

    for thr in threads:
        thr.join()

    thread_read.join()

    print('write to output')
    frames_arr = []
    while(not output_queue.empty()):
        frames_arr.append(output_queue.get())
    frames_arr = sorted(frames_arr, key=lambda x: x[0])
    _, frames_for_video_yolo = zip(*frames_arr)
    del(frames_arr)

    frames_for_video_numpy = []
    for img in frames_for_video_yolo:
        frames_for_video_numpy.append(img)
    del(frames_for_video_yolo)
    frame_width = frames_for_video_numpy[0].shape[1]
    frame_height = frames_for_video_numpy[0].shape[0]
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))
    for frame in frames_for_video_numpy:
        out.write(frame)

    print(f'video is writen in {output_video_path}')
    end_t = time.monotonic()
    print(f'Time: {end_t - start_t}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_video_path', type=str, help='path to input video')
    parser.add_argument('processes', type=int, help='number of processes')
    parser.add_argument('output_video_path', type=str, help='path to output video')
    args = parser.parse_args()
    main(args.input_video_path, args.processes, args.output_video_path)