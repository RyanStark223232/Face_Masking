import cv2
import mediapipe as mp
import argparse
import glob
import os
from tqdm import tqdm


def mask_faces_in_video(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    mp_face_detection = mp.solutions.face_detection.FaceDetection()
    prev_x, prev_y, prev_w, prev_h = 0, 0, 0, 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = tqdm(total=total_frames, desc="Processing Frames", unit="frame")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = mp_face_detection.process(frame_rgb)

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * frame_width)
                y = int(bbox.ymin * frame_height)
                w = int(bbox.width * frame_width)
                h = int(bbox.height * frame_height)

                prev_x, prev_y, prev_w, prev_h = x, y, w, h

                frame[y:y+h, x:x+w] = (0, 0, 0)
        else:
            frame[prev_y:prev_y+prev_h, prev_x:prev_x+prev_w] = (0, 0, 0)

        out.write(frame)
        progress_bar.update(1)

    cap.release()
    out.release()
    progress_bar.close()


def main(input_dir, output_dir):
    input_files = glob.glob(os.path.join(input_dir, '*.mp4'))

    for input_file in input_files:
        filename = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(output_dir, f'{filename}_masked.mp4')
        print(f"Processing: {filename}.mp4")
        mask_faces_in_video(input_file, output_file)
        print(f"Completed: {filename}_masked.mp4")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Face Masking in MP4 Videos')
    parser.add_argument('--input_dir', type=str, default="Video",
                        help='Path to the input directory containing MP4 videos')
    parser.add_argument('--output_dir', type=str, default="Video",
                        help='Path to the output directory to save modified videos')
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)
