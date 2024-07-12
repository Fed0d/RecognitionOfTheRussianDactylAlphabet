import cv2
import argparse
from mmaction.apis import init_recognizer, inference_recognizer
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tempfile
import os


def load_label_map(label_map_path):
    with open(label_map_path, 'r', encoding='utf-8') as f:
        label_map = f.readlines()
    label_map = [label.strip() for label in label_map]
    return label_map


def process_video(input_video_path, output_video_path, config_file, checkpoint_file, prob_threshold, output_text_path):
    model = init_recognizer(config_file, checkpoint_file, device='cuda:0')

    cap = cv2.VideoCapture(input_video_path)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    label_map = load_label_map('label_map.txt')

    frame_sequence = []
    temp_files = []

    with open(output_text_path, 'w') as f:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_sequence.append(frame)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (255, 255, 255), -1)

            if len(frame_sequence) == 25:
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                temp_files.append(temp_file.name)

                temp_out = cv2.VideoWriter(temp_file.name, fourcc, fps, (frame_width, frame_height))

                for frame in frame_sequence:
                    temp_out.write(frame)

                temp_out.release()

                result = inference_recognizer(model, temp_file.name)

                if result.pred_score.cpu()[result.pred_label.item()] >= prob_threshold:
                    gesture = label_map[result.pred_label.item()]
                    f.write(f'{gesture}\n')
                else:
                    gesture = '?'

                for frame in frame_sequence:
                    pil_img = Image.fromarray(frame)
                    draw = ImageDraw.Draw(pil_img)
                    font = ImageFont.truetype("arial.ttf", 30)
                    draw.text((frame.shape[1] // 2, 10), gesture, font=font, fill=(0, 0, 0, 0))
                    frame = np.array(pil_img)
                    out.write(frame)

                frame_sequence = []

    cap.release()
    out.release()

    for temp_file in temp_files:
        os.remove(temp_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a video.')
    parser.add_argument('input_video_path', type=str, help='Input video path.')
    parser.add_argument('output_video_path', type=str, help='Output video path.')
    parser.add_argument('output_text_path', type=str, help='Output text file path.')
    parser.add_argument('config_file', type=str, help='Model config file path.')
    parser.add_argument('checkpoint_file', type=str, help='Model checkpoint file path.')
    parser.add_argument('--prob_threshold', type=float, default=0.3,
                        help='Probability threshold for writing the letter.')
    args = parser.parse_args()

    process_video(args.input_video_path, args.output_video_path, args.config_file, args.checkpoint_file,
                  args.prob_threshold, args.output_text_path)
