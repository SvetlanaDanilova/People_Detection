import argparse
import cv2
from models.yolov8_inference import YOLOv8Detector
from models.detectron2_inference import Detectron2Detector
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")  # Suppress unnecessary warnings

def get_default_video_path():
    # Path to the folder with input videos
    input_dir = "data/input/"
    # Get a list of files with the .mp4 extension in this folder
    mp4_files = [f for f in os.listdir(input_dir) if f.endswith(".mp4")]
    # If there is at least one file, take the first one; otherwise, return an empty string
    if mp4_files:
        return os.path.join(input_dir, mp4_files[0])
    else:
        print(f"No .mp4 files found in {input_dir}")
        return ""

def get_detector(model_name, threshold):
    """
    Returns the appropriate object detection model based on the input model name.
    
    Args:
        model_name (str): The name of the model ('yolov8' or 'detectron2').
        threshold (float): The confidence threshold for detections.
    
    Returns:
        Object: An instance of the corresponding detection model.
    """
    if model_name == "yolov8":
        return YOLOv8Detector(threshold)
    elif model_name == "detectron2":
        return Detectron2Detector(threshold)
    else:
        raise ValueError("Unsupported model. Choose either 'yolov8' or 'detectron2'.")

def process_video(input_video_path, output_video_path, model_name, threshold):
    """
    Processes the input video and applies object detection frame-by-frame.
    
    Args:
        input_video_path (str): Path to the input video file.
        output_video_path (str): Path to save the output video with detections.
        model_name (str): Model to use for detection ('yolov8' or 'detectron2').
        threshold (float): Confidence threshold for object detection.
    """
    # Initialize the detector
    detector = get_detector(model_name, threshold)
    
    # Open the input video file
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video_path}")
        return

    # Get video properties
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec for the output video
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize the video writer for the output video
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Get total number of frames for progress bar
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize progress bar
    pbar = tqdm(total=total_frames, desc='Processing video')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Perform object detection on the current frame
        detections = detector.detect(frame)

        # Draw detections on the frame
        for det in detections:
            x1, y1, x2, y2 = det['bbox']  # Bounding box coordinates
            conf = det['confidence']      # Confidence score of the detection
            label = f"Person: {conf:.2f}"  # Format the label with confidence score
            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Write the processed frame to the output video
        out.write(frame)

        # Update the progress bar
        pbar.update(1)

    # Release video resources
    cap.release()
    out.release()
    pbar.close()

    print(f"Output video saved to: {output_video_path}")

if __name__ == "__main__":
    # Argument parsing for the command-line interface
    parser = argparse.ArgumentParser(description="Object Detection on video.")
    
    # Input video argument
    default_video_path = get_default_video_path()
    parser.add_argument("--input", type=str, default=default_video_path, help="Path to the input video file")
    
    # Output video argument
    parser.add_argument("--output", type=str, default="data/output/output_video.mp4", help="Path to save the output video")
    
    # Model selection argument (choices are yolov8 or detectron2)
    parser.add_argument("--model", type=str, default="detectron2", choices=["yolov8", "detectron2"], help="Model to use: yolov8 or detectron2")
    
    # Confidence threshold argument for object detection
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold for object detection")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Call the video processing function with the provided arguments
    process_video(args.input, args.output, args.model, args.threshold)
