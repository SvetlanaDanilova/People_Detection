# People Detection in Video using YOLOv8 and Detectron2

This project implements object detection for recognizing people in a video file using two different algorithms: YOLOv8 and Detectron2. The output is a video with highlighted people and the confidence scores of the detections. The project is packaged using Docker for cross-platform compatibility.

## Features

- Supports two detection algorithms: YOLOv8 and Detectron2
- Outputs a video with detected people and confidence scores
- Cross-platform support via Docker (Linux, MacOS, Windows)
- Easy model selection via command line arguments

## Requirements

- Docker
- Python 3.9+ (optional for running locally)

## Project Structure

```
.
├── Dockerfile                            # Docker configuration
├── README.md                             # Project documentation
├── requirements.txt                      # Python dependencies
├── code/                                 # Directory containing all code
|    ├── detect_people.py                 # Main script for detecting people in video
|    └── models/                          # Directory containing model-specific code
|        ├── yolov8_inference.py          # YOLOv8 model implementation
|        └── detectron2_inference.py      # Detectron2 model implementation
└── data/                                 # Directory with videos
    ├── input                             # Folder with input videos
    └── output                            # Folder for output videos
```

## Setup

### Clone this repository

### Place the input video in mp4 format into data/input/ folder

### Build the Docker image

```
docker build -t <your-docker-image-name> .
```

### Run the Docker container

You can run the program by passing the path to input video file, the path to output video file, the name of model to use (yolov8 or detectron2) and threshold of confidence for model (from 0 to 1)

```
docker run -it --rm -v "$(pwd)/data/output:/app/data/output" --name <your-docker-container-name> <your-docker-image-name> --input /data/input/<input-video-name>.mp4 --output /data/output/<output-video-name>.mp4 --model <yolov8-or-detectron2> --threshold <your-size-of-threshold>
```

### Output

After running the container, the resulting video with detected people and confidence scores will be saved in the output file you specified 

## Choosing the Model

- YOLOv8: Fast and lightweight, suitable for real-time applications.
- Detectron2: More accurate but slower, suitable for applications where precision is more important than speed.

