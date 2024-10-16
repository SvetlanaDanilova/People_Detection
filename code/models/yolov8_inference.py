from ultralytics import YOLO

class YOLOv8Detector:
    def __init__(self, threshold):
        """
        Initializes the YOLOv8 detector with the given confidence threshold.
        
        Args:
            threshold (float): Confidence threshold for detecting objects.
        """
        # Load the pre-trained YOLOv8 model from the specified path
        self.model = YOLO('code/models/yolov8n.pt')
        
        # Set the confidence threshold for filtering detections
        self.threshold = threshold

    def detect(self, frame):
        """
        Perform object detection on a single video frame.
        
        Args:
            frame (numpy array): The input image frame for object detection.
        
        Returns:
            list: A list of detected objects, each with bounding box coordinates and confidence.
        """
        # Perform inference with the YOLOv8 model on the provided frame
        # The `conf=self.threshold` argument filters out detections below the threshold
        results = self.model(frame, conf=self.threshold, verbose=False)
        
        detections = []
        
        # Iterate over the detection results
        for result in results[0].boxes:
            # Check if the detected class is 'person' (class 0 in YOLOv8)
            if result.cls[0] == 0:
                # Extract bounding box coordinates and confidence score
                x1, y1, x2, y2 = map(int, result.xyxy[0])  # Convert coordinates to integers
                conf = float(result.conf[0])  # Extract the confidence score as a float
                
                # Append detection information as a dictionary to the detections list
                detections.append({'bbox': (x1, y1, x2, y2), 'confidence': conf})
        
        # Return the list of detections for the frame
        return detections
