import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

class Detectron2Detector:
    def __init__(self, threshold):
        """
        Initializes the Detectron2 detector with the given confidence threshold.
        
        Args:
            threshold (float): Confidence threshold for detecting objects.
        """
        # Load and configure the Detectron2 model with Faster R-CNN architecture
        cfg = get_cfg()
        
        # Load the pre-trained Faster R-CNN configuration from the model zoo
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        
        # Load pre-trained weights for the model from the model zoo
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        
        # Set the threshold for object detection
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
        
        # Set the device for computation (CUDA if available, otherwise CPU)
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize the predictor with the loaded configuration
        self.predictor = DefaultPredictor(cfg)

    def detect(self, frame):
        """
        Perform object detection on a single video frame using Detectron2.
        
        Args:
            frame (numpy array): The input image frame for object detection.
        
        Returns:
            list: A list of detected objects, each with bounding box coordinates and confidence score.
        """
        # Perform inference on the frame
        outputs = self.predictor(frame)
        
        # Move the detection results to CPU for further processing
        instances = outputs["instances"].to("cpu")
        
        # Initialize a list to store detections
        detections = []
        
        # Iterate through detected instances
        for i in range(len(instances)):
            # Extract bounding box coordinates and convert them to integers
            bbox = instances.pred_boxes[i].tensor.numpy()[0].astype(int)
            
            # Extract the confidence score for the current instance
            conf = instances.scores[i].item()
            
            # Check if the detected class is 'person' (class 0)
            if instances.pred_classes[i].item() == 0:
                # Append detection information (bounding box and confidence) to the detections list
                detections.append({'bbox': bbox, 'confidence': conf})
        
        # Return the list of detections for the frame
        return detections
