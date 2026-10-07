"""
Object Detection Module
Detects and tracks objects in video frames using YOLO
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict
import logging

import numpy as np
import cv2

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """A detected object"""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    track_id: Optional[int] = None
    
    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)


@dataclass
class FrameDetections:
    """All detections in a frame"""
    frame_number: int
    timestamp: float
    detections: List[Detection]
    
    def get_objects_by_class(self) -> Dict[str, List[Detection]]:
        """Group detections by class name"""
        by_class = defaultdict(list)
        for det in self.detections:
            by_class[det.class_name].append(det)
        return dict(by_class)
    
    def get_object_names(self) -> List[str]:
        """Get unique object names in frame"""
        return list(set(d.class_name for d in self.detections))


@dataclass 
class ObjectTrack:
    """Tracked object across frames"""
    track_id: int
    class_name: str
    detections: List[Tuple[int, Detection]]  # (frame_number, detection)
    
    @property
    def start_frame(self) -> int:
        return self.detections[0][0]
    
    @property
    def end_frame(self) -> int:
        return self.detections[-1][0]
    
    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame


class ObjectDetector:
    """
    Object detection and tracking using YOLOv8
    """
    
    # COCO class names
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
        'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.device = config.device
        self._load_model()
    
    def _load_model(self) -> None:
        """Load YOLO model"""
        try:
            from ultralytics import YOLO
            
            model_name = self.config.object_detection.model
            logger.info(f"Loading YOLO model: {model_name}")
            
            self.model = YOLO(model_name)
            
            # Move to device
            if self.device == "cuda":
                self.model.to("cuda")
            
            logger.info("YOLO model loaded successfully")
            
        except ImportError:
            logger.error("ultralytics not installed. Run: pip install ultralytics")
            raise
    
    def detect(
        self,
        image: Union[str, np.ndarray],
        confidence: Optional[float] = None,
        classes: Optional[List[int]] = None
    ) -> List[Detection]:
        """
        Detect objects in a single image
        
        Args:
            image: Image path or numpy array (BGR)
            confidence: Minimum confidence threshold
            classes: List of class IDs to detect (None for all)
        
        Returns:
            List of Detection objects
        """
        confidence = confidence or self.config.object_detection.confidence
        classes = classes or self.config.object_detection.classes
        
        # Run detection
        results = self.model(
            image,
            conf=confidence,
            classes=classes,
            verbose=False
        )[0]
        
        detections = []
        
        for box in results.boxes:
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]
            conf = float(box.conf[0])
            
            # Get bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            detection = Detection(
                class_id=class_id,
                class_name=class_name,
                confidence=conf,
                bbox=(x1, y1, x2, y2)
            )
            detections.append(detection)
        
        return detections
    
    def detect_and_track(
        self,
        video_path: str,
        confidence: Optional[float] = None,
        classes: Optional[List[int]] = None,
        tracker: str = "bytetrack"
    ) -> Tuple[List[FrameDetections], List[ObjectTrack]]:
        """
        Detect and track objects in video
        
        Args:
            video_path: Path to video file
            confidence: Minimum confidence
            classes: Class IDs to detect
            tracker: Tracker type ('bytetrack', 'botsort')
        
        Returns:
            Tuple of (frame_detections, tracks)
        """
        confidence = confidence or self.config.object_detection.confidence
        classes = classes or self.config.object_detection.classes
        
        logger.info(f"Running detection and tracking on {video_path}")
        
        # Run tracking
        results = self.model.track(
            video_path,
            conf=confidence,
            classes=classes,
            tracker=f"{tracker}.yaml",
            persist=True,
            verbose=False,
            stream=True
        )
        
        frame_detections: List[FrameDetections] = []
        tracks_dict: Dict[int, List[Tuple[int, Detection]]] = defaultdict(list)
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        for frame_num, result in enumerate(results):
            timestamp = frame_num / fps if fps > 0 else 0
            detections = []
            
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Get track ID if available
                    track_id = None
                    if box.id is not None:
                        track_id = int(box.id[0])
                    
                    detection = Detection(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=conf,
                        bbox=(x1, y1, x2, y2),
                        track_id=track_id
                    )
                    detections.append(detection)
                    
                    # Add to tracks
                    if track_id is not None:
                        tracks_dict[track_id].append((frame_num, detection))
            
            frame_det = FrameDetections(
                frame_number=frame_num,
                timestamp=timestamp,
                detections=detections
            )
            frame_detections.append(frame_det)
        
        # Convert tracks
        tracks = []
        for track_id, track_dets in tracks_dict.items():
            if track_dets:
                track = ObjectTrack(
                    track_id=track_id,
                    class_name=track_dets[0][1].class_name,
                    detections=track_dets
                )
                tracks.append(track)
        
        logger.info(f"Processed {len(frame_detections)} frames, found {len(tracks)} tracks")
        
        return frame_detections, tracks
    
    def process_frames(
        self,
        frames: List[Tuple[int, float, np.ndarray]],
        confidence: Optional[float] = None
    ) -> List[FrameDetections]:
        """
        Process list of frames
        
        Args:
            frames: List of (frame_number, timestamp, image)
            confidence: Minimum confidence
        
        Returns:
            List of FrameDetections
        """
        from tqdm import tqdm
        
        results = []
        
        for frame_num, timestamp, image in tqdm(frames, desc="Detecting objects"):
            detections = self.detect(image, confidence)
            
            frame_det = FrameDetections(
                frame_number=frame_num,
                timestamp=timestamp,
                detections=detections
            )
            results.append(frame_det)
        
        return results
    
    def summarize_detections(
        self,
        frame_detections: List[FrameDetections]
    ) -> Dict[str, Any]:
        """
        Summarize all detections
        
        Returns:
            Summary dict with counts and statistics
        """
        object_counts = defaultdict(int)
        object_frames = defaultdict(list)
        total_detections = 0
        
        for fd in frame_detections:
            for det in fd.detections:
                object_counts[det.class_name] += 1
                object_frames[det.class_name].append(fd.frame_number)
                total_detections += 1
        
        # Calculate presence percentages
        total_frames = len(frame_detections)
        presence = {}
        
        for obj, frames in object_frames.items():
            unique_frames = len(set(frames))
            presence[obj] = (unique_frames / total_frames) * 100 if total_frames > 0 else 0
        
        # Sort by count
        sorted_objects = sorted(object_counts.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "total_frames": total_frames,
            "total_detections": total_detections,
            "unique_objects": len(object_counts),
            "object_counts": dict(sorted_objects),
            "presence_percentage": presence,
            "top_objects": [obj for obj, _ in sorted_objects[:10]]
        }
    
    def get_objects_in_timerange(
        self,
        frame_detections: List[FrameDetections],
        start_time: float,
        end_time: float
    ) -> List[str]:
        """Get unique objects detected in a time range"""
        objects = set()
        
        for fd in frame_detections:
            if start_time <= fd.timestamp <= end_time:
                for det in fd.detections:
                    objects.add(det.class_name)
        
        return list(objects)
    
    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[Detection],
        draw_labels: bool = True,
        draw_confidence: bool = True
    ) -> np.ndarray:
        """Draw detections on image"""
        result = image.copy()
        
        # Color palette for different classes
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(len(self.COCO_CLASSES), 3), dtype=np.uint8)
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = tuple(map(int, colors[det.class_id % len(colors)]))
            
            # Draw box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            
            if draw_labels:
                label = det.class_name
                if draw_confidence:
                    label += f" {det.confidence:.2f}"
                if det.track_id is not None:
                    label += f" ID:{det.track_id}"
                
                # Draw label background
                (label_w, label_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    result,
                    (x1, y1 - label_h - 5),
                    (x1 + label_w, y1),
                    color, -1
                )
                
                # Draw label text
                cv2.putText(
                    result, label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1
                )
        
        return result
    
    def filter_by_region(
        self,
        detections: List[Detection],
        region: Tuple[int, int, int, int]  # x1, y1, x2, y2
    ) -> List[Detection]:
        """Filter detections by region of interest"""
        rx1, ry1, rx2, ry2 = region
        filtered = []
        
        for det in detections:
            dx1, dy1, dx2, dy2 = det.bbox
            center_x, center_y = det.center
            
            # Check if center is in region
            if rx1 <= center_x <= rx2 and ry1 <= center_y <= ry2:
                filtered.append(det)
        
        return filtered
    
    def get_important_objects(
        self,
        frame_detections: List[FrameDetections],
        min_presence: float = 10.0
    ) -> List[str]:
        """
        Get objects that appear in at least min_presence% of frames
        """
        summary = self.summarize_detections(frame_detections)
        
        important = [
            obj for obj, pct in summary["presence_percentage"].items()
            if pct >= min_presence
        ]
        
        return important
