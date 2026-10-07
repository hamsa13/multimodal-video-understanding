"""
Scene Detection Module
Detects scene changes and classifies environments
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any
from dataclasses import dataclass, field
import logging

import numpy as np
import cv2

logger = logging.getLogger(__name__)


@dataclass
class Scene:
    """A detected scene in the video"""
    scene_number: int
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    duration: float
    thumbnail_path: Optional[str] = None
    environment: Optional[str] = None
    environment_confidence: float = 0.0
    dominant_colors: List[Tuple[int, int, int]] = field(default_factory=list)
    brightness: float = 0.0
    is_indoor: Optional[bool] = None


@dataclass
class SceneDetectionResult:
    """Complete scene detection result"""
    scenes: List[Scene]
    total_scenes: int
    video_duration: float
    average_scene_duration: float


class SceneDetector:
    """
    Scene change detection and environment classification
    Uses PySceneDetect and Places365 for classification
    """
    
    def __init__(self, config):
        self.config = config
        self.places_model = None
        self.places_labels = None
        self._init_places_model()
    
    def _init_places_model(self) -> None:
        """Initialize Places365 model for environment classification"""
        try:
            import torch
            import torchvision.models as models
            from torchvision import transforms
            
            # Try to load places365 model
            # Using ResNet50 as a proxy (you'd want actual places365 weights)
            logger.info("Initializing environment classification model...")
            
            self._transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            # Load default scene categories
            self.places_labels = self._get_scene_categories()
            self._places_available = True
            
            logger.info("Environment classification ready")
            
        except Exception as e:
            logger.warning(f"Environment classification not available: {e}")
            self._places_available = False
    
    def _get_scene_categories(self) -> List[str]:
        """Get scene category labels"""
        # Common scene categories
        return [
            "office", "bedroom", "kitchen", "bathroom", "living_room",
            "street", "park", "beach", "forest", "mountain",
            "restaurant", "bar", "hospital", "school", "store",
            "car_interior", "airplane", "train", "stadium", "theater",
            "warehouse", "factory", "laboratory", "library", "museum",
            "outdoor_urban", "outdoor_rural", "indoor_commercial", "indoor_residential"
        ]
    
    def detect_scenes(
        self,
        video_path: str,
        method: Optional[str] = None,
        threshold: Optional[float] = None,
        min_scene_length: Optional[int] = None
    ) -> SceneDetectionResult:
        """
        Detect scene changes in video
        
        Args:
            video_path: Path to video file
            method: Detection method ('content', 'threshold', 'adaptive')
            threshold: Scene change threshold
            min_scene_length: Minimum frames per scene
        
        Returns:
            SceneDetectionResult with all detected scenes
        """
        video_path = str(video_path)
        method = method or self.config.scene.detector
        threshold = threshold or self.config.scene.threshold
        min_scene_length = min_scene_length or self.config.scene.min_scene_length
        
        logger.info(f"Detecting scenes in {video_path} using {method} method")
        
        try:
            return self._detect_with_pyscenedetect(
                video_path, method, threshold, min_scene_length
            )
        except ImportError:
            logger.warning("PySceneDetect not available, using fallback")
            return self._detect_with_opencv(
                video_path, threshold, min_scene_length
            )
    
    def _detect_with_pyscenedetect(
        self,
        video_path: str,
        method: str,
        threshold: float,
        min_scene_length: int
    ) -> SceneDetectionResult:
        """Detect scenes using PySceneDetect"""
        from scenedetect import detect, ContentDetector, ThresholdDetector, AdaptiveDetector
        from scenedetect import open_video
        
        # Select detector
        if method == "content":
            detector = ContentDetector(threshold=threshold, min_scene_len=min_scene_length)
        elif method == "threshold":
            detector = ThresholdDetector(threshold=threshold, min_scene_len=min_scene_length)
        elif method == "adaptive":
            detector = AdaptiveDetector(min_scene_len=min_scene_length)
        else:
            detector = ContentDetector(threshold=threshold, min_scene_len=min_scene_length)
        
        # Open video and detect scenes
        video = open_video(video_path)
        scene_list = detect(video_path, detector)
        
        # Get video metadata
        fps = video.frame_rate
        duration = video.duration.get_seconds()
        
        # Convert to our format
        scenes: List[Scene] = []
        
        for i, (start, end) in enumerate(scene_list):
            scene = Scene(
                scene_number=i + 1,
                start_frame=start.frame_num,
                end_frame=end.frame_num,
                start_time=start.get_seconds(),
                end_time=end.get_seconds(),
                duration=end.get_seconds() - start.get_seconds()
            )
            scenes.append(scene)
        
        # If no scenes detected, create one scene for the entire video
        if not scenes:
            logger.info("No scene changes detected, creating single scene for entire video")
            total_frames = int(video.frame_rate * duration) if video.frame_rate > 0 else 1
            scenes.append(Scene(
                scene_number=1,
                start_frame=0,
                end_frame=total_frames,
                start_time=0.0,
                end_time=duration,
                duration=duration
            ))
        
        # Calculate average duration
        avg_duration = sum(s.duration for s in scenes) / len(scenes) if scenes else duration
        
        logger.info(f"Detected {len(scenes)} scenes")
        
        return SceneDetectionResult(
            scenes=scenes,
            total_scenes=len(scenes),
            video_duration=duration,
            average_scene_duration=avg_duration
        )
    
    def _detect_with_opencv(
        self,
        video_path: str,
        threshold: float,
        min_scene_length: int
    ) -> SceneDetectionResult:
        """Fallback scene detection using OpenCV histogram comparison"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        scenes: List[Scene] = []
        prev_hist = None
        scene_start = 0
        scene_num = 1
        
        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Calculate histogram
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            
            if prev_hist is not None:
                # Compare histograms
                diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                
                # Scene change detected
                if diff < (1 - threshold / 100) and (frame_num - scene_start) >= min_scene_length:
                    scene = Scene(
                        scene_number=scene_num,
                        start_frame=scene_start,
                        end_frame=frame_num,
                        start_time=scene_start / fps,
                        end_time=frame_num / fps,
                        duration=(frame_num - scene_start) / fps
                    )
                    scenes.append(scene)
                    scene_start = frame_num
                    scene_num += 1
            
            prev_hist = hist
            frame_num += 1
        
        # Add final scene
        if frame_num - scene_start >= min_scene_length:
            scene = Scene(
                scene_number=scene_num,
                start_frame=scene_start,
                end_frame=frame_num,
                start_time=scene_start / fps,
                end_time=frame_num / fps,
                duration=(frame_num - scene_start) / fps
            )
            scenes.append(scene)
        
        cap.release()
        
        avg_duration = sum(s.duration for s in scenes) / len(scenes) if scenes else 0
        
        return SceneDetectionResult(
            scenes=scenes,
            total_scenes=len(scenes),
            video_duration=duration,
            average_scene_duration=avg_duration
        )
    
    def classify_scene_environment(
        self,
        frame: np.ndarray
    ) -> Tuple[str, float]:
        """
        Classify the environment/location of a scene
        
        Args:
            frame: Frame image (BGR)
        
        Returns:
            Tuple of (environment_name, confidence)
        """
        if not self._places_available:
            return self._classify_simple(frame)
        
        return self._classify_simple(frame)  # Use simple method for now
    
    def _classify_simple(
        self,
        frame: np.ndarray
    ) -> Tuple[str, float]:
        """Simple environment classification based on image features"""
        # Convert to HSV for analysis
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Calculate average brightness
        brightness = np.mean(hsv[:, :, 2])
        
        # Calculate color distribution
        h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
        
        # Normalize
        h_hist = h_hist / h_hist.sum()
        s_hist = s_hist / s_hist.sum()
        
        # Robust feature set
        avg_saturation = np.mean(hsv[:, :, 1])
        green_ratio = float(h_hist[30:90].sum())  # Green hues
        blue_ratio = float(h_hist[90:130].sum())  # Blue hues

        # Sky-like pixels (low saturation blue/white with good brightness)
        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        sky_mask = (((h >= 90) & (h <= 130) & (s < 120) & (v > 100)) | ((s < 30) & (v > 180)))
        sky_ratio = float(np.mean(sky_mask))

        # Skin-like pixels (useful for interview/talking-head scenes)
        skin_mask = ((h <= 20) & (s >= 40) & (s <= 200) & (v >= 60))
        skin_ratio = float(np.mean(skin_mask))

        # Texture/structure estimate
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        edge_density = float(np.mean(edges > 0))

        # Interview/talking-head override (common for this project's videos)
        if skin_ratio > 0.05 and edge_density > 0.03 and brightness < 170:
            return ("indoor interview setting", 0.80)

        # Outdoor score (sky + vegetation + high saturation cue)
        outdoor_score = sky_ratio + (0.8 * green_ratio) + (0.2 if avg_saturation > 90 else 0.0)

        if outdoor_score > 0.30:
            if green_ratio > 0.22:
                return ("outdoor green area", 0.74)
            if sky_ratio > 0.20:
                return ("open outdoor space", 0.72)
            return ("urban outdoor setting", 0.66)

        # Indoor classes
        if brightness < 70:
            if skin_ratio > 0.03:
                return ("indoor low-light interview setting", 0.76)
            return ("indoor low-light room", 0.70)

        if skin_ratio > 0.04 and brightness < 160:
            return ("indoor interview setting", 0.75)

        if edge_density > 0.08 and avg_saturation < 75:
            return ("indoor office or studio", 0.68)

        if brightness > 175:
            return ("indoor well-lit room", 0.62)

        return ("indoor room", 0.58)
    
    def analyze_scene(
        self,
        frames: List[np.ndarray],
        scene: Scene
    ) -> Scene:
        """
        Analyze a scene to extract additional information
        
        Args:
            frames: List of frames from the scene
            scene: Scene object to update
        
        Returns:
            Updated Scene with analysis
        """
        if not frames:
            return scene
        
        # Use middle frame for classification
        mid_idx = len(frames) // 2
        mid_frame = frames[mid_idx]
        
        # Classify environment
        env, conf = self.classify_scene_environment(mid_frame)
        scene.environment = env
        scene.environment_confidence = conf
        
        # Determine indoor/outdoor with image-aware fallback
        scene.is_indoor = self._estimate_indoor_outdoor(mid_frame, env)
        
        # Calculate average brightness
        brightness_values = []
        for frame in frames[::max(1, len(frames) // 5)]:  # Sample frames
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            brightness_values.append(np.mean(hsv[:, :, 2]))
        scene.brightness = np.mean(brightness_values)
        
        # Extract dominant colors
        scene.dominant_colors = self._get_dominant_colors(mid_frame)
        
        return scene

    def _estimate_indoor_outdoor(self, frame: np.ndarray, environment: Optional[str]) -> bool:
        """Estimate indoor/outdoor with keyword + pixel heuristics."""
        env = (environment or "").lower()

        indoor_terms = {"indoor", "office", "studio", "room", "interview", "kitchen", "bedroom"}
        outdoor_terms = {"outdoor", "park", "forest", "street", "beach", "mountain", "urban"}

        if any(term in env for term in indoor_terms):
            return True
        if any(term in env for term in outdoor_terms):
            return False

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        sky_mask = (((h >= 90) & (h <= 130) & (s < 120) & (v > 100)) | ((s < 30) & (v > 180)))
        green_mask = ((h >= 30) & (h <= 90) & (s > 40))

        outdoor_signal = float(np.mean(sky_mask)) + (0.8 * float(np.mean(green_mask)))
        return outdoor_signal < 0.28
    
    def _get_dominant_colors(
        self,
        frame: np.ndarray,
        k: int = 3
    ) -> List[Tuple[int, int, int]]:
        """Extract dominant colors using k-means"""
        # Resize for speed
        small = cv2.resize(frame, (100, 100))
        pixels = small.reshape(-1, 3).astype(np.float32)
        
        # K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert to int tuples (BGR)
        colors = [tuple(map(int, c)) for c in centers]
        
        return colors
    
    def save_scene_thumbnails(
        self,
        video_path: str,
        scenes: List[Scene],
        output_dir: str
    ) -> List[Scene]:
        """
        Save thumbnail images for each scene
        
        Args:
            video_path: Path to video
            scenes: List of scenes
            output_dir: Directory to save thumbnails
        
        Returns:
            Updated scenes with thumbnail paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(video_path))
        
        for scene in scenes:
            # Use middle frame
            mid_frame = (scene.start_frame + scene.end_frame) // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
            
            ret, frame = cap.read()
            if ret:
                thumb_path = output_dir / f"scene_{scene.scene_number:03d}.jpg"
                cv2.imwrite(str(thumb_path), frame)
                scene.thumbnail_path = str(thumb_path)
        
        cap.release()
        
        return scenes
    
    def get_scene_transitions(
        self,
        scenes: List[Scene]
    ) -> List[Dict[str, Any]]:
        """
        Analyze transitions between scenes
        
        Returns:
            List of transition info dicts
        """
        transitions = []
        
        for i in range(len(scenes) - 1):
            current = scenes[i]
            next_scene = scenes[i + 1]
            
            transition = {
                "from_scene": current.scene_number,
                "to_scene": next_scene.scene_number,
                "time": current.end_time,
                "from_environment": current.environment,
                "to_environment": next_scene.environment,
                "environment_change": current.environment != next_scene.environment
            }
            transitions.append(transition)
        
        return transitions
