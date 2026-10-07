"""
Action Recognition Module
Detects and classifies human actions in video clips
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any, Union
from dataclasses import dataclass, field
import logging

import numpy as np
import cv2

logger = logging.getLogger(__name__)


# Common action labels (Kinetics-400 subset
ACTION_LABELS = [
    "applauding", "arm wrestling", "arguing", "attending conference", "baby kissing",
    "blowing kisses", "boxing", "brushing teeth", "carrying baby", "celebrating",
    "cleaning floor", "climbing", "cooking", "crying", "dancing",
    "drinking", "driving", "eating", "exercising", "falling",
    "fighting", "giving speech", "handshaking", "headbanging", "hugging",
    "ironing", "jogging", "jumping", "kicking", "kissing",
    "laughing", "listening to music", "looking at phone", "making bed", "marching",
    "meditating", "opening door", "painting", "playing basketball", "playing cards",
    "playing guitar", "playing piano", "playing video games", "pointing", "praying",
    "presenting", "punching", "pushing", "reading", "recording",
    "riding bike", "running", "saluting", "shaking hands", "shooting",
    "shouting", "singing", "sitting", "sleeping", "smoking",
    "sneezing", "standing", "stretching", "swimming", "talking on phone",
    "texting", "thinking", "throwing", "typing", "waking up",
    "walking", "waving", "working on computer", "writing", "yawning"
]


@dataclass
class ActionPrediction:
    """A predicted action"""
    action: str
    confidence: float
    start_time: float
    end_time: float
    clip_frames: Optional[List[int]] = None


@dataclass
class ClipActions:
    """Actions detected in a video clip"""
    clip_number: int
    start_time: float
    end_time: float
    predictions: List[ActionPrediction]
    
    @property
    def top_action(self) -> Optional[ActionPrediction]:
        if self.predictions:
            return max(self.predictions, key=lambda x: x.confidence)
        return None


class ActionRecognizer:
    """
    Human action recognition using video models
    Supports SlowFast, I3D, and Video Swin Transformer
    """
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.device = config.device
        self.labels = ACTION_LABELS
        self._init_model()
    
    def _init_model(self) -> None:
        """Initialize action recognition model"""
        model_name = self.config.action.model.lower()
        
        if model_name == "slowfast":
            self._init_slowfast()
        elif model_name == "i3d":
            self._init_i3d()
        elif model_name == "video_swin":
            self._init_video_swin()
        else:
            logger.warning(f"Unknown model {model_name}, using simplified model")
            self._init_simplified()
    
    def _init_slowfast(self) -> None:
        """Initialize SlowFast model from PyTorchHub"""
        try:
            import torch
            
            logger.info("Loading SlowFast model from PyTorch Hub...")
            self.model = torch.hub.load(
                'facebookresearch/pytorchvideo',
                'slowfast_r50',
                pretrained=True
            )
            self.model.eval()
            
            if self.device == "cuda":
                self.model = self.model.cuda()
            
            self._model_type = "slowfast"
            self._init_transforms_slowfast()
            
            logger.info("SlowFast model loaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load SlowFast: {e}, using simplified model")
            self._init_simplified()
    
    def _init_i3d(self) -> None:
        """Initialize I3D model"""
        try:
            import torch
            
            logger.info("Loading I3D model from PyTorch Hub...")
            self.model = torch.hub.load(
                'facebookresearch/pytorchvideo',
                'i3d_r50',
                pretrained=True
            )
            self.model.eval()
            
            if self.device == "cuda":
                self.model = self.model.cuda()
            
            self._model_type = "i3d"
            self._init_transforms_i3d()
            
            logger.info("I3D model loaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load I3D: {e}, using simplified model")
            self._init_simplified()
    
    def _init_video_swin(self) -> None:
        """Initialize Video Swin Transformer"""
        try:
            import torch
            from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
            
            logger.info("Loading Video model from HuggingFace...")
            
            self.processor = VideoMAEImageProcessor.from_pretrained(
                "MCG-NJU/videomae-base-finetuned-kinetics"
            )
            self.model = VideoMAEForVideoClassification.from_pretrained(
                "MCG-NJU/videomae-base-finetuned-kinetics"
            )
            
            if self.device == "cuda":
                self.model = self.model.cuda()
            
            self._model_type = "videomae"
            
            logger.info("Video model loaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load Video Swin: {e}, using simplified model")
            self._init_simplified()
    
    def _init_simplified(self) -> None:
        """Initialize simplified action recognition using pose/motion"""
        logger.info("Using simplified motion-based action recognition")
        self._model_type = "simplified"
        self.model = None
    
    def _init_transforms_slowfast(self) -> None:
        """Initialize transforms for SlowFast"""
        try:
            from pytorchvideo.transforms import (
                ApplyTransformToKey,
                ShortSideScale,
                UniformTemporalSubsample,
                Normalize
            )
            from torchvision.transforms import Compose, Lambda
            
            self._transform = ApplyTransformToKey(
                key="video",
                transform=Compose([
                    UniformTemporalSubsample(32),
                    Lambda(lambda x: x / 255.0),
                    Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                    ShortSideScale(size=256),
                ]),
            )
        except ImportError:
            self._transform = None
    
    def _init_transforms_i3d(self) -> None:
        """Initialize transforms for I3D"""
        self._init_transforms_slowfast()  # Same transforms work
    
    def recognize_clip(
        self,
        frames: List[np.ndarray],
        start_time: float = 0,
        end_time: float = 0
    ) -> ClipActions:
        """
        Recognize actions in a video clip
        
        Args:
            frames: List of frame images (BGR)
            start_time: Clip start time
            end_time: Clip end time
        
        Returns:
            ClipActions with predictions
        """
        if self._model_type == "simplified":
            return self._recognize_simplified(frames, start_time, end_time)
        elif self._model_type == "videomae":
            return self._recognize_videomae(frames, start_time, end_time)
        else:
            return self._recognize_pytorch(frames, start_time, end_time)
    
    def _recognize_pytorch(
        self,
        frames: List[np.ndarray],
        start_time: float,
        end_time: float
    ) -> ClipActions:
        """Recognize using PyTorch video models"""
        import torch
        
        # Sample frames
        clip_length = self.config.action.clip_length
        if len(frames) > clip_length:
            indices = np.linspace(0, len(frames) - 1, clip_length, dtype=int)
            frames = [frames[i] for i in indices]
        
        # Convert to tensor
        frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
        frames_tensor = np.array(frames_rgb)  # T, H, W, C
        frames_tensor = frames_tensor.transpose(3, 0, 1, 2)  # C, T, H, W
        frames_tensor = torch.from_numpy(frames_tensor).float()
        
        # Normalize
        frames_tensor = frames_tensor / 255.0
        
        if self.device == "cuda":
            frames_tensor = frames_tensor.cuda()
        
        # For SlowFast, we need fast and slow pathways
        if self._model_type == "slowfast":
            # Create slow and fast inputs
            slow_pathway = torch.index_select(
                frames_tensor,
                1,
                torch.linspace(0, frames_tensor.shape[1] - 1, 8).long().to(frames_tensor.device)
            )
            fast_pathway = frames_tensor
            
            inputs = [slow_pathway.unsqueeze(0), fast_pathway.unsqueeze(0)]
        else:
            inputs = frames_tensor.unsqueeze(0)
        
        # Run inference
        with torch.no_grad():
            if self._model_type == "slowfast":
                outputs = self.model(inputs)
            else:
                outputs = self.model(inputs)
        
        # Get predictions
        probs = torch.softmax(outputs, dim=1)[0]
        top_k = self.config.action.top_k
        top_probs, top_indices = torch.topk(probs, top_k)
        
        predictions = []
        for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
            if prob >= self.config.action.threshold:
                label = self.labels[idx] if idx < len(self.labels) else f"action_{idx}"
                pred = ActionPrediction(
                    action=label,
                    confidence=float(prob),
                    start_time=start_time,
                    end_time=end_time
                )
                predictions.append(pred)
        
        return ClipActions(
            clip_number=0,
            start_time=start_time,
            end_time=end_time,
            predictions=predictions
        )
    
    def _recognize_videomae(
        self,
        frames: List[np.ndarray],
        start_time: float,
        end_time: float
    ) -> ClipActions:
        """Recognize using VideoMAE"""
        import torch
        
        # Convert frames to RGB PIL images
        frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
        
        # Sample to 16 frames
        if len(frames_rgb) > 16:
            indices = np.linspace(0, len(frames_rgb) - 1, 16, dtype=int)
            frames_rgb = [frames_rgb[i] for i in indices]
        
        # Process
        inputs = self.processor(frames_rgb, return_tensors="pt")
        
        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get predictions
        probs = torch.softmax(outputs.logits, dim=1)[0]
        top_k = self.config.action.top_k
        top_probs, top_indices = torch.topk(probs, top_k)
        
        predictions = []
        for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
            if prob >= self.config.action.threshold:
                label = self.model.config.id2label.get(idx, f"action_{idx}")
                pred = ActionPrediction(
                    action=label,
                    confidence=float(prob),
                    start_time=start_time,
                    end_time=end_time
                )
                predictions.append(pred)
        
        return ClipActions(
            clip_number=0,
            start_time=start_time,
            end_time=end_time,
            predictions=predictions
        )
    
    def _recognize_simplified(
        self,
        frames: List[np.ndarray],
        start_time: float,
        end_time: float
    ) -> ClipActions:
        """
        Simplified action recognition using motion analysis
        Works without heavy ML models
        """
        if len(frames) < 2:
            return ClipActions(
                clip_number=0,
                start_time=start_time,
                end_time=end_time,
                predictions=[]
            )
        
        # Calculate optical flow for motion analysis
        motion_magnitude = self._analyze_motion(frames)
        
        # Classify based on motion
        predictions = self._classify_by_motion(motion_magnitude, start_time, end_time)
        
        return ClipActions(
            clip_number=0,
            start_time=start_time,
            end_time=end_time,
            predictions=predictions
        )
    
    def _analyze_motion(self, frames: List[np.ndarray]) -> float:
        """Analyze motion in frames using optical flow"""
        total_motion = 0
        
        for i in range(len(frames) - 1):
            prev_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
            
            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            # Calculate magnitude
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            total_motion += np.mean(magnitude)
        
        avg_motion = total_motion / (len(frames) - 1)
        return avg_motion
    
    def _classify_by_motion(
        self,
        motion: float,
        start_time: float,
        end_time: float
    ) -> List[ActionPrediction]:
        """Classify action based on motion magnitude"""
        predictions = []
        
        if motion < 1.0:
            predictions.append(ActionPrediction(
                action="stationary",
                confidence=0.8,
                start_time=start_time,
                end_time=end_time
            ))
        elif motion < 3.0:
            predictions.append(ActionPrediction(
                action="slow_movement",
                confidence=0.7,
                start_time=start_time,
                end_time=end_time
            ))
            predictions.append(ActionPrediction(
                action="talking",
                confidence=0.5,
                start_time=start_time,
                end_time=end_time
            ))
        elif motion < 8.0:
            predictions.append(ActionPrediction(
                action="walking",
                confidence=0.7,
                start_time=start_time,
                end_time=end_time
            ))
            predictions.append(ActionPrediction(
                action="gesturing",
                confidence=0.5,
                start_time=start_time,
                end_time=end_time
            ))
        else:
            predictions.append(ActionPrediction(
                action="fast_movement",
                confidence=0.7,
                start_time=start_time,
                end_time=end_time
            ))
            predictions.append(ActionPrediction(
                action="running",
                confidence=0.5,
                start_time=start_time,
                end_time=end_time
            ))
        
        return predictions
    
    def process_video(
        self,
        video_path: str,
        clip_length: Optional[int] = None,
        overlap: float = 0.5
    ) -> List[ClipActions]:
        """
        Process entire video for action recognition
        
        Args:
            video_path: Path to video file
            clip_length: Frames per clip
            overlap: Overlap between clips (0-1)
        
        Returns:
            List of ClipActions for each clip
        """
        clip_length = clip_length or self.config.action.clip_length
        
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Processing video: {total_frames} frames at {fps} fps")
        
        step = int(clip_length * (1 - overlap))
        results = []
        clip_num = 0
        
        from tqdm import tqdm
        
        for start_frame in tqdm(range(0, total_frames - clip_length, step), desc="Recognizing actions"):
            # Read clip frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frames = []
            
            for _ in range(clip_length):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            
            if len(frames) < clip_length // 2:
                continue
            
            # Process clip
            start_time = start_frame / fps
            end_time = (start_frame + len(frames)) / fps
            
            clip_actions = self.recognize_clip(frames, start_time, end_time)
            clip_actions.clip_number = clip_num
            results.append(clip_actions)
            clip_num += 1
        
        cap.release()
        
        logger.info(f"Processed {len(results)} clips")
        return results
    
    def summarize_actions(
        self,
        clip_actions: List[ClipActions]
    ) -> Dict[str, Any]:
        """Summarize actions across all clips"""
        from collections import Counter
        
        action_counts = Counter()
        action_confidences = {}
        
        for clip in clip_actions:
            for pred in clip.predictions:
                action_counts[pred.action] += 1
                if pred.action not in action_confidences:
                    action_confidences[pred.action] = []
                action_confidences[pred.action].append(pred.confidence)
        
        # Calculate average confidence
        avg_confidences = {
            action: np.mean(confs)
            for action, confs in action_confidences.items()
        }
        
        # Get dominant actions
        dominant = action_counts.most_common(5)
        
        return {
            "total_clips": len(clip_actions),
            "action_counts": dict(action_counts),
            "action_confidences": avg_confidences,
            "dominant_actions": [action for action, _ in dominant],
            "unique_actions": len(action_counts)
        }
    
    def get_actions_timeline(
        self,
        clip_actions: List[ClipActions],
        min_confidence: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Get a timeline of actions
        
        Returns:
            List of action events with timestamps
        """
        timeline = []
        
        for clip in clip_actions:
            for pred in clip.predictions:
                if pred.confidence >= min_confidence:
                    timeline.append({
                        "action": pred.action,
                        "start": pred.start_time,
                        "end": pred.end_time,
                        "confidence": pred.confidence
                    })
        
        # Sort by start time
        timeline.sort(key=lambda x: x["start"])
        
        return timeline
