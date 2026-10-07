"""
Accuracy Metrics Module
Tracks and reports accuracy/confidence metrics for all pipeline components
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AudioMetrics:
    """Metrics for audio processing"""
    duration_seconds: float = 0.0
    sample_rate: int = 0
    has_speech: bool = False
    speech_percentage: float = 0.0
    avg_speech_confidence: float = 0.0
    word_count: int = 0
    segments_count: int = 0
    noise_level: str = "unknown"  # low, medium, high
    transcription_confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "duration_seconds": round(self.duration_seconds, 2),
            "sample_rate": self.sample_rate,
            "has_speech": self.has_speech,
            "speech_percentage": round(self.speech_percentage, 2),
            "avg_speech_confidence": round(self.avg_speech_confidence, 2),
            "word_count": self.word_count,
            "segments_count": self.segments_count,
            "noise_level": self.noise_level,
            "transcription_confidence": round(self.transcription_confidence, 2)
        }


@dataclass
class VideoMetrics:
    """Metrics for video processing"""
    duration_seconds: float = 0.0
    fps: float = 0.0
    width: int = 0
    height: int = 0
    total_frames: int = 0
    frames_analyzed: int = 0
    avg_brightness: float = 0.0
    motion_level: str = "unknown"  # static, low, medium, high
    scene_count: int = 0
    quality_score: float = 0.0  # 0-100
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "duration_seconds": round(self.duration_seconds, 2),
            "fps": round(self.fps, 2),
            "resolution": f"{self.width}x{self.height}",
            "total_frames": self.total_frames,
            "frames_analyzed": self.frames_analyzed,
            "avg_brightness": round(self.avg_brightness, 2),
            "motion_level": self.motion_level,
            "scene_count": self.scene_count,
            "quality_score": round(self.quality_score, 2)
        }


@dataclass
class ObjectDetectionMetrics:
    """Metrics for object detection"""
    unique_objects: int = 0
    total_detections: int = 0
    avg_confidence: float = 0.0
    high_confidence_count: int = 0  # confidence > 0.7
    object_types: List[str] = field(default_factory=list)
    detection_coverage: float = 0.0  # % of frames with detections
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "unique_objects": self.unique_objects,
            "total_detections": self.total_detections,
            "avg_confidence": round(self.avg_confidence, 2),
            "high_confidence_count": self.high_confidence_count,
            "object_types": self.object_types[:10],  # Top 10
            "detection_coverage_percent": round(self.detection_coverage, 2)
        }


@dataclass
class FaceDetectionMetrics:
    """Metrics for face detection"""
    unique_faces: int = 0
    total_detections: int = 0
    avg_confidence: float = 0.0
    frames_with_faces: int = 0
    face_coverage: float = 0.0  # % of frames with faces
    recognized_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "unique_faces": self.unique_faces,
            "total_detections": self.total_detections,
            "avg_confidence": round(self.avg_confidence, 2),
            "frames_with_faces": self.frames_with_faces,
            "face_coverage_percent": round(self.face_coverage, 2),
            "recognized_count": self.recognized_count
        }


@dataclass 
class SceneMetrics:
    """Metrics for scene detection"""
    scene_count: int = 0
    avg_scene_duration: float = 0.0
    indoor_count: int = 0
    outdoor_count: int = 0
    environments_detected: List[str] = field(default_factory=list)
    transition_quality: float = 0.0  # 0-100
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_count": self.scene_count,
            "avg_scene_duration_seconds": round(self.avg_scene_duration, 2),
            "indoor_scenes": self.indoor_count,
            "outdoor_scenes": self.outdoor_count,
            "environments": list(set(self.environments_detected)),
            "transition_quality": round(self.transition_quality, 2)
        }


@dataclass
class ActionMetrics:
    """Metrics for action recognition"""
    unique_actions: int = 0
    total_detections: int = 0
    avg_confidence: float = 0.0
    action_types: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "unique_actions": self.unique_actions,
            "total_detections": self.total_detections,
            "avg_confidence": round(self.avg_confidence, 2),
            "action_types": self.action_types[:10]
        }


@dataclass
class SpeakerMetrics:
    """Metrics for speaker diarization"""
    speaker_count: int = 0
    turns_count: int = 0
    avg_turn_duration: float = 0.0
    speaker_labels: List[str] = field(default_factory=list)
    diarization_confidence: float = 0.0
    gender_detected: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "speaker_count": self.speaker_count,
            "dialogue_turns": self.turns_count,
            "avg_turn_duration_seconds": round(self.avg_turn_duration, 2),
            "speakers": self.speaker_labels,
            "diarization_confidence": round(self.diarization_confidence, 2),
            "gender_breakdown": self.gender_detected
        }


@dataclass
class PipelineMetrics:
    """Complete pipeline metrics"""
    timestamp: str = ""
    total_processing_time: float = 0.0
    video_path: str = ""
    
    audio: AudioMetrics = field(default_factory=AudioMetrics)
    video: VideoMetrics = field(default_factory=VideoMetrics)
    objects: ObjectDetectionMetrics = field(default_factory=ObjectDetectionMetrics)
    faces: FaceDetectionMetrics = field(default_factory=FaceDetectionMetrics)
    scenes: SceneMetrics = field(default_factory=SceneMetrics)
    actions: ActionMetrics = field(default_factory=ActionMetrics)
    speakers: SpeakerMetrics = field(default_factory=SpeakerMetrics)
    
    # Overall scores
    overall_confidence: float = 0.0
    data_quality_score: float = 0.0
    
    def calculate_overall_scores(self):
        """Calculate overall confidence and quality scores"""
        scores = []
        weights = []
        
        # Audio score
        if self.audio.has_speech:
            scores.append(self.audio.transcription_confidence)
            weights.append(0.25)
        
        # Object detection score
        if self.objects.total_detections > 0:
            scores.append(self.objects.avg_confidence)
            weights.append(0.15)
        
        # Face detection score  
        if self.faces.total_detections > 0:
            scores.append(self.faces.avg_confidence)
            weights.append(0.15)
        
        # Scene detection score
        if self.scenes.scene_count > 0:
            scores.append(self.scenes.transition_quality / 100)
            weights.append(0.15)
        
        # Action recognition score
        if self.actions.total_detections > 0:
            scores.append(self.actions.avg_confidence)
            weights.append(0.15)
        
        # Speaker diarization score
        if self.speakers.speaker_count > 0:
            scores.append(self.speakers.diarization_confidence)
            weights.append(0.15)
        
        if scores and weights:
            total_weight = sum(weights)
            self.overall_confidence = sum(s * w for s, w in zip(scores, weights)) / total_weight
        else:
            self.overall_confidence = 0.5  # Default confidence
        
        # Data quality score based on coverage and completeness (capped at 100)
        quality_factors = []
        
        # Frame analysis coverage (what % of expected frames were analyzed)
        if self.video.total_frames > 0:
            expected = max(1, self.video.total_frames // 30)  # ~1 frame per second
            actual = self.video.frames_analyzed
            coverage = min(1.0, actual / expected)
            quality_factors.append(coverage)
        
        # Speech quality
        if self.audio.duration_seconds > 0:
            if self.audio.has_speech:
                speech_factor = min(1.0, self.audio.speech_percentage / 50)  # 50% speech is good
                quality_factors.append(speech_factor)
            else:
                quality_factors.append(0.3)  # Low quality if no speech
        
        # Object detection quality
        if self.video.frames_analyzed > 0:
            obj_coverage = min(1.0, self.objects.detection_coverage / 80)  # 80% coverage is good
            quality_factors.append(obj_coverage)
        
        if quality_factors:
            self.data_quality_score = min(100.0, (sum(quality_factors) / len(quality_factors)) * 100)
        else:
            self.data_quality_score = 50.0  # Default quality
    
    def to_dict(self) -> Dict[str, Any]:
        self.calculate_overall_scores()
        return {
            "metadata": {
                "timestamp": self.timestamp,
                "video_path": self.video_path,
                "total_processing_time_seconds": round(self.total_processing_time, 2)
            },
            "overall_scores": {
                "confidence_score": round(self.overall_confidence * 100, 2),
                "data_quality_score": round(self.data_quality_score, 2)
            },
            "audio_metrics": self.audio.to_dict(),
            "video_metrics": self.video.to_dict(),
            "object_detection_metrics": self.objects.to_dict(),
            "face_detection_metrics": self.faces.to_dict(),
            "scene_detection_metrics": self.scenes.to_dict(),
            "action_recognition_metrics": self.actions.to_dict(),
            "speaker_metrics": self.speakers.to_dict()
        }
    
    def generate_report(self) -> str:
        """Generate a human-readable metrics report"""
        self.calculate_overall_scores()
        
        lines = [
            "=" * 60,
            "VIDEO ANALYSIS ACCURACY METRICS REPORT",
            "=" * 60,
            "",
            f"Video: {self.video_path}",
            f"Processed: {self.timestamp}",
            f"Processing Time: {self.total_processing_time:.2f} seconds",
            "",
            "-" * 40,
            "OVERALL SCORES",
            "-" * 40,
            f"  Confidence Score: {self.overall_confidence * 100:.1f}%",
            f"  Data Quality Score: {self.data_quality_score:.1f}%",
            "",
            "-" * 40,
            "AUDIO ANALYSIS",
            "-" * 40,
            f"  Duration: {self.audio.duration_seconds:.2f}s",
            f"  Has Speech: {'Yes' if self.audio.has_speech else 'No'}",
            f"  Speech Coverage: {self.audio.speech_percentage:.1f}%",
            f"  Words Transcribed: {self.audio.word_count}",
            f"  Segments: {self.audio.segments_count}",
            f"  Transcription Confidence: {self.audio.transcription_confidence * 100:.1f}%",
            f"  Noise Level: {self.audio.noise_level}",
            "",
            "-" * 40,
            "VIDEO ANALYSIS",
            "-" * 40,
            f"  Duration: {self.video.duration_seconds:.2f}s",
            f"  Resolution: {self.video.width}x{self.video.height}",
            f"  FPS: {self.video.fps:.2f}",
            f"  Total Frames: {self.video.total_frames}",
            f"  Frames Analyzed: {self.video.frames_analyzed}",
            f"  Quality Score: {self.video.quality_score:.1f}%",
            f"  Motion Level: {self.video.motion_level}",
            "",
            "-" * 40,
            "SCENE DETECTION",
            "-" * 40,
            f"  Scenes Detected: {self.scenes.scene_count}",
            f"  Avg Scene Duration: {self.scenes.avg_scene_duration:.2f}s",
            f"  Indoor Scenes: {self.scenes.indoor_count}",
            f"  Outdoor Scenes: {self.scenes.outdoor_count}",
            f"  Environments: {', '.join(set(self.scenes.environments_detected)) or 'N/A'}",
            f"  Detection Quality: {self.scenes.transition_quality:.1f}%",
            "",
            "-" * 40,
            "OBJECT DETECTION",
            "-" * 40,
            f"  Unique Objects: {self.objects.unique_objects}",
            f"  Total Detections: {self.objects.total_detections}",
            f"  Avg Confidence: {self.objects.avg_confidence * 100:.1f}%",
            f"  High-Confidence: {self.objects.high_confidence_count}",
            f"  Frame Coverage: {self.objects.detection_coverage:.1f}%",
            f"  Object Types: {', '.join(self.objects.object_types[:5]) or 'N/A'}",
            "",
            "-" * 40,
            "FACE DETECTION",
            "-" * 40,
            f"  Unique Faces: {self.faces.unique_faces}",
            f"  Total Detections: {self.faces.total_detections}",
            f"  Avg Confidence: {self.faces.avg_confidence * 100:.1f}%",
            f"  Frame Coverage: {self.faces.face_coverage:.1f}%",
            "",
            "-" * 40,
            "ACTION RECOGNITION",
            "-" * 40,
            f"  Unique Actions: {self.actions.unique_actions}",
            f"  Total Detections: {self.actions.total_detections}",
            f"  Avg Confidence: {self.actions.avg_confidence * 100:.1f}%",
            f"  Actions: {', '.join(self.actions.action_types[:5]) or 'N/A'}",
            "",
            "-" * 40,
            "SPEAKER ANALYSIS",
            "-" * 40,
            f"  Speakers Detected: {self.speakers.speaker_count}",
            f"  Dialogue Turns: {self.speakers.turns_count}",
            f"  Avg Turn Duration: {self.speakers.avg_turn_duration:.2f}s",
            f"  Gender Breakdown: {self.speakers.gender_detected or 'N/A'}",
            f"  Diarization Confidence: {self.speakers.diarization_confidence * 100:.1f}%",
            "",
            "=" * 60
        ]
        
        return "\n".join(lines)


class MetricsCollector:
    """Collects metrics during pipeline execution"""
    
    def __init__(self):
        self.metrics = PipelineMetrics()
        self.metrics.timestamp = datetime.now().isoformat()
    
    def set_video_path(self, path: str):
        self.metrics.video_path = path
    
    def set_processing_time(self, time_seconds: float):
        self.metrics.total_processing_time = time_seconds
    
    def update_audio_metrics(self, transcription_result, audio_duration: float = 0):
        """Update audio metrics from transcription result"""
        if transcription_result is None:
            return
            
        self.metrics.audio.duration_seconds = audio_duration
        self.metrics.audio.word_count = getattr(transcription_result, 'word_count', 0)
        self.metrics.audio.has_speech = self.metrics.audio.word_count > 0
        
        segments = getattr(transcription_result, 'segments', [])
        self.metrics.audio.segments_count = len(segments)
        
        # Calculate speech percentage
        if segments and audio_duration > 0:
            speech_duration = sum(
                (getattr(s, 'end', 0) - getattr(s, 'start', 0)) 
                for s in segments
            )
            self.metrics.audio.speech_percentage = (speech_duration / audio_duration) * 100
        
        # Calculate average confidence
        confidences = []
        for seg in segments:
            if hasattr(seg, 'avg_logprob'):
                # Convert log probability to confidence (approximate)
                conf = min(1.0, max(0.0, 1.0 + seg.avg_logprob / 3))
                confidences.append(conf)
        
        if confidences:
            self.metrics.audio.avg_speech_confidence = sum(confidences) / len(confidences)
            self.metrics.audio.transcription_confidence = self.metrics.audio.avg_speech_confidence
        else:
            # Default confidence based on word count
            self.metrics.audio.transcription_confidence = 0.7 if self.metrics.audio.word_count > 10 else 0.5
    
    def update_video_metrics(self, metadata, frames_analyzed: int = 0):
        """Update video metrics from video metadata"""
        if metadata is None:
            return
            
        self.metrics.video.duration_seconds = getattr(metadata, 'duration', 0)
        self.metrics.video.fps = getattr(metadata, 'fps', 0)
        self.metrics.video.width = getattr(metadata, 'width', 0)
        self.metrics.video.height = getattr(metadata, 'height', 0)
        self.metrics.video.total_frames = getattr(metadata, 'total_frames', 0)
        self.metrics.video.frames_analyzed = frames_analyzed
        
        # Calculate quality score based on resolution
        resolution = self.metrics.video.width * self.metrics.video.height
        if resolution >= 1920 * 1080:
            self.metrics.video.quality_score = 100
        elif resolution >= 1280 * 720:
            self.metrics.video.quality_score = 80
        elif resolution >= 854 * 480:
            self.metrics.video.quality_score = 60
        else:
            self.metrics.video.quality_score = 40
    
    def update_object_metrics(self, object_results: List, total_frames: int = 1):
        """Update object detection metrics"""
        if not object_results:
            return
            
        all_detections = []
        object_counts = {}
        frames_with_objects = 0
        
        for frame_result in object_results:
            detections = getattr(frame_result, 'detections', [])
            if detections:
                frames_with_objects += 1
            for det in detections:
                class_name = getattr(det, 'class_name', 'unknown')
                confidence = getattr(det, 'confidence', 0)
                all_detections.append(confidence)
                object_counts[class_name] = object_counts.get(class_name, 0) + 1
        
        self.metrics.objects.total_detections = len(all_detections)
        self.metrics.objects.unique_objects = len(object_counts)
        self.metrics.objects.object_types = sorted(object_counts.keys(), key=lambda x: -object_counts[x])
        
        if all_detections:
            self.metrics.objects.avg_confidence = sum(all_detections) / len(all_detections)
            self.metrics.objects.high_confidence_count = sum(1 for c in all_detections if c > 0.7)
        
        self.metrics.objects.detection_coverage = (frames_with_objects / max(1, total_frames)) * 100
    
    def update_face_metrics(self, face_results: List, total_frames: int = 1):
        """Update face detection metrics"""
        if not face_results:
            return
            
        all_confidences = []
        frames_with_faces = 0
        unique_identities = set()
        
        for frame_result in face_results:
            detections = getattr(frame_result, 'detections', [])
            matches = getattr(frame_result, 'matches', [])
            
            if detections:
                frames_with_faces += 1
            
            for det in detections:
                conf = getattr(det, 'confidence', 0.5)
                all_confidences.append(conf)
            
            for match in matches:
                identity = getattr(match, 'identity', None)
                if identity:
                    unique_identities.add(identity)
        
        self.metrics.faces.total_detections = len(all_confidences)
        self.metrics.faces.unique_faces = len(unique_identities) or (1 if all_confidences else 0)
        self.metrics.faces.frames_with_faces = frames_with_faces
        self.metrics.faces.recognized_count = len(unique_identities)
        
        if all_confidences:
            self.metrics.faces.avg_confidence = sum(all_confidences) / len(all_confidences)
        
        self.metrics.faces.face_coverage = (frames_with_faces / max(1, total_frames)) * 100
    
    def update_scene_metrics(self, scenes: List, video_duration: float = 0):
        """Update scene detection metrics"""
        if not scenes:
            return
            
        self.metrics.scenes.scene_count = len(scenes)
        
        durations = []
        for scene in scenes:
            duration = getattr(scene, 'duration', 0) or (
                getattr(scene, 'end_time', 0) - getattr(scene, 'start_time', 0)
            )
            durations.append(duration)
            
            env = getattr(scene, 'environment', '')
            if env:
                self.metrics.scenes.environments_detected.append(env)
            
            is_indoor = getattr(scene, 'is_indoor', None)
            if is_indoor is True:
                self.metrics.scenes.indoor_count += 1
            elif is_indoor is False:
                self.metrics.scenes.outdoor_count += 1
        
        if durations:
            self.metrics.scenes.avg_scene_duration = sum(durations) / len(durations)
        
        # Quality based on reasonable scene detection
        if len(scenes) > 0 and video_duration > 0:
            avg_duration = video_duration / len(scenes)
            if 2 < avg_duration < 30:  # Reasonable scene lengths
                self.metrics.scenes.transition_quality = 80
            elif 1 < avg_duration < 60:
                self.metrics.scenes.transition_quality = 60
            else:
                self.metrics.scenes.transition_quality = 40
        else:
            self.metrics.scenes.transition_quality = 50
    
    def update_action_metrics(self, action_results: List):
        """Update action recognition metrics"""
        if not action_results:
            return
            
        all_confidences = []
        action_counts = {}
        
        for result in action_results:
            predictions = getattr(result, 'predictions', [])
            for pred in predictions:
                action = getattr(pred, 'action', 'unknown')
                confidence = getattr(pred, 'confidence', 0)
                all_confidences.append(confidence)
                action_counts[action] = action_counts.get(action, 0) + 1
        
        self.metrics.actions.total_detections = len(all_confidences)
        self.metrics.actions.unique_actions = len(action_counts)
        self.metrics.actions.action_types = sorted(action_counts.keys(), key=lambda x: -action_counts[x])
        
        if all_confidences:
            self.metrics.actions.avg_confidence = sum(all_confidences) / len(all_confidences)
    
    def update_speaker_metrics(self, video_analysis):
        """Update speaker metrics from video analysis"""
        if video_analysis is None:
            return
            
        speakers = set()
        turn_count = 0
        turn_durations = []
        gender_counts = {'Male': 0, 'Female': 0, 'Unknown': 0}
        
        for scene in getattr(video_analysis, 'scenes', []):
            for dlg in getattr(scene, 'dialogues', []):
                speaker = getattr(dlg, 'speaker', 'Unknown')
                speakers.add(speaker)
                turn_count += 1
                
                start = getattr(dlg, 'start_time', 0)
                end = getattr(dlg, 'end_time', 0)
                if end > start:
                    turn_durations.append(end - start)
                
                # Count genders
                if 'Man' in speaker or 'Male' in speaker:
                    gender_counts['Male'] += 1
                elif 'Woman' in speaker or 'Female' in speaker:
                    gender_counts['Female'] += 1
                else:
                    gender_counts['Unknown'] += 1
        
        self.metrics.speakers.speaker_count = len(speakers)
        self.metrics.speakers.turns_count = turn_count
        self.metrics.speakers.speaker_labels = list(speakers)
        self.metrics.speakers.gender_detected = {k: v for k, v in gender_counts.items() if v > 0}
        
        if turn_durations:
            self.metrics.speakers.avg_turn_duration = sum(turn_durations) / len(turn_durations)
        
        # Confidence based on speaker detection quality
        if len(speakers) > 0 and turn_count > 0:
            self.metrics.speakers.diarization_confidence = min(0.9, 0.5 + len(speakers) * 0.1)
        else:
            self.metrics.speakers.diarization_confidence = 0.3
    
    def get_metrics(self) -> PipelineMetrics:
        """Get the collected metrics"""
        return self.metrics
    
    def save_metrics(self, output_path: str):
        """Save metrics to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics.to_dict(), f, indent=2, ensure_ascii=False)
    
    def save_report(self, output_path: str):
        """Save human-readable report"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(self.metrics.generate_report())
