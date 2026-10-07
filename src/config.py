"""
Configuration Manager
Loads and validates configuration from YAML files
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv


@dataclass
class VideoConfig:
    fps: int = 1
    max_frames: int = 1000
    output_format: str = "jpg"
    quality: int = 95
    sample_rate: int = 16000
    channels: int = 1
    audio_format: str = "wav"


@dataclass
class SpeechConfig:
    model: str = "base"
    language: Optional[str] = None
    task: str = "transcribe"
    beam_size: int = 5
    word_timestamps: bool = True
    use_faster_whisper: bool = True


@dataclass
class DiarizationConfig:
    enabled: bool = True
    min_speakers: int = 1
    max_speakers: int = 10
    huggingface_token: str = ""


@dataclass
class FaceConfig:
    detection_model: str = "mtcnn"
    confidence_threshold: float = 0.9
    min_face_size: int = 50
    recognition_model: str = "VGG-Face"
    distance_metric: str = "cosine"
    threshold: float = 0.6
    known_faces_dir: str = "known_faces"
    store_type: str = "chromadb"


@dataclass
class SceneConfig:
    detector: str = "content"
    threshold: float = 30.0
    min_scene_length: int = 15


@dataclass
class ObjectConfig:
    model: str = "yolov8n.pt"
    confidence: float = 0.5
    iou_threshold: float = 0.45
    classes: Optional[list] = None
    track_objects: bool = True


@dataclass
class ActionConfig:
    model: str = "slowfast"
    clip_length: int = 32
    sampling_rate: int = 2
    top_k: int = 5
    threshold: float = 0.3


@dataclass
class FusionConfig:
    strategy: str = "temporal"
    window_size: int = 5
    overlap: float = 0.5


@dataclass
class LLMConfig:
    provider: str = "openai"
    model: str = "gpt-4"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.7
    max_tokens: int = 2048
    prompts: Dict[str, str] = field(default_factory=dict)


@dataclass
class Config:
    """Main configuration container"""
    output_dir: str = "outputs"
    temp_dir: str = "temp"
    log_level: str = "INFO"
    device: str = "auto"
    max_workers: int = 4
    
    video: VideoConfig = field(default_factory=VideoConfig)
    speech: SpeechConfig = field(default_factory=SpeechConfig)
    diarization: DiarizationConfig = field(default_factory=DiarizationConfig)
    face: FaceConfig = field(default_factory=FaceConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    object_detection: ObjectConfig = field(default_factory=ObjectConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    
    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "Config":
        """Load configuration from YAML file"""
        load_dotenv()
        
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        
        config_path = Path(config_path)
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f)
        else:
            data = {}
        
        config = cls()
        config._load_from_dict(data)
        config._resolve_env_vars()
        config._setup_device()
        
        return config
    
    def _load_from_dict(self, data: Dict[str, Any]) -> None:
        """Load values from dictionary"""
        general = data.get('general', {})
        self.output_dir = general.get('output_dir', self.output_dir)
        self.temp_dir = general.get('temp_dir', self.temp_dir)
        self.log_level = general.get('log_level', self.log_level)
        self.device = general.get('device', self.device)
        self.max_workers = general.get('max_workers', self.max_workers)
        
        # Video config
        video_data = data.get('video', {})
        frame_data = video_data.get('frame_extraction', {})
        audio_data = video_data.get('audio_extraction', {})
        self.video = VideoConfig(
            fps=frame_data.get('fps', 1),
            max_frames=frame_data.get('max_frames', 1000),
            output_format=frame_data.get('output_format', 'jpg'),
            quality=frame_data.get('quality', 95),
            sample_rate=audio_data.get('sample_rate', 16000),
            channels=audio_data.get('channels', 1),
            audio_format=audio_data.get('format', 'wav')
        )
        
        # Speech config
        speech_data = data.get('speech', {})
        self.speech = SpeechConfig(
            model=speech_data.get('model', 'base'),
            language=speech_data.get('language'),
            task=speech_data.get('task', 'transcribe'),
            beam_size=speech_data.get('beam_size', 5),
            word_timestamps=speech_data.get('word_timestamps', True),
            use_faster_whisper=speech_data.get('use_faster_whisper', True)
        )
        
        # Diarization config
        diar_data = data.get('diarization', {})
        self.diarization = DiarizationConfig(
            enabled=diar_data.get('enabled', True),
            min_speakers=diar_data.get('min_speakers', 1),
            max_speakers=diar_data.get('max_speakers', 10),
            huggingface_token=diar_data.get('huggingface_token', '')
        )
        
        # Face config
        face_data = data.get('face', {})
        detection = face_data.get('detection', {})
        recognition = face_data.get('recognition', {})
        embedding = face_data.get('embedding', {})
        self.face = FaceConfig(
            detection_model=detection.get('model', 'mtcnn'),
            confidence_threshold=detection.get('confidence_threshold', 0.9),
            min_face_size=detection.get('min_face_size', 50),
            recognition_model=recognition.get('model', 'VGG-Face'),
            distance_metric=recognition.get('distance_metric', 'cosine'),
            threshold=recognition.get('threshold', 0.6),
            known_faces_dir=recognition.get('known_faces_dir', 'known_faces'),
            store_type=embedding.get('store_type', 'chromadb')
        )
        
        # Scene config
        scene_data = data.get('scene', {})
        self.scene = SceneConfig(
            detector=scene_data.get('detector', 'content'),
            threshold=scene_data.get('threshold', 30.0),
            min_scene_length=scene_data.get('min_scene_length', 15)
        )
        
        # Object detection config
        obj_data = data.get('object_detection', {})
        self.object_detection = ObjectConfig(
            model=obj_data.get('model', 'yolov8n.pt'),
            confidence=obj_data.get('confidence', 0.5),
            iou_threshold=obj_data.get('iou_threshold', 0.45),
            classes=obj_data.get('classes'),
            track_objects=obj_data.get('track_objects', True)
        )
        
        # Action config
        action_data = data.get('action', {})
        self.action = ActionConfig(
            model=action_data.get('model', 'slowfast'),
            clip_length=action_data.get('clip_length', 32),
            sampling_rate=action_data.get('sampling_rate', 2),
            top_k=action_data.get('top_k', 5),
            threshold=action_data.get('threshold', 0.3)
        )

        # Fusion config
        fusion_data = data.get('fusion', {})
        self.fusion = FusionConfig(
            strategy=fusion_data.get('strategy', 'temporal'),
            window_size=fusion_data.get('window_size', 5),
            overlap=fusion_data.get('overlap', 0.5)
        )
        
        # LLM config
        llm_data = data.get('llm', {})
        self.llm = LLMConfig(
            provider=llm_data.get('provider', 'openai'),
            model=llm_data.get('model', 'gpt-4'),
            api_key_env=llm_data.get('api_key_env', 'OPENAI_API_KEY'),
            temperature=llm_data.get('temperature', 0.7),
            max_tokens=llm_data.get('max_tokens', 2048),
            prompts=llm_data.get('prompts', {})
        )
    
    def _resolve_env_vars(self) -> None:
        """Resolve environment variables"""
        if not self.diarization.huggingface_token:
            self.diarization.huggingface_token = os.getenv('HF_TOKEN', '')
    
    def _setup_device(self) -> None:
        """Setup compute device"""
        if self.device == "auto":
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
    
    def save(self, path: str) -> None:
        """Save configuration to file"""
        # Convert to dict and save
        pass  # Implement if needed
    
    def create_dirs(self) -> None:
        """Create necessary directories"""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.temp_dir).mkdir(parents=True, exist_ok=True)
        Path(self.face.known_faces_dir).mkdir(parents=True, exist_ok=True)
