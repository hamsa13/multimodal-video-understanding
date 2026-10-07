"""
Video & Audio Extraction Module
Handles frame extraction and audio extraction from video files
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional, Generator
from dataclasses import dataclass
import logging

import cv2
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class FrameInfo:
    """Information about an extracted frame"""
    frame_number: int
    timestamp: float
    path: str
    image: Optional[np.ndarray] = None


@dataclass
class VideoMetadata:
    """Video file metadata"""
    path: str
    duration: float
    fps: float
    width: int
    height: int
    total_frames: int
    codec: str
    has_audio: bool


class VideoProcessor:
    """
    Video and Audio processing utilities
    Extracts frames and audio from video files
    """
    
    def __init__(self, config):
        self.config = config
        self.temp_dir = Path(config.temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def get_metadata(self, video_path: str) -> VideoMetadata:
        """Extract video metadata using OpenCV and ffprobe"""
        video_path = str(video_path)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        codec = self._get_codec(cap)
        cap.release()
        
        # Check for audio using ffprobe
        has_audio = self._check_audio(video_path)
        
        return VideoMetadata(
            path=video_path,
            duration=duration,
            fps=fps,
            width=width,
            height=height,
            total_frames=total_frames,
            codec=codec,
            has_audio=has_audio
        )
    
    def _get_codec(self, cap: cv2.VideoCapture) -> str:
        """Get video codec from capture"""
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        return codec
    
    def _check_audio(self, video_path: str) -> bool:
        """Check if video has audio stream"""
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'quiet', '-select_streams', 'a',
                 '-show_entries', 'stream=codec_type', '-of', 'csv=p=0',
                 video_path],
                capture_output=True, text=True, timeout=30
            )
            return 'audio' in result.stdout.lower()
        except Exception:
            return True  # Assume audio exists
    
    def extract_frames(
        self,
        video_path: str,
        output_dir: Optional[str] = None,
        fps: Optional[int] = None,
        max_frames: Optional[int] = None,
        start_time: float = 0,
        end_time: Optional[float] = None,
        return_images: bool = False
    ) -> List[FrameInfo]:
        """
        Extract frames from video
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save frames (if None, uses temp)
            fps: Frames per second to extract (default from config)
            max_frames: Maximum frames to extract
            start_time: Start time in seconds
            end_time: End time in seconds
            return_images: Whether to return numpy arrays
        
        Returns:
            List of FrameInfo objects
        """
        video_path = str(video_path)
        fps = fps or self.config.video.fps
        max_frames = max_frames or self.config.video.max_frames
        
        if output_dir is None:
            output_dir = self.temp_dir / "frames"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps if video_fps > 0 else 0
        
        if end_time is None:
            end_time = duration
        
        # Calculate frame sampling
        frame_interval = int(video_fps / fps) if fps < video_fps else 1
        start_frame = int(start_time * video_fps)
        end_frame = int(min(end_time, duration) * video_fps)
        
        frames: List[FrameInfo] = []
        frame_count = 0
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        pbar = tqdm(total=min(max_frames, (end_frame - start_frame) // frame_interval),
                    desc="Extracting frames")
        
        current_frame = start_frame
        while current_frame < end_frame and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if (current_frame - start_frame) % frame_interval == 0:
                timestamp = current_frame / video_fps
                frame_path = str(output_dir / f"frame_{frame_count:06d}.{self.config.video.output_format}")
                
                # Save frame
                if self.config.video.output_format.lower() == 'jpg':
                    cv2.imwrite(frame_path, frame, 
                               [cv2.IMWRITE_JPEG_QUALITY, self.config.video.quality])
                else:
                    cv2.imwrite(frame_path, frame)
                
                frame_info = FrameInfo(
                    frame_number=frame_count,
                    timestamp=timestamp,
                    path=frame_path,
                    image=frame if return_images else None
                )
                frames.append(frame_info)
                frame_count += 1
                pbar.update(1)
            
            current_frame += 1
        
        cap.release()
        pbar.close()
        
        logger.info(f"Extracted {len(frames)} frames from {video_path}")
        return frames
    
    def extract_frames_generator(
        self,
        video_path: str,
        fps: Optional[int] = None,
        start_time: float = 0,
        end_time: Optional[float] = None
    ) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """
        Generate frames without saving to disk (memory efficient)
        
        Yields:
            Tuple of (frame_number, timestamp, frame_array)
        """
        video_path = str(video_path)
        fps = fps or self.config.video.fps
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps if video_fps > 0 else 0
        
        if end_time is None:
            end_time = duration
        
        frame_interval = int(video_fps / fps) if fps < video_fps else 1
        start_frame = int(start_time * video_fps)
        end_frame = int(min(end_time, duration) * video_fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frame_count = 0
        current_frame = start_frame
        
        while current_frame < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            
            if (current_frame - start_frame) % frame_interval == 0:
                timestamp = current_frame / video_fps
                yield frame_count, timestamp, frame
                frame_count += 1
            
            current_frame += 1
        
        cap.release()
    
    def extract_audio(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
        start_time: float = 0,
        end_time: Optional[float] = None
    ) -> str:
        """
        Extract audio from video file
        
        Args:
            video_path: Path to video file
            output_path: Path for output audio (if None, uses temp)
            sample_rate: Audio sample rate
            channels: Number of audio channels (1=mono, 2=stereo)
            start_time: Start time in seconds
            end_time: End time in seconds
        
        Returns:
            Path to extracted audio file
        """
        video_path = str(video_path)
        sample_rate = sample_rate or self.config.video.sample_rate
        channels = channels or self.config.video.channels
        audio_format = self.config.video.audio_format
        
        if output_path is None:
            output_path = self.temp_dir / f"audio.{audio_format}"
        output_path = str(output_path)
        
        # Build ffmpeg command
        cmd = ['ffmpeg', '-y', '-i', video_path]
        
        # Time range
        if start_time > 0:
            cmd.extend(['-ss', str(start_time)])
        if end_time is not None:
            cmd.extend(['-t', str(end_time - start_time)])
        
        # Audio settings
        cmd.extend([
            '-vn',  # No video
            '-ar', str(sample_rate),
            '-ac', str(channels),
            '-q:a', '0',  # Best quality
            output_path
        ])
        
        logger.info(f"Extracting audio: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg error: {result.stderr}")
        
        logger.info(f"Extracted audio to {output_path}")
        return output_path
    
    def extract_clip(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        output_path: Optional[str] = None
    ) -> str:
        """Extract a video clip"""
        video_path = str(video_path)
        
        if output_path is None:
            output_path = str(self.temp_dir / f"clip_{start_time}_{end_time}.mp4")
        
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-ss', str(start_time),
            '-t', str(end_time - start_time),
            '-c', 'copy',
            output_path
        ]
        
        subprocess.run(cmd, capture_output=True)
        return output_path
    
    def get_frame_at_time(
        self,
        video_path: str,
        timestamp: float
    ) -> np.ndarray:
        """Get a single frame at specific timestamp"""
        video_path = str(video_path)
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(timestamp * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"Cannot read frame at {timestamp}s")
        
        return frame
    
    def cleanup(self) -> None:
        """Clean up temporary files"""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            logger.info(f"Cleaned up temp directory: {self.temp_dir}")
