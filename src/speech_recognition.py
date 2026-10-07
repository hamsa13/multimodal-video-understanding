"""
Speech Recognition Module
Transcribes audio using OpenAI Whisper or Faster Whisper
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass, field
import logging

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WordInfo:
    """Information about a transcribed word"""
    word: str
    start: float
    end: float
    confidence: float = 1.0


@dataclass
class TranscriptSegment:
    """A segment of transcribed speech"""
    id: int
    start: float
    end: float
    text: str
    words: List[WordInfo] = field(default_factory=list)
    speaker: Optional[str] = None
    confidence: float = 1.0
    language: Optional[str] = None


@dataclass
class TranscriptionResult:
    """Complete transcription result"""
    text: str
    segments: List[TranscriptSegment]
    language: str
    duration: float
    word_count: int


class SpeechRecognizer:
    """
    Speech-to-text using Whisper models
    Supports both OpenAI Whisper and Faster-Whisper
    """
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.model_name = config.speech.model
        self.use_faster = config.speech.use_faster_whisper
        self.device = config.device
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the Whisper model"""
        if self.use_faster:
            self._load_faster_whisper()
        else:
            self._load_openai_whisper()
    
    def _load_faster_whisper(self) -> None:
        """Load Faster-Whisper model"""
        try:
            from faster_whisper import WhisperModel
            
            compute_type = "float16" if self.device == "cuda" else "int8"
            
            logger.info(f"Loading Faster-Whisper model: {self.model_name}")
            self.model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=compute_type
            )
            self._model_type = "faster"
            logger.info("Faster-Whisper model loaded successfully")
            
        except ImportError:
            logger.warning("Faster-Whisper not available, falling back to OpenAI Whisper")
            self._load_openai_whisper()
    
    def _load_openai_whisper(self) -> None:
        """Load OpenAI Whisper model"""
        import whisper
        
        logger.info(f"Loading OpenAI Whisper model: {self.model_name}")
        self.model = whisper.load_model(self.model_name, device=self.device)
        self._model_type = "openai"
        logger.info("OpenAI Whisper model loaded successfully")
    
    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: Optional[str] = None,
        word_timestamps: Optional[bool] = None,
        initial_prompt: Optional[str] = None,
        **kwargs
    ) -> TranscriptionResult:
        """
        Transcribe audio file
        
        Args:
            audio_path: Path to audio file
            language: Language code (e.g., 'en', 'es') or None for auto-detect
            task: 'transcribe' or 'translate'
            word_timestamps: Whether to include word-level timestamps
            initial_prompt: Optional prompt to guide transcription
        
        Returns:
            TranscriptionResult with full transcription
        """
        audio_path = str(audio_path)
        language = language or self.config.speech.language
        task = task or self.config.speech.task
        word_timestamps = word_timestamps if word_timestamps is not None else self.config.speech.word_timestamps
        
        logger.info(f"Transcribing: {audio_path}")
        
        if self._model_type == "faster":
            return self._transcribe_faster(
                audio_path, language, task, word_timestamps, initial_prompt
            )
        else:
            return self._transcribe_openai(
                audio_path, language, task, word_timestamps, initial_prompt
            )
    
    def _transcribe_faster(
        self,
        audio_path: str,
        language: Optional[str],
        task: str,
        word_timestamps: bool,
        initial_prompt: Optional[str]
    ) -> TranscriptionResult:
        """Transcribe using Faster-Whisper"""
        segments_iter, info = self.model.transcribe(
            audio_path,
            language=language,
            task=task,
            beam_size=self.config.speech.beam_size,
            word_timestamps=word_timestamps,
            initial_prompt=initial_prompt
        )
        
        segments: List[TranscriptSegment] = []
        full_text_parts: List[str] = []
        
        for i, segment in enumerate(segments_iter):
            words = []
            if word_timestamps and segment.words:
                words = [
                    WordInfo(
                        word=w.word.strip(),
                        start=w.start,
                        end=w.end,
                        confidence=w.probability if hasattr(w, 'probability') else 1.0
                    )
                    for w in segment.words
                ]
            
            seg = TranscriptSegment(
                id=i,
                start=segment.start,
                end=segment.end,
                text=segment.text.strip(),
                words=words,
                confidence=segment.avg_logprob if hasattr(segment, 'avg_logprob') else 1.0,
                language=info.language
            )
            segments.append(seg)
            full_text_parts.append(segment.text.strip())
        
        full_text = " ".join(full_text_parts)
        
        return TranscriptionResult(
            text=full_text,
            segments=segments,
            language=info.language,
            duration=info.duration,
            word_count=len(full_text.split())
        )
    
    def _transcribe_openai(
        self,
        audio_path: str,
        language: Optional[str],
        task: str,
        word_timestamps: bool,
        initial_prompt: Optional[str]
    ) -> TranscriptionResult:
        """Transcribe using OpenAI Whisper"""
        import whisper
        
        options = {
            "task": task,
            "beam_size": self.config.speech.beam_size,
            "word_timestamps": word_timestamps,
        }
        
        if language:
            options["language"] = language
        if initial_prompt:
            options["initial_prompt"] = initial_prompt
        
        result = self.model.transcribe(audio_path, **options)
        
        segments: List[TranscriptSegment] = []
        
        for i, seg in enumerate(result.get("segments", [])):
            words = []
            if word_timestamps and "words" in seg:
                words = [
                    WordInfo(
                        word=w["word"].strip(),
                        start=w["start"],
                        end=w["end"],
                        confidence=w.get("probability", 1.0)
                    )
                    for w in seg["words"]
                ]
            
            segment = TranscriptSegment(
                id=i,
                start=seg["start"],
                end=seg["end"],
                text=seg["text"].strip(),
                words=words,
                language=result.get("language", "unknown")
            )
            segments.append(segment)
        
        # Calculate duration from last segment
        duration = segments[-1].end if segments else 0
        
        return TranscriptionResult(
            text=result["text"].strip(),
            segments=segments,
            language=result.get("language", "unknown"),
            duration=duration,
            word_count=len(result["text"].split())
        )
    
    def transcribe_with_timestamps(
        self,
        audio_path: str,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Transcribe and return simple timestamp format
        
        Returns:
            List of dicts with 'start', 'end', 'text'
        """
        result = self.transcribe(audio_path, word_timestamps=False, **kwargs)
        
        return [
            {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text
            }
            for seg in result.segments
        ]
    
    def detect_language(self, audio_path: str) -> str:
        """Detect the language of audio file"""
        if self._model_type == "faster":
            _, info = self.model.transcribe(audio_path, task="transcribe")
            return info.language
        else:
            import whisper
            audio = whisper.load_audio(audio_path)
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(self.device)
            _, probs = self.model.detect_language(mel)
            return max(probs, key=probs.get)
    
    def to_srt(self, result: TranscriptionResult) -> str:
        """Convert transcription to SRT subtitle format"""
        srt_content = []
        
        for i, seg in enumerate(result.segments, 1):
            start = self._format_timestamp_srt(seg.start)
            end = self._format_timestamp_srt(seg.end)
            
            srt_content.append(f"{i}")
            srt_content.append(f"{start} --> {end}")
            srt_content.append(seg.text)
            srt_content.append("")
        
        return "\n".join(srt_content)
    
    def to_vtt(self, result: TranscriptionResult) -> str:
        """Convert transcription to WebVTT subtitle format"""
        vtt_content = ["WEBVTT", ""]
        
        for seg in result.segments:
            start = self._format_timestamp_vtt(seg.start)
            end = self._format_timestamp_vtt(seg.end)
            
            vtt_content.append(f"{start} --> {end}")
            vtt_content.append(seg.text)
            vtt_content.append("")
        
        return "\n".join(vtt_content)
    
    @staticmethod
    def _format_timestamp_srt(seconds: float) -> str:
        """Format timestamp for SRT (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    @staticmethod
    def _format_timestamp_vtt(seconds: float) -> str:
        """Format timestamp for VTT (HH:MM:SS.mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
