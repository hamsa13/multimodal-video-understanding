"""
Speaker Diarization Module
Identifies who is speaking when using Pyannote.audio
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class SpeakerSegment:
    """A segment of speech by one speaker"""
    speaker: str
    start: float
    end: float
    confidence: float = 1.0


@dataclass
class DiarizationResult:
    """Complete diarization result"""
    segments: List[SpeakerSegment]
    speakers: List[str]
    duration: float
    num_speakers: int


class SpeakerDiarizer:
    """
    Speaker Diarization using Pyannote.audio
    Identifies different speakers in an audio file
    """
    
    def __init__(self, config):
        self.config = config
        self.pipeline = None
        self.device = config.device
        self._load_pipeline()
    
    def _load_pipeline(self) -> None:
        """Load the diarization pipeline"""
        if not self.config.diarization.enabled:
            logger.info("Speaker diarization is disabled")
            return
        
        try:
            from pyannote.audio import Pipeline
            import torch
            
            hf_token = self.config.diarization.huggingface_token
            if not hf_token:
                hf_token = os.getenv('HF_TOKEN')
            
            if not hf_token:
                logger.warning(
                    "HuggingFace token not found. Speaker diarization requires "
                    "a token for pyannote.audio. Set HF_TOKEN environment variable."
                )
                return
            
            logger.info("Loading Pyannote speaker diarization pipeline...")
            # pyannote.audio API changed; try both old and new token param names
            try:
                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    token=hf_token
                )
            except TypeError:
                # older versions expect use_auth_token
                try:
                    self.pipeline = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1",
                        use_auth_token=hf_token
                    )
                except Exception as e:
                    raise e
            
            # Move to GPU if available
            if self.device == "cuda" and torch.cuda.is_available():
                import torch
                self.pipeline = self.pipeline.to(torch.device("cuda"))
            
            logger.info("Diarization pipeline loaded successfully")
            
        except ImportError:
            logger.error("pyannote.audio not installed. Run: pip install pyannote.audio")
        except Exception as e:
            logger.error(f"Failed to load diarization pipeline: {e}")
    
    def diarize(
        self,
        audio_path: str,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None
    ) -> DiarizationResult:
        """
        Perform speaker diarization on audio file
        
        Args:
            audio_path: Path to audio file
            min_speakers: Minimum expected speakers
            max_speakers: Maximum expected speakers
        
        Returns:
            DiarizationResult with speaker segments
        """
        audio_path = str(audio_path)
        min_speakers = min_speakers or self.config.diarization.min_speakers
        max_speakers = max_speakers or self.config.diarization.max_speakers
        
        if self.pipeline is None:
            logger.warning("Diarization pipeline not loaded, returning empty result")
            return DiarizationResult(
                segments=[],
                speakers=[],
                duration=0,
                num_speakers=0
            )
        
        logger.info(f"Diarizing audio: {audio_path}")
        
        # Run diarization
        diarization = self.pipeline(
            audio_path,
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )
        
        # Convert to our format
        segments: List[SpeakerSegment] = []
        speakers_set = set()
        max_end = 0
        
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segment = SpeakerSegment(
                speaker=speaker,
                start=turn.start,
                end=turn.end
            )
            segments.append(segment)
            speakers_set.add(speaker)
            max_end = max(max_end, turn.end)
        
        # Sort by start time
        segments.sort(key=lambda x: x.start)
        speakers = sorted(list(speakers_set))
        
        logger.info(f"Found {len(speakers)} speakers in {len(segments)} segments")
        
        return DiarizationResult(
            segments=segments,
            speakers=speakers,
            duration=max_end,
            num_speakers=len(speakers)
        )
    
    def merge_with_transcription(
        self,
        diarization: DiarizationResult,
        transcription_segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Merge diarization results with transcription segments
        
        Args:
            diarization: DiarizationResult
            transcription_segments: List of segments with 'start', 'end', 'text'
        
        Returns:
            List of segments with speaker labels added
        """
        if not diarization.segments:
            return transcription_segments
        
        merged = []
        
        for trans_seg in transcription_segments:
            trans_start = trans_seg.get('start', 0)
            trans_end = trans_seg.get('end', 0)
            trans_mid = (trans_start + trans_end) / 2
            
            # Find the speaker for this segment
            speaker = self._find_speaker_at_time(diarization.segments, trans_mid)
            
            merged_seg = {
                **trans_seg,
                'speaker': speaker
            }
            merged.append(merged_seg)
        
        return merged
    
    def _find_speaker_at_time(
        self,
        segments: List[SpeakerSegment],
        time: float
    ) -> Optional[str]:
        """Find which speaker is talking at a specific time"""
        for seg in segments:
            if seg.start <= time <= seg.end:
                return seg.speaker
        
        # If no exact match, find closest
        min_dist = float('inf')
        closest_speaker = None
        
        for seg in segments:
            mid = (seg.start + seg.end) / 2
            dist = abs(mid - time)
            if dist < min_dist:
                min_dist = dist
                closest_speaker = seg.speaker
        
        return closest_speaker
    
    def get_speaker_timeline(
        self,
        diarization: DiarizationResult
    ) -> Dict[str, List[Tuple[float, float]]]:
        """
        Get timeline for each speaker
        
        Returns:
            Dict mapping speaker ID to list of (start, end) tuples
        """
        timeline = {speaker: [] for speaker in diarization.speakers}
        
        for seg in diarization.segments:
            timeline[seg.speaker].append((seg.start, seg.end))
        
        return timeline
    
    def get_speaker_stats(
        self,
        diarization: DiarizationResult
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate statistics for each speaker
        
        Returns:
            Dict with speaking time, percentage, and segment count per speaker
        """
        stats = {}
        total_time = 0
        
        for speaker in diarization.speakers:
            speaker_segments = [s for s in diarization.segments if s.speaker == speaker]
            speaking_time = sum(s.end - s.start for s in speaker_segments)
            total_time += speaking_time
            
            stats[speaker] = {
                'speaking_time': speaking_time,
                'segment_count': len(speaker_segments)
            }
        
        # Calculate percentages
        for speaker in stats:
            if total_time > 0:
                stats[speaker]['percentage'] = (stats[speaker]['speaking_time'] / total_time) * 100
            else:
                stats[speaker]['percentage'] = 0
        
        return stats
    
    def assign_names(
        self,
        diarization: DiarizationResult,
        speaker_names: Dict[str, str]
    ) -> DiarizationResult:
        """
        Assign human-readable names to speakers
        
        Args:
            diarization: Original DiarizationResult
            speaker_names: Dict mapping speaker IDs to names (e.g., {'SPEAKER_00': 'John'})
        
        Returns:
            New DiarizationResult with updated names
        """
        new_segments = []
        
        for seg in diarization.segments:
            new_speaker = speaker_names.get(seg.speaker, seg.speaker)
            new_seg = SpeakerSegment(
                speaker=new_speaker,
                start=seg.start,
                end=seg.end,
                confidence=seg.confidence
            )
            new_segments.append(new_seg)
        
        new_speakers = [speaker_names.get(s, s) for s in diarization.speakers]
        
        return DiarizationResult(
            segments=new_segments,
            speakers=new_speakers,
            duration=diarization.duration,
            num_speakers=diarization.num_speakers
        )
