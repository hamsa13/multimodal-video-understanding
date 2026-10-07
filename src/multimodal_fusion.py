"""
Multimodal Fusion Module
Combines data from all analysis modules into structured output
"""

import json
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import logging

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DialogueTurn:
    """A single dialogue turn"""
    speaker: str
    text: str
    start_time: float
    end_time: float
    emotion: Optional[str] = None


@dataclass
class PersonPresence:
    """A person's presence in a scene"""
    name: str
    is_speaking: bool = False
    face_visible: bool = False
    position: Optional[str] = None  # e.g., "left", "center", "right"


@dataclass
class SceneData:
    """Complete data for a single scene"""
    scene_number: int
    start_time: float
    end_time: float
    duration: float
    
    # Environment
    location: Optional[str] = None
    is_indoor: Optional[bool] = None
    lighting: Optional[str] = None  # "bright", "dim", "dark"
    
    # People
    people: List[PersonPresence] = field(default_factory=list)
    face_count: int = 0
    
    # Dialogue
    dialogues: List[DialogueTurn] = field(default_factory=list)
    
    # Actions
    actions: List[str] = field(default_factory=list)
    dominant_action: Optional[str] = None
    
    # Objects
    objects: List[str] = field(default_factory=list)
    important_objects: List[str] = field(default_factory=list)
    
    # Additional
    mood: Optional[str] = None
    summary: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "scene": self.scene_number,
            "time": {
                "start": round(self.start_time, 2),
                "end": round(self.end_time, 2),
                "duration": round(self.duration, 2)
            },
            "location": self.location,
            "is_indoor": self.is_indoor,
            "lighting": self.lighting,
            "people": [
                {"name": p.name, "speaking": p.is_speaking, "visible": p.face_visible}
                for p in self.people
            ],
            "dialogue": [
                {"speaker": d.speaker, "text": d.text, "start": d.start_time, "end": d.end_time}
                for d in self.dialogues
            ],
            "actions": self.actions,
            "dominant_action": self.dominant_action,
            "objects": self.objects,
            "mood": self.mood,
            "summary": self.summary
        }


@dataclass
class VideoAnalysis:
    """Complete video analysis result"""
    video_path: str
    duration: float
    scenes: List[SceneData]
    
    # Global info
    all_speakers: List[str] = field(default_factory=list)
    all_people: List[str] = field(default_factory=list)
    all_objects: List[str] = field(default_factory=list)
    all_actions: List[str] = field(default_factory=list)
    all_locations: List[str] = field(default_factory=list)
    
    # Full transcript
    full_transcript: str = ""
    
    # Metadata
    fps: float = 0
    resolution: Tuple[int, int] = (0, 0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "video": {
                "path": self.video_path,
                "duration": round(self.duration, 2),
                "fps": self.fps,
                "resolution": list(self.resolution)
            },
            "summary": {
                "total_scenes": len(self.scenes),
                "speakers": self.all_speakers,
                "people": self.all_people,
                "locations": self.all_locations,
                "key_objects": self.all_objects[:20],
                "key_actions": self.all_actions[:10]
            },
            "scenes": [s.to_dict() for s in self.scenes],
            "transcript": self.full_transcript
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    def save(self, path: str) -> None:
        """Save to JSON file"""
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
    
    def rebuild_transcript(self) -> None:
        """Rebuild full_transcript from current dialogue data (after speaker labels updated)"""
        lines = []
        seen_texts = set()  # avoid duplicates
        
        for scene in self.scenes:
            for dlg in scene.dialogues:
                # Create timestamp
                minutes = int(dlg.start_time // 60)
                seconds = int(dlg.start_time % 60)
                timestamp = f"[{minutes:02d}:{seconds:02d}]"
                
                speaker = dlg.speaker or 'Unknown'
                text = dlg.text or ''
                
                # Skip duplicates
                key = f"{timestamp}|{text}"
                if key in seen_texts:
                    continue
                seen_texts.add(key)
                
                lines.append(f"{timestamp} {speaker}: {text}")
        
        self.full_transcript = "\n".join(lines)


class MultimodalFusion:
    """
    Fuses data from multiple analysis modules into coherent scene descriptions
    """
    
    def __init__(self, config):
        self.config = config
        self.window_size = config.fusion.window_size
    
    def fuse(
        self,
        video_metadata: Dict[str, Any],
        scenes: List[Dict[str, Any]],
        transcription: Dict[str, Any],
        diarization: Optional[Dict[str, Any]],
        face_results: List[Dict[str, Any]],
        object_results: List[Dict[str, Any]],
        action_results: List[Dict[str, Any]],
        speaker_names: Optional[Dict[str, str]] = None
    ) -> VideoAnalysis:
        """
        Fuse all analysis results into structured output
        
        Args:
            video_metadata: Video file metadata
            scenes: Scene detection results
            transcription: Speech transcription
            diarization: Speaker diarization
            face_results: Face detection/recognition results
            object_results: Object detection results
            action_results: Action recognition results
            speaker_names: Optional mapping of speaker IDs to names
        
        Returns:
            VideoAnalysis with all data fused
        """
        logger.info("Fusing multimodal analysis results...")
        
        # Create scene data objects
        scene_data_list = []
        
        for scene in scenes:
            scene_data = self._process_scene(
                scene=scene,
                transcription=transcription,
                diarization=diarization,
                face_results=face_results,
                object_results=object_results,
                action_results=action_results,
                speaker_names=speaker_names
            )
            scene_data_list.append(scene_data)
        
        # Collect global information
        all_speakers = self._collect_speakers(transcription, diarization, speaker_names)
        all_people = self._collect_people(face_results)
        all_objects = self._collect_objects(object_results)
        all_actions = self._collect_actions(action_results)
        all_locations = self._collect_locations(scene_data_list)
        
        # Build full transcript
        full_transcript = self._build_transcript(transcription, diarization, speaker_names)
        
        # Create final result
        result = VideoAnalysis(
            video_path=video_metadata.get('path', ''),
            duration=video_metadata.get('duration', 0),
            scenes=scene_data_list,
            all_speakers=all_speakers,
            all_people=all_people,
            all_objects=all_objects,
            all_actions=all_actions,
            all_locations=all_locations,
            full_transcript=full_transcript,
            fps=video_metadata.get('fps', 0),
            resolution=(video_metadata.get('width', 0), video_metadata.get('height', 0))
        )
        
        logger.info(f"Fused {len(scene_data_list)} scenes with all modalities")
        
        return result
    
    def _process_scene(
        self,
        scene: Dict[str, Any],
        transcription: Dict[str, Any],
        diarization: Optional[Dict[str, Any]],
        face_results: List[Dict[str, Any]],
        object_results: List[Dict[str, Any]],
        action_results: List[Dict[str, Any]],
        speaker_names: Optional[Dict[str, str]]
    ) -> SceneData:
        """Process a single scene with all modalities"""
        start_time = scene.get('start_time', 0)
        end_time = scene.get('end_time', 0)
        
        # Get dialogues for this scene
        dialogues = self._get_dialogues_in_range(
            transcription, diarization, start_time, end_time, speaker_names
        )
        
        # Get faces in this scene
        people, face_count = self._get_faces_in_range(
            face_results, start_time, end_time
        )
        
        # Match speakers to visible faces
        people = self._match_speakers_to_faces(people, dialogues)
        
        # Get objects in this scene
        objects = self._get_objects_in_range(object_results, start_time, end_time)
        important_objects = self._filter_important_objects(objects)

        # Refine location using object cues to reduce false indoor/outdoor labels
        refined_location, refined_is_indoor = self._refine_location_with_objects(
            scene.get('environment'),
            scene.get('is_indoor'),
            objects
        )
        
        # Get actions in this scene
        actions, dominant_action = self._get_actions_in_range(
            action_results, start_time, end_time
        )
        
        # Determine lighting
        lighting = self._determine_lighting(scene)
        
        # Estimate mood
        mood = self._estimate_mood(dialogues, actions, scene)
        
        return SceneData(
            scene_number=scene.get('scene_number', 0),
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
            location=refined_location,
            is_indoor=refined_is_indoor,
            lighting=lighting,
            people=people,
            face_count=face_count,
            dialogues=dialogues,
            actions=actions,
            dominant_action=dominant_action,
            objects=list(objects.keys())[:15],
            important_objects=important_objects,
            mood=mood
        )

    def _refine_location_with_objects(
        self,
        location: Optional[str],
        is_indoor: Optional[bool],
        objects: Dict[str, int]
    ) -> Tuple[Optional[str], Optional[bool]]:
        """Refine location label using object evidence."""
        if not objects:
            return location, is_indoor

        indoor_objects = {
            'laptop', 'tv', 'remote', 'keyboard', 'mouse', 'book', 'cell phone',
            'chair', 'couch', 'bed', 'dining table', 'cup', 'bottle'
        }
        outdoor_objects = {
            'car', 'bus', 'truck', 'motorcycle', 'bicycle', 'traffic light',
            'stop sign', 'bench', 'fire hydrant'
        }

        indoor_score = sum(count for obj, count in objects.items() if obj in indoor_objects)
        outdoor_score = sum(count for obj, count in objects.items() if obj in outdoor_objects)

        current = (location or '').lower()

        # Strong indoor evidence should override weak outdoor visual classification
        if indoor_score >= max(3, outdoor_score + 2):
            if 'interview' in current:
                return 'indoor interview setting', True
            if 'office' in current or 'studio' in current:
                return 'indoor office or studio', True
            return location or 'indoor room', True

        if outdoor_score >= max(3, indoor_score + 2):
            return location or 'outdoor setting', False

        return location, is_indoor
    
    def _get_dialogues_in_range(
        self,
        transcription: Dict[str, Any],
        diarization: Optional[Dict[str, Any]],
        start_time: float,
        end_time: float,
        speaker_names: Optional[Dict[str, str]]
    ) -> List[DialogueTurn]:
        """Get dialogue turns within a time range"""
        dialogues = []
        
        segments = transcription.get('segments', [])
        diar_segments = diarization.get('segments', []) if diarization else []
        
        for seg in segments:
            seg_start = seg.get('start', 0)
            seg_end = seg.get('end', 0)
            
            # Check if segment overlaps with scene
            if seg_end < start_time or seg_start > end_time:
                continue
            
            text = seg.get('text', '').strip()
            if not text:
                continue
            
            # Find speaker
            speaker = seg.get('speaker', 'Unknown')
            if not speaker or speaker == 'Unknown':
                speaker = self._find_speaker_at_time(diar_segments, (seg_start + seg_end) / 2)
            
            # Apply name mapping
            if speaker_names and speaker in speaker_names:
                speaker = speaker_names[speaker]
            
            dialogue = DialogueTurn(
                speaker=speaker or "Unknown",
                text=text,
                start_time=seg_start,
                end_time=seg_end
            )
            dialogues.append(dialogue)
        
        return dialogues
    
    def _find_speaker_at_time(
        self,
        diar_segments: List[Dict[str, Any]],
        time: float
    ) -> Optional[str]:
        """Find the speaker at a specific time"""
        for seg in diar_segments:
            if seg.get('start', 0) <= time <= seg.get('end', 0):
                return seg.get('speaker')
        return None
    
    def _get_faces_in_range(
        self,
        face_results: List[Dict[str, Any]],
        start_time: float,
        end_time: float
    ) -> Tuple[List[PersonPresence], int]:
        """Get unique people seen in time range"""
        people_seen = {}
        total_faces = 0
        
        for result in face_results:
            timestamp = result.get('timestamp', 0)
            if timestamp < start_time or timestamp > end_time:
                continue
            
            # Count all faces
            detections = result.get('detections', [])
            total_faces += len(detections)
            
            # Track recognized people
            matches = result.get('matches', [])
            for match in matches:
                name = match.get('identity', 'Unknown')
                if name not in people_seen:
                    people_seen[name] = PersonPresence(name=name, face_visible=True)
        
        # Add unknown people if we have more detections than matches
        if total_faces > len(people_seen):
            unknown_count = total_faces - len(people_seen)
            for i in range(unknown_count):
                unknown_name = f"Person_{i+1}"
                if unknown_name not in people_seen:
                    people_seen[unknown_name] = PersonPresence(name=unknown_name, face_visible=True)
        
        return list(people_seen.values()), total_faces
    
    def _match_speakers_to_faces(
        self,
        people: List[PersonPresence],
        dialogues: List[DialogueTurn]
    ) -> List[PersonPresence]:
        """Mark people as speaking if they have dialogue"""
        speakers = set(d.speaker for d in dialogues)
        
        for person in people:
            if person.name in speakers:
                person.is_speaking = True
        
        # Add speakers who weren't seen
        seen_names = {p.name for p in people}
        for speaker in speakers:
            if speaker not in seen_names and speaker != "Unknown":
                people.append(PersonPresence(
                    name=speaker,
                    is_speaking=True,
                    face_visible=False
                ))
        
        return people
    
    def _get_objects_in_range(
        self,
        object_results: List[Dict[str, Any]],
        start_time: float,
        end_time: float
    ) -> Dict[str, int]:
        """Get objects detected in time range with counts"""
        objects = defaultdict(int)
        
        for result in object_results:
            timestamp = result.get('timestamp', 0)
            if timestamp < start_time or timestamp > end_time:
                continue
            
            for det in result.get('detections', []):
                objects[det.get('class_name', 'unknown')] += 1
        
        # Sort by count
        return dict(sorted(objects.items(), key=lambda x: x[1], reverse=True))
    
    def _filter_important_objects(
        self,
        objects: Dict[str, int]
    ) -> List[str]:
        """Filter to important/notable objects"""
        important_classes = {
            'person', 'car', 'gun', 'knife', 'phone', 'laptop', 'book',
            'bottle', 'chair', 'table', 'dog', 'cat', 'bicycle', 'motorcycle'
        }
        
        important = []
        for obj, count in objects.items():
            if obj in important_classes or count > 3:
                important.append(obj)
        
        return important[:10]
    
    def _get_actions_in_range(
        self,
        action_results: List[Dict[str, Any]],
        start_time: float,
        end_time: float
    ) -> Tuple[List[str], Optional[str]]:
        """Get actions detected in time range"""
        actions = defaultdict(float)  # action -> max confidence
        
        for result in action_results:
            clip_start = result.get('start_time', 0)
            clip_end = result.get('end_time', 0)
            
            # Check overlap
            if clip_end < start_time or clip_start > end_time:
                continue
            
            for pred in result.get('predictions', []):
                action = pred.get('action', '')
                conf = pred.get('confidence', 0)
                if action and conf > actions[action]:
                    actions[action] = conf
        
        # Sort by confidence
        sorted_actions = sorted(actions.items(), key=lambda x: x[1], reverse=True)
        action_list = [a for a, _ in sorted_actions]
        dominant = action_list[0] if action_list else None
        
        return action_list[:5], dominant
    
    def _determine_lighting(self, scene: Dict[str, Any]) -> str:
        """Determine lighting from scene data"""
        brightness = scene.get('brightness', 128)
        
        if brightness > 180:
            return "bright"
        elif brightness > 100:
            return "normal"
        elif brightness > 50:
            return "dim"
        else:
            return "dark"
    
    def _estimate_mood(
        self,
        dialogues: List[DialogueTurn],
        actions: List[str],
        scene: Dict[str, Any]
    ) -> str:
        """Estimate the mood of a scene"""
        # Simple rule-based mood estimation
        negative_actions = {'fighting', 'crying', 'arguing', 'shouting', 'punching'}
        positive_actions = {'laughing', 'dancing', 'celebrating', 'hugging', 'kissing'}
        tense_actions = {'running', 'shooting', 'falling', 'fighting'}
        
        action_set = set(actions)
        
        if action_set & tense_actions:
            return "tense"
        elif action_set & negative_actions:
            return "dramatic"
        elif action_set & positive_actions:
            return "upbeat"
        
        # Check lighting
        brightness = scene.get('brightness', 128)
        if brightness < 80:
            return "moody"
        
        return "neutral"
    
    def _collect_speakers(
        self,
        transcription: Dict[str, Any],
        diarization: Optional[Dict[str, Any]],
        speaker_names: Optional[Dict[str, str]]
    ) -> List[str]:
        """Collect all unique speakers"""
        speakers = set()
        
        # From transcription
        for seg in transcription.get('segments', []):
            if seg.get('speaker'):
                speakers.add(seg['speaker'])
        
        # From diarization
        if diarization:
            for seg in diarization.get('segments', []):
                if seg.get('speaker'):
                    speakers.add(seg['speaker'])
        
        # Apply name mapping
        if speaker_names:
            speakers = {speaker_names.get(s, s) for s in speakers}
        
        speakers.discard('Unknown')
        return sorted(list(speakers))
    
    def _collect_people(self, face_results: List[Dict[str, Any]]) -> List[str]:
        """Collect all recognized people"""
        people = set()
        
        for result in face_results:
            for match in result.get('matches', []):
                if match.get('identity'):
                    people.add(match['identity'])
        
        return sorted(list(people))
    
    def _collect_objects(self, object_results: List[Dict[str, Any]]) -> List[str]:
        """Collect most common objects"""
        objects = defaultdict(int)
        
        for result in object_results:
            for det in result.get('detections', []):
                objects[det.get('class_name', '')] += 1
        
        # Sort by frequency
        sorted_objects = sorted(objects.items(), key=lambda x: x[1], reverse=True)
        return [obj for obj, _ in sorted_objects if obj][:30]
    
    def _collect_actions(self, action_results: List[Dict[str, Any]]) -> List[str]:
        """Collect all detected actions"""
        actions = set()
        
        for result in action_results:
            for pred in result.get('predictions', []):
                if pred.get('action'):
                    actions.add(pred['action'])
        
        return sorted(list(actions))
    
    def _collect_locations(self, scenes: List[SceneData]) -> List[str]:
        """Collect unique locations"""
        locations = set()
        
        for scene in scenes:
            if scene.location:
                locations.add(scene.location)
        
        return sorted(list(locations))
    
    def _build_transcript(
        self,
        transcription: Dict[str, Any],
        diarization: Optional[Dict[str, Any]],
        speaker_names: Optional[Dict[str, str]]
    ) -> str:
        """Build full transcript with speaker labels"""
        lines = []
        diar_segments = diarization.get('segments', []) if diarization else []
        
        for seg in transcription.get('segments', []):
            text = seg.get('text', '').strip()
            if not text:
                continue
            
            # Get speaker
            speaker = seg.get('speaker')
            if not speaker:
                mid_time = (seg.get('start', 0) + seg.get('end', 0)) / 2
                speaker = self._find_speaker_at_time(diar_segments, mid_time) or "Unknown"
            
            # Apply name mapping
            if speaker_names and speaker in speaker_names:
                speaker = speaker_names[speaker]
            
            # Format timestamp
            start = seg.get('start', 0)
            timestamp = f"[{int(start//60):02d}:{int(start%60):02d}]"
            
            lines.append(f"{timestamp} {speaker}: {text}")
        
        return "\n".join(lines)
