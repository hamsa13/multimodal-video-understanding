"""
Detailed Summary Generator
Generates comprehensive video analysis summaries from JSON and screenplay data
Uses LLM for intelligent combined understanding of video content
"""

import json
import logging
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


class LLMSummarizer:
    """Uses LLM to generate intelligent combined summaries"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self._loaded = False
    
    def _load_model(self):
        """Load the LLM model for summarization"""
        if self._loaded:
            return True
        
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Use Flan-T5 Large for better quality (still free)
            model_name = "google/flan-t5-base"
            logger.info(f"Loading LLM for summarization: {model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            self.model = self.model.to(self.device)
            self._loaded = True
            logger.info(f"LLM loaded on {self.device}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load LLM: {e}")
            return False
    
    def generate(self, prompt: str, max_length: int = 512) -> str:
        """Generate text using the LLM"""
        if not self._load_model():
            return ""
        
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=1024,
                truncation=True
            ).to(self.device)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                num_return_sequences=1
            )
            
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return result.strip()
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return ""
    
    def create_video_understanding(self, data: Dict, screenplay_text: str = "") -> str:
        """Create a comprehensive understanding of the video content"""
        # Extract all relevant information
        scenes = data.get('scenes', [])
        video_info = data.get('video', {})
        summary_info = data.get('summary', {})
        
        # Collect all dialogues
        all_dialogues = []
        for scene in scenes:
            for dlg in scene.get('dialogue', []):
                all_dialogues.append({
                    'speaker': dlg.get('speaker', 'Unknown'),
                    'text': dlg.get('text', ''),
                    'time': dlg.get('start', 0)
                })
        
        # Also extract dialogues from screenplay if available
        if screenplay_text and not all_dialogues:
            import re
            # Parse screenplay format: [Speaker]: text
            matches = re.findall(r'\[([^\]]+)\]:\s*(.+)', screenplay_text)
            for speaker, text in matches:
                all_dialogues.append({
                    'speaker': speaker,
                    'text': text.strip(),
                    'time': 0
                })
        
        # Collect all objects and their contexts
        object_contexts = {}
        for scene in scenes:
            location = scene.get('location', 'unknown')
            for obj in scene.get('objects', []):
                if obj not in object_contexts:
                    object_contexts[obj] = []
                object_contexts[obj].append(location)
        
        # Collect all actions
        all_actions = set()
        for scene in scenes:
            for action in scene.get('actions', []):
                all_actions.add(action)
            if scene.get('dominant_action'):
                all_actions.add(scene['dominant_action'])
        
        # Build the prompt for LLM
        prompt = self._build_understanding_prompt(
            video_info, scenes, all_dialogues, object_contexts, all_actions, summary_info
        )
        
        # Generate understanding
        understanding = self.generate(prompt, max_length=600)
        
        # If LLM output is too short, use fallback
        if not understanding or len(understanding.split()) < 30:
            understanding = self._fallback_understanding(
                video_info, scenes, all_dialogues, object_contexts, all_actions
            )
            
            # If LLM gave something, add it as an intro
            if understanding and len(understanding.split()) > 5:
                llm_output = understanding
                fallback = self._fallback_understanding(
                    video_info, scenes, all_dialogues, object_contexts, all_actions
                )
                understanding = f"{llm_output}\n\n{fallback}"
        
        return understanding
    
    def _build_understanding_prompt(
        self, 
        video_info: Dict,
        scenes: List[Dict],
        dialogues: List[Dict],
        objects: Dict,
        actions: set,
        summary_info: Dict
    ) -> str:
        """Build the prompt for video understanding"""
        duration = video_info.get('duration', 0)
        locations = summary_info.get('locations', [])
        
        # Format dialogue summary
        dialogue_summary = ""
        if dialogues:
            speakers = set(d['speaker'] for d in dialogues)
            dialogue_summary = f"Speakers: {', '.join(speakers)}. "
            dialogue_summary += "Key dialogue: "
            for dlg in dialogues[:5]:
                dialogue_summary += f'{dlg["speaker"]} says "{dlg["text"][:50]}..." '
        
        # Format object relationships
        object_summary = ""
        if objects:
            object_summary = "Visual elements: "
            for obj, locs in list(objects.items())[:8]:
                unique_locs = list(set(locs))
                object_summary += f"{obj} (seen in {unique_locs[0]}), "
        
        # Format scene flow
        scene_flow = ""
        for i, scene in enumerate(scenes[:5]):
            loc = scene.get('location', 'unknown')
            action = scene.get('dominant_action', 'activity')
            scene_flow += f"Scene {i+1}: {loc} - {action}. "
        
        prompt = f"""Analyze this video and write a comprehensive summary that combines visual content, audio, and context:

Video Duration: {duration:.1f} seconds
Locations: {', '.join(locations) if locations else 'Various'}
Scene Flow: {scene_flow}
{object_summary}
{dialogue_summary}
Actions observed: {', '.join(list(actions)[:8]) if actions else 'general activity'}

Write a cohesive 3-4 paragraph summary describing:
1. What the video is about and its main message
2. The visual setting and key objects/people
3. The dialogue and what speakers are communicating
4. The overall narrative and emotional tone

Summary:"""
        
        return prompt
    
    def _fallback_understanding(
        self,
        video_info: Dict,
        scenes: List[Dict],
        dialogues: List[Dict],
        objects: Dict,
        actions: set
    ) -> str:
        """Fallback understanding when LLM is not available"""
        parts = []
        
        duration = video_info.get('duration', 0)
        parts.append(f"This {duration:.1f}-second video contains {len(scenes)} distinct scene(s).")
        
        # Describe locations
        locations = set()
        for scene in scenes:
            if scene.get('location'):
                locations.add(scene['location'])
        if locations:
            parts.append(f"The video takes place in: {', '.join(locations)}.")
        
        # Describe speakers and dialogue
        if dialogues:
            speakers = set(d['speaker'] for d in dialogues)
            parts.append(f"The video features {len(speakers)} speaker(s): {', '.join(speakers)}.")
            
            # Summarize what they're talking about
            all_text = " ".join(d['text'] for d in dialogues)
            word_count = len(all_text.split())
            parts.append(f"There are {len(dialogues)} dialogue exchanges totaling approximately {word_count} words.")
            
            # Include key quotes
            if dialogues:
                parts.append("\nKey dialogue excerpts:")
                for dlg in dialogues[:4]:
                    parts.append(f'  • {dlg["speaker"]}: "{dlg["text"]}"')
        
        # Describe objects
        if objects:
            obj_list = list(objects.keys())[:10]
            parts.append(f"\nVisible objects and elements include: {', '.join(obj_list)}.")
        
        # Describe actions
        if actions:
            parts.append(f"Observed activities: {', '.join(list(actions)[:6])}.")
        
        return "\n".join(parts)
    
    def create_entity_relationship_summary(self, data: Dict) -> str:
        """Analyze relationships between entities in the video"""
        scenes = data.get('scenes', [])
        
        # Track entity co-occurrences
        entity_relations = []
        speakers_locations = {}
        objects_by_location = {}
        
        for scene in scenes:
            location = scene.get('location', 'scene')
            people = [p.get('name', 'person') for p in scene.get('people', [])]
            objects = scene.get('objects', [])
            speakers = [d.get('speaker', '') for d in scene.get('dialogue', [])]
            
            # Track objects by location
            if location not in objects_by_location:
                objects_by_location[location] = set()
            objects_by_location[location].update(objects)
            
            # People in location
            for person in people[:3]:
                entity_relations.append(f"{person} appears in {location}")
            
            # Speakers and their context  
            for speaker in set(speakers):
                if speaker:
                    if speaker not in speakers_locations:
                        speakers_locations[speaker] = set()
                    speakers_locations[speaker].add(location)
                    entity_relations.append(f"{speaker} speaks in {location}")
            
            # Objects in scene with people
            if people and objects:
                entity_relations.append(f"{people[0]} is near {', '.join(objects[:3])}")
        
        if not entity_relations:
            return "No significant entity relationships detected."
        
        # Use LLM to summarize relationships
        prompt = f"""Summarize these entity relationships from a video in natural language:

Relationships:
{chr(10).join(entity_relations[:15])}

Write a brief paragraph describing how people, objects, and locations relate to each other:"""
        
        summary = self.generate(prompt, max_length=200)
        
        # If LLM output is too short, use detailed fallback
        if not summary or len(summary.split()) < 15:
            summary_parts = ["Entity Relationships Detected:\n"]
            
            # Speaker locations
            if speakers_locations:
                summary_parts.append("Speakers and Locations:")
                for speaker, locs in speakers_locations.items():
                    summary_parts.append(f"  • {speaker} appears in: {', '.join(locs)}")
            
            # Objects by location
            if objects_by_location:
                summary_parts.append("\nObjects by Setting:")
                for loc, objs in objects_by_location.items():
                    summary_parts.append(f"  • {loc}: {', '.join(objs)}")
            
            # Key relationships
            summary_parts.append("\nKey Relationships:")
            for rel in entity_relations[:8]:
                summary_parts.append(f"  • {rel}")
            
            summary = "\n".join(summary_parts)
        
        return summary


class DetailedSummaryGenerator:
    """Generates detailed summaries from video analysis data"""
    
    def __init__(self, use_llm: bool = True):
        """
        Initialize the summary generator
        
        Args:
            use_llm: Whether to use LLM for enhanced understanding (default: True)
        """
        self.use_llm = use_llm
        self.llm_summarizer = None
        
        if use_llm:
            try:
                self.llm_summarizer = LLMSummarizer()
                logger.info("LLM Summarizer initialized successfully")
            except Exception as e:
                logger.warning(f"Could not initialize LLM Summarizer: {e}")
                self.llm_summarizer = None
    
    def generate_summary(
        self,
        analysis_json: Dict[str, Any],
        screenplay_text: str = "",
        metrics: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a comprehensive detailed summary
        
        Args:
            analysis_json: The analysis.json data as dict
            screenplay_text: The screenplay.txt content
            metrics: Optional metrics data
            
        Returns:
            Detailed summary as formatted text
        """
        sections = []
        
        # Header
        sections.append(self._generate_header(analysis_json))
        
        # LLM-Generated Combined Video Understanding (NEW)
        if self.llm_summarizer:
            sections.append(self._generate_llm_understanding(analysis_json, screenplay_text))
        
        # Video Overview
        sections.append(self._generate_video_overview(analysis_json))
        
        # Entity Relationships (using LLM)
        if self.llm_summarizer:
            sections.append(self._generate_entity_relationships(analysis_json))
        
        # Scene-by-Scene Analysis (from screenplay if available)
        sections.append(self._generate_scene_analysis(analysis_json, screenplay_text))
        
        # Character/Speaker Analysis
        sections.append(self._generate_speaker_analysis(analysis_json))
        
        # Dialogue Summary (extract from screenplay)
        sections.append(self._generate_dialogue_summary(analysis_json, screenplay_text))
        
        # Objects and Environment
        sections.append(self._generate_environment_analysis(analysis_json))
        
        # Actions and Activities
        sections.append(self._generate_action_summary(analysis_json))
        
        # Screenplay Section
        if screenplay_text:
            sections.append(self._include_screenplay(screenplay_text))
        
        # Technical Quality
        if metrics:
            sections.append(self._generate_quality_section(metrics))
        
        # Full Transcript
        sections.append(self._generate_transcript_section(analysis_json))
        
        # Conclusion
        sections.append(self._generate_conclusion(analysis_json, screenplay_text))
        
        return "\n\n".join(filter(None, sections))
    
    def _generate_llm_understanding(self, data: Dict, screenplay_text: str) -> str:
        """Generate LLM-based combined video understanding"""
        try:
            understanding = self.llm_summarizer.create_video_understanding(data, screenplay_text)
            
            return f"""{'='*50}
COMBINED VIDEO UNDERSTANDING (AI-Generated)
{'='*50}

{understanding}"""
        except Exception as e:
            logger.warning(f"Could not generate LLM understanding: {e}")
            return ""
    
    def _generate_entity_relationships(self, data: Dict) -> str:
        """Generate entity-object relationships using LLM"""
        try:
            relationships = self.llm_summarizer.create_entity_relationship_summary(data)
            
            return f"""{'='*50}
ENTITY & OBJECT RELATIONSHIPS
{'='*50}

{relationships}"""
        except Exception as e:
            logger.warning(f"Could not generate entity relationships: {e}")
            return ""
    
    def _get_video_info(self, data: Dict) -> Dict:
        """Extract video info from various possible structures"""
        # Try different possible keys
        if 'video' in data:
            video = data['video']
            return {
                'duration': video.get('duration', 0),
                'fps': video.get('fps', 0),
                'width': video.get('resolution', [0, 0])[1] if isinstance(video.get('resolution'), list) else video.get('width', 0),
                'height': video.get('resolution', [0, 0])[0] if isinstance(video.get('resolution'), list) else video.get('height', 0),
            }
        elif 'video_info' in data:
            return data['video_info']
        return {}
    
    def _get_scenes(self, data: Dict) -> List[Dict]:
        """Extract scenes from data"""
        return data.get('scenes', [])
    
    def _get_summary_info(self, data: Dict) -> Dict:
        """Extract summary info"""
        return data.get('summary', {})
    
    def _generate_header(self, data: Dict) -> str:
        """Generate header section"""
        video_info = self._get_video_info(data)
        duration = video_info.get('duration', 0)
        
        mins = int(duration // 60)
        secs = int(duration % 60)
        duration_str = f"{mins}:{secs:02d}" if mins > 0 else f"{secs:.1f} seconds"
        
        return f"""{'='*70}
COMPREHENSIVE VIDEO ANALYSIS SUMMARY
{'='*70}

Generated Analysis Report
Video Duration: {duration_str}
"""
    
    def _generate_video_overview(self, data: Dict) -> str:
        """Generate video overview section"""
        video_info = self._get_video_info(data)
        summary = self._get_summary_info(data)
        scenes = self._get_scenes(data)
        
        width = video_info.get('width', 0)
        height = video_info.get('height', 0)
        fps = video_info.get('fps', 0)
        duration = video_info.get('duration', 0)
        
        # Get speakers from scenes
        all_speakers = set()
        for scene in scenes:
            for dlg in scene.get('dialogue', []):
                speaker = dlg.get('speaker', '')
                if speaker:
                    all_speakers.add(speaker)
        
        all_objects = summary.get('key_objects', [])
        all_actions = summary.get('key_actions', [])
        locations = summary.get('locations', [])
        
        return f"""{'-'*50}
VIDEO OVERVIEW
{'-'*50}

Technical Specifications:
  • Resolution: {height}x{width}
  • Frame Rate: {fps:.2f} FPS
  • Duration: {duration:.2f} seconds

Content Summary:
  • Total Scenes: {len(scenes)}
  • Speakers Identified: {len(all_speakers)}
  • Unique Objects: {len(all_objects)}
  • Actions Detected: {len(all_actions)}
  • Locations: {', '.join(locations) if locations else 'N/A'}
  • Speakers: {', '.join(sorted(all_speakers)) if all_speakers else 'None identified'}
"""
    
    def _generate_scene_analysis(self, data: Dict, screenplay_text: str = "") -> str:
        """Generate scene-by-scene analysis"""
        scenes = self._get_scenes(data)
        
        if not scenes:
            return f"""{'-'*50}
SCENE ANALYSIS
{'-'*50}

No distinct scenes were detected in this video.
The entire video appears to be a continuous sequence.
"""
        
        lines = [f"{'-'*50}", "SCENE-BY-SCENE ANALYSIS", f"{'-'*50}", ""]
        
        for scene in scenes:
            scene_num = scene.get('scene', scene.get('scene_number', 0))
            time_info = scene.get('time', {})
            start = time_info.get('start', scene.get('start_time', 0))
            end = time_info.get('end', scene.get('end_time', 0))
            duration = time_info.get('duration', scene.get('duration', end - start))
            location = scene.get('location', scene.get('environment', 'Unknown'))
            is_indoor = scene.get('is_indoor')
            lighting = scene.get('lighting', 'Normal')
            mood = scene.get('mood', '')
            
            location_type = "Indoor" if is_indoor else "Outdoor" if is_indoor is False else "Unknown"
            
            lines.append(f"SCENE {scene_num}: [{self._format_time(start)} - {self._format_time(end)}]")
            lines.append(f"  Duration: {duration:.1f} seconds")
            lines.append(f"  Location: {location}")
            lines.append(f"  Setting: {location_type}")
            lines.append(f"  Lighting: {lighting}")
            if mood:
                lines.append(f"  Mood: {mood}")
            
            # Scene dialogues (key is 'dialogue' not 'dialogues')
            dialogues = scene.get('dialogue', scene.get('dialogues', []))
            if dialogues:
                lines.append(f"  Dialogue Turns: {len(dialogues)}")
                for dlg in dialogues:
                    speaker = dlg.get('speaker', 'Unknown')
                    text = dlg.get('text', '')
                    dlg_start = dlg.get('start', 0)
                    display_text = text[:80] + "..." if len(text) > 80 else text
                    lines.append(f"    [{self._format_time(dlg_start)}] {speaker}: \"{display_text}\"")
            
            # Scene people
            people = scene.get('people', [])
            if people:
                people_names = [p.get('name', 'Unknown') for p in people if isinstance(p, dict)]
                if people_names:
                    lines.append(f"  People Visible: {', '.join(people_names[:5])}")
            
            # Scene objects
            objects = scene.get('objects', [])
            if objects:
                obj_list = ', '.join(objects[:5])
                if len(objects) > 5:
                    obj_list += f", +{len(objects) - 5} more"
                lines.append(f"  Objects Present: {obj_list}")
            
            # Scene actions
            actions = scene.get('actions', [])
            dominant_action = scene.get('dominant_action', '')
            if actions:
                lines.append(f"  Activities: {', '.join(actions[:3])}")
            if dominant_action:
                lines.append(f"  Primary Action: {dominant_action}")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def _generate_speaker_analysis(self, data: Dict) -> str:
        """Generate speaker/character analysis"""
        scenes = self._get_scenes(data)
        
        # Count dialogue turns per speaker
        speaker_turns = {}
        speaker_words = {}
        
        for scene in scenes:
            for dlg in scene.get('dialogue', scene.get('dialogues', [])):
                speaker = dlg.get('speaker', 'Unknown')
                text = dlg.get('text', '')
                
                speaker_turns[speaker] = speaker_turns.get(speaker, 0) + 1
                speaker_words[speaker] = speaker_words.get(speaker, 0) + len(text.split())
        
        if not speaker_turns:
            return ""
        
        lines = [f"{'-'*50}", "SPEAKER ANALYSIS", f"{'-'*50}", ""]
        
        for speaker in sorted(speaker_turns.keys()):
            turns = speaker_turns.get(speaker, 0)
            words = speaker_words.get(speaker, 0)
            
            lines.append(f"{speaker}:")
            lines.append(f"  • Dialogue Turns: {turns}")
            lines.append(f"  • Word Count: {words}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _generate_dialogue_summary(self, data: Dict, screenplay_text: str = "") -> str:
        """Generate complete dialogue summary"""
        scenes = self._get_scenes(data)
        
        all_dialogues = []
        for scene in scenes:
            for dlg in scene.get('dialogue', scene.get('dialogues', [])):
                all_dialogues.append(dlg)
        
        if not all_dialogues:
            return f"""{'-'*50}
DIALOGUE SUMMARY
{'-'*50}

No dialogue was detected in this video.
This may be due to:
  • No speech present in the video
  • Audio quality issues
  • Background noise interference
"""
        
        lines = [f"{'-'*50}", "COMPLETE DIALOGUE TRANSCRIPT", f"{'-'*50}", ""]
        lines.append(f"Total Dialogue Turns: {len(all_dialogues)}")
        lines.append("")
        lines.append("Full Conversation:")
        lines.append("-" * 30)
        lines.append("")
        
        for i, dlg in enumerate(all_dialogues, 1):
            speaker = dlg.get('speaker', 'Unknown')
            text = dlg.get('text', '')
            start = dlg.get('start', dlg.get('start_time', 0))
            end = dlg.get('end', dlg.get('end_time', 0))
            
            lines.append(f"[{self._format_time(start)} - {self._format_time(end)}]")
            lines.append(f"  {speaker}:")
            # Word wrap the text
            wrapped = self._word_wrap(text, 55)
            for line in wrapped:
                lines.append(f"    \"{line}\"")
            lines.append("")
        
        return "\n".join(lines)
    
    def _generate_environment_analysis(self, data: Dict) -> str:
        """Generate environment and objects analysis"""
        summary = self._get_summary_info(data)
        scenes = self._get_scenes(data)
        
        all_objects = summary.get('key_objects', [])
        locations = summary.get('locations', [])
        
        # Collect environments from scenes if not in summary
        if not locations:
            for scene in scenes:
                location = scene.get('location', scene.get('environment', ''))
                if location and location not in locations:
                    locations.append(location)
        
        # Collect objects from scenes if not in summary
        if not all_objects:
            obj_set = set()
            for scene in scenes:
                for obj in scene.get('objects', []):
                    obj_set.add(obj)
            all_objects = list(obj_set)
        
        lines = [f"{'-'*50}", "ENVIRONMENT & OBJECTS", f"{'-'*50}", ""]
        
        if locations:
            lines.append("Settings/Locations:")
            for loc in locations:
                lines.append(f"  • {loc.title()}")
            lines.append("")
        
        if all_objects:
            lines.append(f"Objects Identified ({len(all_objects)} types):")
            for obj in sorted(all_objects):
                lines.append(f"  • {obj}")
        else:
            lines.append("No specific objects were detected.")
        
        return "\n".join(lines)
    
    def _generate_action_summary(self, data: Dict) -> str:
        """Generate action/activity summary"""
        summary = self._get_summary_info(data)
        scenes = self._get_scenes(data)
        
        all_actions = summary.get('key_actions', [])
        
        # Collect from scenes if not in summary
        if not all_actions:
            action_set = set()
            for scene in scenes:
                for action in scene.get('actions', []):
                    action_set.add(action)
                dominant = scene.get('dominant_action', '')
                if dominant:
                    action_set.add(dominant)
            all_actions = list(action_set)
        
        if not all_actions:
            return ""
        
        lines = [f"{'-'*50}", "ACTIONS & ACTIVITIES", f"{'-'*50}", ""]
        lines.append(f"Detected Activities ({len(all_actions)}):")
        
        for action in sorted(all_actions):
            lines.append(f"  • {action}")
        
        return "\n".join(lines)
    
    def _include_screenplay(self, screenplay_text: str) -> str:
        """Include the screenplay in the summary"""
        if not screenplay_text or not screenplay_text.strip():
            return ""
        
        lines = [f"{'-'*50}", "SCREENPLAY FORMAT", f"{'-'*50}", ""]
        lines.append(screenplay_text.strip())
        
        return "\n".join(lines)
    def _generate_quality_section(self, metrics: Dict) -> str:
        """Generate quality metrics section"""
        overall = metrics.get('overall_scores', {})
        audio = metrics.get('audio_metrics', {})
        video = metrics.get('video_metrics', {})
        
        confidence = overall.get('confidence_score', 0)
        quality = overall.get('data_quality_score', 0)
        
        lines = [f"{'-'*50}", "ANALYSIS QUALITY METRICS", f"{'-'*50}", ""]
        lines.append(f"Overall Confidence Score: {confidence:.1f}%")
        lines.append(f"Data Quality Score: {quality:.1f}%")
        lines.append("")
        lines.append("Audio Quality:")
        lines.append(f"  • Transcription Confidence: {audio.get('transcription_confidence', 0) * 100:.1f}%")
        lines.append(f"  • Speech Coverage: {audio.get('speech_percentage', 0):.1f}%")
        lines.append("")
        lines.append("Video Quality:")
        lines.append(f"  • Resolution Score: {video.get('quality_score', 0):.1f}%")
        
        return "\n".join(lines)
    
    def _generate_transcript_section(self, data: Dict) -> str:
        """Generate full transcript section"""
        transcript = data.get('full_transcript', '')
        
        if not transcript or transcript.strip() == '':
            return ""
        
        lines = [f"{'-'*50}", "FULL TRANSCRIPT", f"{'-'*50}", ""]
        lines.append(transcript)
        
        return "\n".join(lines)
    
    def _generate_conclusion(self, data: Dict, screenplay_text: str = "") -> str:
        """Generate conclusion section"""
        scenes = self._get_scenes(data)
        summary = self._get_summary_info(data)
        video_info = self._get_video_info(data)
        
        # Get speakers from scenes
        all_speakers = set()
        for scene in scenes:
            for dlg in scene.get('dialogue', scene.get('dialogues', [])):
                speaker = dlg.get('speaker', '')
                if speaker:
                    all_speakers.add(speaker)
        
        all_objects = summary.get('key_objects', [])
        all_actions = summary.get('key_actions', [])
        locations = summary.get('locations', [])
        
        # Count total dialogues
        total_dialogues = sum(len(s.get('dialogue', s.get('dialogues', []))) for s in scenes)
        
        # Determine video type
        video_type = "video content"
        if total_dialogues > 5 and len(all_speakers) >= 2:
            video_type = "conversation/dialogue"
        elif total_dialogues > 0 and len(all_speakers) == 1:
            video_type = "monologue/speech"
        elif total_dialogues > 0:
            video_type = "narrated content"
        elif len(all_objects) > 3:
            video_type = "visual sequence"
        
        duration = video_info.get('duration', 0)
        
        lines = [f"{'='*70}", "SUMMARY CONCLUSION", f"{'='*70}", ""]
        
        lines.append(f"This {video_type} ({duration:.1f} seconds) consists of:")
        lines.append(f"  • {len(scenes)} scene(s)")
        lines.append(f"  • {len(all_speakers)} speaker(s): {', '.join(sorted(all_speakers)) if all_speakers else 'None'}")
        lines.append(f"  • {total_dialogues} dialogue exchange(s)")
        lines.append(f"  • {len(all_objects)} object type(s) detected")
        lines.append(f"  • {len(all_actions)} activity type(s) recognized")
        
        if locations:
            lines.append(f"\nLocations featured: {', '.join(locations)}")
        
        if all_speakers:
            lines.append(f"\nKey participants: {', '.join(sorted(all_speakers))}")
        
        # Add a narrative summary based on the content
        if total_dialogues > 0:
            lines.append(f"\nThe video features spoken content with {total_dialogues} dialogue segments "
                        f"across {len(scenes)} scene(s).")
        
        lines.append("")
        lines.append(f"{'='*70}")
        lines.append("END OF DETAILED ANALYSIS REPORT")
        lines.append(f"{'='*70}")
        
        return "\n".join(lines)
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds as MM:SS"""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"
    
    def _word_wrap(self, text: str, width: int) -> List[str]:
        """Wrap text to specified width"""
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= width:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines if lines else ['']
    
    def generate_from_files(
        self,
        analysis_json_path: str,
        screenplay_path: str = "",
        metrics_path: str = "",
        output_path: str = ""
    ) -> str:
        """
        Generate summary from file paths
        
        Args:
            analysis_json_path: Path to analysis.json
            screenplay_path: Path to screenplay.txt (optional)
            metrics_path: Path to metrics.json (optional)
            output_path: Path to save summary (optional)
            
        Returns:
            Generated summary text
        """
        # Load analysis JSON
        with open(analysis_json_path, 'r', encoding='utf-8') as f:
            analysis_data = json.load(f)
        
        # Load screenplay if provided
        screenplay_text = ""
        if screenplay_path and Path(screenplay_path).exists():
            with open(screenplay_path, 'r', encoding='utf-8') as f:
                screenplay_text = f.read()
        
        # Load metrics if provided
        metrics_data = None
        if metrics_path and Path(metrics_path).exists():
            with open(metrics_path, 'r', encoding='utf-8') as f:
                metrics_data = json.load(f)
        
        # Generate summary
        summary = self.generate_summary(analysis_data, screenplay_text, metrics_data)
        
        # Save if output path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(summary)
            logger.info(f"Summary saved to {output_path}")
        
        return summary
