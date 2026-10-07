"""
LLM Text Generation Module
Generates narrative text from structured scene data using FREE Hugging Face models
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class GeneratedText:
    """Generated text output"""
    scene_number: int
    narrative: str
    summary: str
    dialogue_enhanced: str
    tokens_used: int = 0


class TextGenerator:
    """
    LLM-based text generation for cinematic descriptions
    Uses FREE Hugging Face models (runs locally, no API keys needed)
    Models: Flan-T5, BLIP for vision
    """
    
    DEFAULT_PROMPTS = {
        "story_generation": """Convert this scene data into cinematic prose. Describe setting, characters, actions, and dialogue vividly.

Scene Data:
{scene_data}

Narrative:""",

        "dialogue_enhancement": """Clean up and format this dialogue as a screenplay with speaker names and emotions.

Dialogue:
{dialogue}

Enhanced:""",

        "scene_summary": """Summarize this scene in 2 sentences focusing on key actions and characters.

Scene:
{scene_data}

Summary:""",

        "full_narrative": """Convert this video analysis into engaging narrative prose describing scenes cinematically.

Analysis:
{video_data}

Narrative:"""
    }
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.vision_model = None
        self.vision_processor = None
        self.device = "cuda" if self._check_cuda() else "cpu"
        self._model_loaded = False
        self._vision_loaded = False
        self._load_prompts()
        logger.info(f"TextGenerator initialized (device: {self.device})")
    
    def _check_cuda(self) -> bool:
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def _load_model(self) -> None:
        """Lazy load the text generation model (Flan-T5)"""
        if self._model_loaded:
            return
        
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            import torch
            
            model_name = "google/flan-t5-base"  # FREE, ~1GB, good quality
            logger.info(f"Loading {model_name}...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self._model_loaded = True
            logger.info(f"✓ Text model loaded on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load text model: {e}")
            self._model_loaded = False
    
    def _load_vision_model(self) -> None:
        """Lazy load vision model (BLIP) for image descriptions"""
        if self._vision_loaded:
            return
        
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            import torch
            
            model_name = "Salesforce/blip-image-captioning-base"  # FREE, ~1GB
            logger.info(f"Loading vision model {model_name}...")
            
            self.vision_processor = BlipProcessor.from_pretrained(model_name)
            self.vision_model = BlipForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            self.vision_model = self.vision_model.to(self.device)
            
            self._vision_loaded = True
            logger.info(f"✓ Vision model loaded on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load vision model: {e}")
            self._vision_loaded = False
    
    def _load_prompts(self) -> None:
        """Load prompts from config or use defaults"""
        self.prompts = self.DEFAULT_PROMPTS.copy()
        
        if hasattr(self.config, 'llm') and hasattr(self.config.llm, 'prompts') and self.config.llm.prompts:
            self.prompts.update(self.config.llm.prompts)
    
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Generate text using FREE Flan-T5 model
        
        Args:
            prompt: The prompt to send
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        
        Returns:
            Generated text
        """
        self._load_model()
        
        if not self._model_loaded:
            return self._fallback_generation(prompt)
        
        max_tokens = max_tokens or 512
        temperature = temperature if temperature is not None else 0.7
        
        try:
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                max_length=1024, 
                truncation=True
            ).to(self.device)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.9,
                num_return_sequences=1
            )
            
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return result.strip()
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return self._fallback_generation(prompt)
    
    def _fallback_generation(self, prompt: str) -> str:
        """Fallback when model not available - template-based generation"""
        # Extract key info from prompt and create basic narrative
        lines = prompt.split('\n')
        
        narrative_parts = []
        for line in lines:
            line = line.strip()
            if 'Scene' in line and 'seconds' in line:
                narrative_parts.append(f"The scene unfolds. {line}")
            elif 'Location:' in line:
                loc = line.replace('Location:', '').strip()
                narrative_parts.append(f"We are in {loc}.")
            elif 'People present:' in line:
                people = line.replace('People present:', '').strip()
                narrative_parts.append(f"Present are {people}.")
            elif line.startswith('"') or '":' in line:
                narrative_parts.append(f"Dialogue: {line}")
            elif 'objects:' in line.lower():
                objs = line.split(':')[-1].strip()
                narrative_parts.append(f"Visible in the scene: {objs}.")
        
        if narrative_parts:
            return " ".join(narrative_parts)
        return "[Scene description based on visual analysis]"
    
    def describe_frame(self, image_path: str) -> str:
        """Describe a video frame using BLIP (FREE vision model)"""
        self._load_vision_model()
        
        if not self._vision_loaded:
            return "[Image description unavailable]"
        
        try:
            from PIL import Image
            
            image = Image.open(image_path).convert('RGB')
            
            inputs = self.vision_processor(image, return_tensors="pt").to(self.device)
            
            outputs = self.vision_model.generate(
                **inputs,
                max_new_tokens=100,
                num_beams=3
            )
            
            caption = self.vision_processor.decode(outputs[0], skip_special_tokens=True)
            return caption.strip()
            
        except Exception as e:
            logger.error(f"Frame description failed: {e}")
            return "[Image description failed]"
    
    def describe_frames_batch(self, image_paths: List[str]) -> List[str]:
        """Describe multiple frames efficiently"""
        self._load_vision_model()
        
        descriptions = []
        for path in image_paths:
            desc = self.describe_frame(path)
            descriptions.append(desc)
        
        return descriptions
    
    def generate_scene_narrative(
        self,
        scene_data: Dict[str, Any]
    ) -> str:
        """
        Generate narrative text for a single scene
        
        Args:
            scene_data: Scene data dictionary
        
        Returns:
            Narrative prose
        """
        # Format scene data for prompt
        formatted_data = self._format_scene_for_prompt(scene_data)
        
        prompt = self.prompts["story_generation"].format(scene_data=formatted_data)
        return self.generate(prompt)
    
    def generate_scene_summary(
        self,
        scene_data: Dict[str, Any]
    ) -> str:
        """Generate a brief summary of a scene"""
        formatted_data = self._format_scene_for_prompt(scene_data)
        prompt = self.prompts["scene_summary"].format(scene_data=formatted_data)
        return self.generate(prompt, max_tokens=200)
    
    def enhance_dialogue(
        self,
        dialogue: List[Dict[str, str]]
    ) -> str:
        """
        Enhance dialogue with emotional directions
        
        Args:
            dialogue: List of {speaker, text} dicts
        
        Returns:
            Enhanced screenplay-style dialogue
        """
        # Format dialogue
        dialogue_text = "\n".join([
            f"{d.get('speaker', 'Unknown')}: {d.get('text', '')}"
            for d in dialogue
        ])
        
        prompt = self.prompts["dialogue_enhancement"].format(dialogue=dialogue_text)
        return self.generate(prompt)
    
    def generate_full_narrative(
        self,
        video_analysis: Dict[str, Any]
    ) -> str:
        """
        Generate complete narrative for entire video
        
        Args:
            video_analysis: Complete video analysis data
        
        Returns:
            Full narrative text
        """
        # Summarize for prompt (avoid token limits)
        summary = self._summarize_for_prompt(video_analysis)
        
        prompt = self.prompts["full_narrative"].format(video_data=summary)
        result = self.generate(prompt, max_tokens=4000)

        # Remove duplicate paragraphs introduced by generation
        try:
            result = self._dedupe_paragraphs(result)
        except Exception:
            pass

        return result
    
    def _format_scene_for_prompt(self, scene: Dict[str, Any]) -> str:
        """Format scene data for LLM prompt"""
        lines = []
        
        # Time info
        time_info = scene.get('time', {})
        lines.append(f"Scene {scene.get('scene', 0)} ({time_info.get('duration', 0):.1f} seconds)")
        
        # Location
        if scene.get('location'):
            indoor_outdoor = "indoor" if scene.get('is_indoor') else "outdoor"
            lines.append(f"Location: {scene['location']} ({indoor_outdoor})")
        
        # Lighting
        if scene.get('lighting'):
            lines.append(f"Lighting: {scene['lighting']}")
        
        # People
        people = scene.get('people', [])
        if people:
            people_str = ", ".join([
                f"{p['name']}" + (" (speaking)" if p.get('speaking') else "")
                for p in people
            ])
            lines.append(f"People present: {people_str}")
        
        # Dialogue
        dialogues = scene.get('dialogue', [])
        if dialogues:
            lines.append("\nDialogue:")
            for d in dialogues:
                lines.append(f"  {d.get('speaker', 'Unknown')}: \"{d.get('text', '')}\"")
        
        # Actions
        if scene.get('dominant_action'):
            lines.append(f"\nMain action: {scene['dominant_action']}")
        
        actions = scene.get('actions', [])
        if actions:
            lines.append(f"Other actions: {', '.join(actions[:5])}")
        
        # Objects
        objects = scene.get('objects', [])
        if objects:
            lines.append(f"\nVisible objects: {', '.join(objects[:10])}")
        
        # Mood
        if scene.get('mood'):
            lines.append(f"\nMood: {scene['mood']}")
        
        return "\n".join(lines)
    
    def _summarize_for_prompt(self, video: Dict[str, Any]) -> str:
        """Summarize video analysis for prompt"""
        lines = []
        
        # Video info
        video_info = video.get('video', {})
        lines.append(f"Video duration: {video_info.get('duration', 0):.1f} seconds")
        
        # Summary
        summary = video.get('summary', {})
        lines.append(f"Total scenes: {summary.get('total_scenes', 0)}")
        lines.append(f"Characters: {', '.join(summary.get('people', []))}")
        lines.append(f"Locations: {', '.join(summary.get('locations', []))}")
        
        # Scene summaries
        lines.append("\nScenes:")
        for scene in video.get('scenes', [])[:10]:  # Limit scenes
            scene_summary = self._format_scene_brief(scene)
            lines.append(scene_summary)
        
        # Transcript excerpt
        transcript = video.get('transcript', '')
        if transcript:
            lines.append(f"\nTranscript excerpt:\n{transcript[:2000]}...")
        
        return "\n".join(lines)

    def _assign_fallback_speakers(self, scene: Dict[str, Any]) -> None:
        """Ensure dialogue entries have speaker labels; assign fallbacks and remove duplicate lines."""
        dialogues = scene.get('dialogue', []) or []
        if not dialogues:
            return

        # Remove exact duplicate dialogue texts (case-insensitive)
        seen_texts = set()
        cleaned = []
        for i, d in enumerate(dialogues):
            text = (d.get('text') or '').strip()
            if not text:
                continue
            key = text.lower()
            if key in seen_texts:
                continue
            seen_texts.add(key)

            # Assign fallback speaker if missing
            if not d.get('speaker'):
                d['speaker'] = f"Speaker {len(cleaned) + 1}"

            cleaned.append(d)

        scene['dialogue'] = cleaned

    def _dedupe_paragraphs(self, text: str) -> str:
        """Remove duplicate paragraphs while preserving order."""
        if not text:
            return text

        parts = [p.strip() for p in text.split('\n\n') if p.strip()]
        seen = set()
        out = []
        for p in parts:
            key = ' '.join(p.split()).lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(p)

        return '\n\n'.join(out)
    
    def _format_scene_brief(self, scene: Dict[str, Any]) -> str:
        """Format scene as brief summary"""
        parts = [f"Scene {scene.get('scene', 0)}:"]
        
        if scene.get('location'):
            parts.append(scene['location'])
        
        people = scene.get('people', [])
        if people:
            names = [p['name'] for p in people[:3]]
            parts.append(f"with {', '.join(names)}")
        
        if scene.get('dominant_action'):
            parts.append(f"- {scene['dominant_action']}")
        
        return " ".join(parts)
    
    def process_all_scenes(
        self,
        scenes: List[Dict[str, Any]],
        generate_summary: bool = True
    ) -> List[GeneratedText]:
        """
        Process all scenes and generate text
        
        Args:
            scenes: List of scene data dicts
            generate_summary: Whether to also generate summaries
        
        Returns:
            List of GeneratedText for each scene
        """
        from tqdm import tqdm
        
        results = []

        for scene in tqdm(scenes, desc="Generating narratives"):
            # Assign fallback speaker labels and remove duplicate dialogue lines
            try:
                self._assign_fallback_speakers(scene)
            except Exception:
                logger.debug("Failed to assign fallback speakers for scene %s", scene.get('scene'))

            narrative = self.generate_scene_narrative(scene)

            summary = ""
            if generate_summary:
                summary = self.generate_scene_summary(scene)

            # Enhance dialogue if present
            dialogue_enhanced = ""
            if scene.get('dialogue'):
                dialogue_enhanced = self.enhance_dialogue(scene['dialogue'])

            result = GeneratedText(
                scene_number=scene.get('scene', 0),
                narrative=narrative,
                summary=summary,
                dialogue_enhanced=dialogue_enhanced
            )
            results.append(result)

        # Deduplicate narrative texts across scenes (preserve first occurrence)
        unique = []
        seen = set()
        for r in results:
            key = ' '.join((r.narrative or '').split()).lower()
            if not key:
                unique.append(r)
                continue
            if key in seen:
                logger.info(f"Removed duplicate narrative for scene {r.scene_number}")
                continue
            seen.add(key)
            unique.append(r)

        return unique
    
    def generate_screenplay_format(
        self,
        video_analysis: Dict[str, Any]
    ) -> str:
        """Generate output in screenplay format"""
        lines = []
        
        # Title
        lines.append("=" * 60)
        lines.append("SCREENPLAY")
        lines.append("=" * 60)
        lines.append("")
        
        for scene in video_analysis.get('scenes', []):
            # Scene header - improved location formatting
            location_value = scene.get('location') or 'UNKNOWN LOCATION'
            # Make location more readable
            location = str(location_value).upper().replace('_', ' ')
            indoor = "INT." if scene.get('is_indoor') else "EXT."
            
            # Determine time of day based on lighting
            lighting = scene.get('lighting', '')
            if lighting == 'bright':
                time_of_day = "DAY"
            elif lighting in ['dark', 'dim']:
                time_of_day = "NIGHT"
            else:
                time_of_day = "CONTINUOUS"
            
            lines.append(f"{indoor} {location} - {time_of_day}")
            lines.append("")
            
            # Description - create more cinematic description
            objects = scene.get('objects', [])[:5]
            lighting_desc = ""
            if lighting == 'dark':
                lighting_desc = "In the dimly lit space, "
            elif lighting == 'dim':
                lighting_desc = "Under soft lighting, "
            elif lighting == 'bright':
                lighting_desc = "In the well-lit room, "
            
            if objects:
                # Filter out 'person' for cleaner description
                obj_list = [o for o in objects if o.lower() != 'person'][:4]
                if obj_list:
                    lines.append(f"{lighting_desc}we see {', '.join(obj_list)}.")
                else:
                    lines.append(f"{lighting_desc}the scene unfolds.")
            
            # People - limit to reasonable count and simplify
            people = scene.get('people', [])
            if people:
                # Count unique people, max 5 for display
                unique_count = min(len(people), 5)
                if unique_count == 1:
                    lines.append("A single figure is present.")
                elif unique_count <= 3:
                    lines.append(f"{unique_count} people are present in the scene.")
                else:
                    lines.append("Several people are present.")
            
            # Action - make it more descriptive
            action = scene.get('dominant_action')
            if action and action != 'stationary':
                action_desc = action.replace('_', ' ')
                lines.append(f"The dominant activity is {action_desc}.")
            
            lines.append("")
            
            # Dialogue - deduplicate and format nicely
            seen_dialogue = set()
            for d in scene.get('dialogue', []):
                speaker = str(d.get('speaker') or 'SPEAKER').upper()
                text = str(d.get('text') or '')
                
                # Skip duplicates
                if text in seen_dialogue:
                    continue
                seen_dialogue.add(text)
                
                lines.append(f"                    {speaker}")
                lines.append(f"          {text}")
                lines.append("")
            
            lines.append("-" * 40)
            lines.append("")
        
        return "\n".join(lines)
