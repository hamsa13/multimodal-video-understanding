"""
Groq LLM Summary Generator
Uses Groq's llama-3.3-70b-versatile for high-quality video summaries
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)

# Default Groq API key - user should set GROQ_API_KEY environment variable
DEFAULT_MODEL = "llama-3.3-70b-versatile"


class GroqSummaryGenerator:
    """
    Generates detailed video summaries using Groq's LLama 3.3 70B model
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Groq client
        
        Args:
            api_key: Groq API key (or set GROQ_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.client = None
        self.model = DEFAULT_MODEL
        
        if self.api_key:
            self._init_client()
        else:
            logger.warning("No Groq API key provided. Set GROQ_API_KEY environment variable.")
    
    def _init_client(self):
        """Initialize the Groq client"""
        try:
            from groq import Groq
            self.client = Groq(api_key=self.api_key)
            logger.info(f"Groq client initialized with model: {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")
            self.client = None
    
    def generate_narrative(self, screenplay_text: str, analysis_data: Dict[str, Any]) -> str:
        """
        Generate a detailed narrative from screenplay and analysis data
        
        Args:
            screenplay_text: The screenplay.txt content
            analysis_data: The analysis.json data as dict
            
        Returns:
            Detailed narrative text
        """
        if not self.client:
            return self._fallback_narrative(screenplay_text, analysis_data)
        
        # Build comprehensive prompt
        prompt = self._build_narrative_prompt(screenplay_text, analysis_data)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are a professional video analyst and narrator. Your task is to create 
detailed, engaging narrative descriptions of videos based on screenplay and analysis data.
Write in a cinematic, descriptive style that brings the video to life for readers.
Include details about:
- Visual scenes and settings
- Character actions and movements  
- Dialogue and conversations
- Objects and their significance
- Mood and atmosphere
- Transitions between scenes"""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=4096,
                top_p=0.9
            )
            
            narrative = response.choices[0].message.content
            logger.info(f"Generated narrative with {len(narrative.split())} words")
            return narrative
            
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            return self._fallback_narrative(screenplay_text, analysis_data)
    
    def generate_detailed_summary(
        self,
        analysis_data: Dict[str, Any],
        screenplay_text: str = "",
        transcript_text: str = ""
    ) -> str:
        """
        Generate a comprehensive video summary combining all analysis data
        
        Args:
            analysis_data: The analysis.json data
            screenplay_text: The screenplay content
            transcript_text: The transcript content
            
        Returns:
            Detailed summary text
        """
        if not self.client:
            return self._fallback_summary(analysis_data, screenplay_text)
        
        prompt = self._build_summary_prompt(analysis_data, screenplay_text, transcript_text)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert video analyst. Create a comprehensive, detailed summary 
of the video that combines:

1. VISUAL ANALYSIS: Describe what is seen - people, objects, settings, actions, scene changes
2. AUDIO ANALYSIS: Analyze the dialogue, who speaks, what they say, tone and emotion
3. ENTITY RELATIONSHIPS: How do people and objects interact? What are the spatial/temporal relationships?
4. NARRATIVE FLOW: How does the story/content progress from start to end?
5. KEY MOMENTS: Highlight the most important or impactful moments
6. OVERALL THEME: What is the video about? What message does it convey?

Write in a professional, comprehensive style. Be specific with timestamps and details.
The summary should allow someone who hasn't seen the video to fully understand its content."""
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.5,
                max_tokens=4096,
                top_p=0.9
            )
            
            summary = response.choices[0].message.content
            logger.info(f"Generated detailed summary with {len(summary.split())} words")
            return summary
            
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            return self._fallback_summary(analysis_data, screenplay_text)
    
    def generate_entity_relationship_summary(self, analysis_data: Dict[str, Any]) -> str:
        """
        Generate summary focused on entity and object relationships
        
        Args:
            analysis_data: The analysis.json data
            
        Returns:
            Entity relationship summary
        """
        if not self.client:
            return self._extract_relationships(analysis_data)
        
        # Extract scene data for analysis
        scenes = analysis_data.get('scenes', [])
        summary_info = analysis_data.get('summary', {})
        
        scene_descriptions = []
        for scene in scenes:
            scene_num = scene.get('scene', scene.get('scene_number', 0))
            location = scene.get('location', 'Unknown')
            people = scene.get('people', [])
            objects = scene.get('objects', [])
            dialogue = scene.get('dialogue', [])
            actions = scene.get('actions', [])
            
            people_names = [p.get('name', 'Unknown') for p in people if isinstance(p, dict)]
            speakers = list(set([d.get('speaker', '') for d in dialogue if d.get('speaker')]))
            
            scene_descriptions.append(f"""
Scene {scene_num} ({location}):
- People present: {', '.join(people_names) if people_names else 'None'}
- Objects: {', '.join(objects) if objects else 'None'}
- Speakers: {', '.join(speakers) if speakers else 'None'}
- Actions: {', '.join(actions) if actions else 'None'}
- Dialogue count: {len(dialogue)}""")
        
        prompt = f"""Analyze the entity and object relationships in this video:

VIDEO DATA:
{chr(10).join(scene_descriptions)}

KEY OBJECTS: {', '.join(summary_info.get('key_objects', []))}
KEY ACTIONS: {', '.join(summary_info.get('key_actions', []))}
LOCATIONS: {', '.join(summary_info.get('locations', []))}

Please provide:
1. ENTITY IDENTIFICATION: Who are the main people/entities in the video?
2. OBJECT ANALYSIS: What objects are present and what is their significance?
3. SPATIAL RELATIONSHIPS: How are entities positioned relative to each other and objects?
4. TEMPORAL RELATIONSHIPS: How do relationships change across scenes?
5. INTERACTIONS: What interactions occur between entities? Between entities and objects?
6. ENTITY ROLES: What role does each entity play in the video?

Be specific and detailed."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at analyzing entity relationships in video content. Provide detailed, structured analysis."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.5,
                max_tokens=2048
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            return self._extract_relationships(analysis_data)
    
    def _build_narrative_prompt(self, screenplay: str, analysis: Dict) -> str:
        """Build the narrative generation prompt"""
        video_info = analysis.get('video', {})
        duration = video_info.get('duration', 0)
        
        scenes = analysis.get('scenes', [])
        
        # Extract all dialogue
        all_dialogue = []
        for scene in scenes:
            for dlg in scene.get('dialogue', []):
                speaker = dlg.get('speaker', 'Unknown')
                text = dlg.get('text', '')
                start = dlg.get('start', 0)
                all_dialogue.append(f"[{start:.1f}s] {speaker}: {text}")
        
        prompt = f"""Create a detailed narrative description of this video:

VIDEO DURATION: {duration:.1f} seconds

SCREENPLAY:
{screenplay[:3000]}

DIALOGUE TRANSCRIPT:
{chr(10).join(all_dialogue[:30])}

SCENE INFORMATION:
{json.dumps(scenes[:5], indent=2)[:2000]}

Write a vivid, engaging narrative that describes:
1. The visual setting and atmosphere
2. The people/characters and their actions
3. The dialogue and conversations
4. How scenes transition
5. The overall message or story

Write in present tense, cinematic style. Be detailed and descriptive."""

        return prompt
    
    def _build_summary_prompt(self, analysis: Dict, screenplay: str, transcript: str) -> str:
        """Build the comprehensive summary prompt"""
        video_info = analysis.get('video', {})
        duration = video_info.get('duration', 0)
        resolution = video_info.get('resolution', [0, 0])
        
        summary_info = analysis.get('summary', {})
        scenes = analysis.get('scenes', [])
        
        # Build scene summaries
        scene_summaries = []
        for scene in scenes:
            scene_num = scene.get('scene', 0)
            time_info = scene.get('time', {})
            start = time_info.get('start', 0)
            end = time_info.get('end', 0)
            location = scene.get('location', 'Unknown')
            is_indoor = scene.get('is_indoor', None)
            lighting = scene.get('lighting', 'normal')
            mood = scene.get('mood', '')
            objects = scene.get('objects', [])
            actions = scene.get('actions', [])
            dialogue = scene.get('dialogue', [])
            
            dialogue_text = []
            for dlg in dialogue:
                dialogue_text.append(f"{dlg.get('speaker', 'Unknown')}: \"{dlg.get('text', '')}\"")
            
            scene_summaries.append(f"""
SCENE {scene_num} [{start:.1f}s - {end:.1f}s]:
- Location: {location} ({'Indoor' if is_indoor else 'Outdoor'})
- Lighting: {lighting}, Mood: {mood}
- Objects: {', '.join(objects)}
- Actions: {', '.join(actions)}
- Dialogue:
  {chr(10).join(dialogue_text[:5])}""")
        
        prompt = f"""Create a COMPREHENSIVE DETAILED SUMMARY of this video:

=== VIDEO METADATA ===
Duration: {duration:.1f} seconds
Resolution: {resolution[0]}x{resolution[1]} 

=== CONTENT OVERVIEW ===
Total Scenes: {summary_info.get('total_scenes', len(scenes))}
Locations: {', '.join(summary_info.get('locations', []))}
Key Objects: {', '.join(summary_info.get('key_objects', []))}
Key Actions: {', '.join(summary_info.get('key_actions', []))}

=== SCENE-BY-SCENE BREAKDOWN ===
{chr(10).join(scene_summaries)}

=== SCREENPLAY FORMAT ===
{screenplay[:2500]}

=== TRANSCRIPT ===
{transcript[:1500] if transcript else 'See dialogue above'}

Please provide a DETAILED SUMMARY that includes:

1. **VIDEO OVERVIEW** (2-3 sentences describing what the video is about)

2. **VISUAL DESCRIPTION** 
   - Settings and locations
   - People and their appearance
   - Objects and their arrangement
   - Lighting and visual style

3. **AUDIO/DIALOGUE ANALYSIS**
   - Who speaks and what do they say?
   - Tone and emotional content of speech
   - Key quotes and their significance

4. **SCENE PROGRESSION**
   - How the video flows from scene to scene
   - Key transitions and their effect
   - Narrative arc

5. **ENTITY & OBJECT RELATIONSHIPS**
   - How people interact with each other
   - How people interact with objects
   - Spatial relationships

6. **KEY THEMES & MESSAGE**
   - What is the main message?
   - What themes are present?
   - Target audience and purpose

Be specific, use timestamps, and provide enough detail that someone could visualize the entire video."""

        return prompt
    
    def _fallback_narrative(self, screenplay: str, analysis: Dict) -> str:
        """Fallback narrative when Groq is unavailable"""
        scenes = analysis.get('scenes', [])
        video_info = analysis.get('video', {})
        
        lines = [
            "VIDEO NARRATIVE",
            "=" * 50,
            f"\nDuration: {video_info.get('duration', 0):.1f} seconds\n"
        ]
        
        for scene in scenes:
            scene_num = scene.get('scene', 0)
            location = scene.get('location', 'Unknown location')
            time_info = scene.get('time', {})
            
            lines.append(f"\n--- Scene {scene_num} ({time_info.get('start', 0):.1f}s - {time_info.get('end', 0):.1f}s) ---")
            lines.append(f"Location: {location}")
            
            for dlg in scene.get('dialogue', []):
                lines.append(f"  {dlg.get('speaker', 'Speaker')}: \"{dlg.get('text', '')}\"")
        
        return "\n".join(lines)
    
    def _fallback_summary(self, analysis: Dict, screenplay: str) -> str:
        """Fallback summary when Groq is unavailable"""
        return f"""VIDEO SUMMARY (Fallback Mode - No API Key)
{'=' * 50}

To get detailed AI-generated summaries, please set the GROQ_API_KEY environment variable.

Get your free API key at: https://console.groq.com/

Basic Analysis:
- Scenes: {len(analysis.get('scenes', []))}
- Locations: {', '.join(analysis.get('summary', {}).get('locations', ['Unknown']))}
- Objects: {', '.join(analysis.get('summary', {}).get('key_objects', ['None detected']))}

SCREENPLAY CONTENT:
{screenplay[:2000]}
"""
    
    def _extract_relationships(self, analysis: Dict) -> str:
        """Extract basic entity relationships without LLM"""
        scenes = analysis.get('scenes', [])
        
        lines = ["ENTITY RELATIONSHIPS (Basic Analysis)", "=" * 50, ""]
        
        all_people = set()
        all_objects = set()
        interactions = []
        
        for scene in scenes:
            scene_num = scene.get('scene', 0)
            people = scene.get('people', [])
            objects = scene.get('objects', [])
            dialogue = scene.get('dialogue', [])
            
            for p in people:
                if isinstance(p, dict):
                    all_people.add(p.get('name', 'Unknown'))
            
            all_objects.update(objects)
            
            speakers = [d.get('speaker') for d in dialogue if d.get('speaker')]
            if len(speakers) > 1:
                interactions.append(f"Scene {scene_num}: {' & '.join(set(speakers))} have dialogue")
        
        lines.append(f"People identified: {', '.join(all_people) if all_people else 'None'}")
        lines.append(f"Objects detected: {', '.join(all_objects) if all_objects else 'None'}")
        lines.append("\nInteractions:")
        lines.extend(interactions if interactions else ["No interactions detected"])
        
        return "\n".join(lines)


def generate_video_summary_with_groq(
    analysis_json_path: str,
    screenplay_path: str = "",
    transcript_path: str = "",
    output_path: str = "",
    api_key: Optional[str] = None
) -> str:
    """
    Convenience function to generate a complete video summary
    
    Args:
        analysis_json_path: Path to analysis.json
        screenplay_path: Path to screenplay.txt
        transcript_path: Path to transcript.txt  
        output_path: Path to save the summary
        api_key: Groq API key (optional, uses env var if not provided)
        
    Returns:
        Generated summary text
    """
    # Load analysis JSON
    with open(analysis_json_path, 'r', encoding='utf-8') as f:
        analysis_data = json.load(f)
    
    # Load screenplay
    screenplay_text = ""
    if screenplay_path and Path(screenplay_path).exists():
        with open(screenplay_path, 'r', encoding='utf-8') as f:
            screenplay_text = f.read()
    
    # Load transcript
    transcript_text = ""
    if transcript_path and Path(transcript_path).exists():
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript_text = f.read()
    
    # Generate summary
    generator = GroqSummaryGenerator(api_key=api_key)
    summary = generator.generate_detailed_summary(
        analysis_data,
        screenplay_text,
        transcript_text
    )
    
    # Save if output path provided
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        logger.info(f"Summary saved to {output_path}")
    
    return summary
