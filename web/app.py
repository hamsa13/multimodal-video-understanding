"""
Video Scene Understanding - Web Interface
Comprehensive Gradio-based web application for multimodal video analysis
With Groq LLM integration for AI-powered summaries
With GNN-based entity relationship visualization
Compatible with Gradio 6.x
"""

import os
import sys
import json
import shutil
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from datetime import datetime

import gradio as gr
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.pipeline import VideoPipeline
from src.entity_relationship_gnn import EntityRelationshipGNN
from src.text_to_audio import TextToAudioGenerator


# Global pipeline instance
pipeline: Optional[VideoPipeline] = None

# Store last analysis results for download
last_results: Dict[str, Any] = {}

# Store last GNN graph data
last_graph_data: Dict[str, Any] = {}


def get_pipeline() -> VideoPipeline:
    """Get or create pipeline instance"""
    global pipeline
    if pipeline is None:
        config = Config.load()
        pipeline = VideoPipeline(config)
    return pipeline


def format_time(seconds: float) -> str:
    """Format seconds as MM:SS.ms"""
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins:02d}:{secs:05.2f}"


def analyze_video(
    video_file: str,
    use_groq: bool,
    skip_faces: bool,
    skip_actions: bool,
    audio_screenplay: bool,
    audio_narrative: bool,
    audio_subtitles: bool,
    audio_transcript: bool,
    speaker_mapping: str,
    progress=gr.Progress()
) -> Tuple[str, str, str, str, str, str, str, str, str, str, Any, Any, Any, Any, str]:
    """
    Main analysis function for Gradio interface
    Returns 15 outputs: text tabs + graph + audio files + status
    """
    global last_results, last_graph_data
    
    if video_file is None:
        empty = "Please upload a video file first."
        return empty, empty, empty, empty, empty, empty, empty, "{}", empty, None, None, None, None, None, "⚠️ No video uploaded"
    
    try:
        progress(0, desc="Initializing pipeline...")
        pipe = get_pipeline()
        
        # Parse speaker mapping
        speaker_names = None
        if speaker_mapping and speaker_mapping.strip():
            try:
                speaker_names = json.loads(speaker_mapping)
            except json.JSONDecodeError:
                pass
        
        # Get Groq API key from environment
        groq_api_key = os.getenv("GROQ_API_KEY") if use_groq else None

        # Audio generation options
        audio_outputs = []
        if audio_screenplay:
            audio_outputs.append("screenplay")
        if audio_narrative:
            audio_outputs.append("narrative")
        if audio_subtitles:
            audio_outputs.append("subtitles")
        if audio_transcript:
            audio_outputs.append("transcript")
        
        # Create persistent output directory
        video_name = Path(video_file).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("outputs") / f"{video_name}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        stages = [
            "Extracting metadata",
            "Extracting audio",
            "Transcribing speech",
            "Identifying speakers", 
            "Detecting scenes",
            "Extracting frames",
            "Detecting objects",
            "Processing faces",
            "Recognizing actions",
            "Analyzing scenes",
            "Fusing multimodal data",
            "Generating screenplay",
            "Generating narrative",
            "Calculating metrics",
            "Generating summary",
            "Building entity graph",
            "Saving outputs"
        ]
        current_stage = [0]
        
        def progress_callback(stage: str, pct: float):
            for i, s in enumerate(stages):
                if s.lower() in stage.lower() or stage.lower() in s.lower():
                    current_stage[0] = i
                    break
            progress((current_stage[0] + 1) / len(stages), desc=stage)
        
        # Run pipeline
        result = pipe.process(
            video_path=video_file,
            output_dir=str(output_dir),
            speaker_names=speaker_names,
            skip_faces=skip_faces,
            skip_actions=skip_actions,
            generate_text=True,
            use_groq=use_groq,
            groq_api_key=groq_api_key,
            audio_outputs=audio_outputs,
            progress_callback=progress_callback
        )
        
        # Build Entity Relationship Graph
        progress(0.95, desc="Building entity relationship graph...")
        analysis_dict = result.analysis.to_dict()
        gnn = EntityRelationshipGNN()
        gnn.build_graph_from_analysis(analysis_dict)
        
        # Generate graph visualization
        graph_image_path = str(output_dir / "entity_graph.png")
        gnn.visualize(output_path=graph_image_path)
        
        # Get graph summary
        graph_data = gnn.to_dict()
        last_graph_data = graph_data
        
        graph_summary = _format_graph_summary(graph_data)
        
        # Store for downloads
        last_results = {
            'output_dir': str(output_dir),
            'files': result.output_files,
            'graph_data': graph_data
        }
        
        # Save graph data
        with open(output_dir / "entity_graph.json", "w") as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)
        
        # Format outputs
        transcript = result.analysis.full_transcript or "No speech detected."
        screenplay = result.generated_screenplay or "Screenplay not generated."
        narrative = result.generated_narrative or "Narrative not generated."
        detailed_summary = result.detailed_summary or "Summary not generated."
        scenes_text = _format_scenes(result.analysis.scenes)
        dialogue_text = _format_dialogues(result.analysis.scenes)
        entity_text = _format_entities(result.analysis)
        analysis_json = json.dumps(result.analysis.to_dict(), indent=2, ensure_ascii=False)
        
        # Status
        groq_status = "✓ Groq AI (llama-3.3-70b)" if use_groq and groq_api_key else "✗ Local generation"
        status = f"""✅ Analysis Complete!

📹 Video: {Path(video_file).name}
⏱️ Duration: {result.analysis.duration:.2f} seconds
🎬 Scenes: {len(result.analysis.scenes)}
👥 Speakers: {len(result.analysis.all_speakers)}
📦 Objects: {len(result.analysis.all_objects)}
🔗 Graph Nodes: {graph_data['summary']['total_entities']}
🔗 Graph Edges: {graph_data['summary']['total_relationships']}
🔊 Audio Files: {len(result.generated_audio_files)}
⚡ Processing: {result.processing_time:.1f}s
🤖 AI Engine: {groq_status}

📁 Output: {output_dir}"""
        
        # Return graph image if exists
        graph_img = graph_image_path if Path(graph_image_path).exists() else None

        # Robustly resolve audio file paths (absolute) so Gradio can serve them
        def _resolve_audio_key(key: str) -> Optional[str]:
            # Prefer value from result.output_files
            try:
                val = result.output_files.get(key)
            except Exception:
                val = None
            if val:
                p = Path(val)
                if p.exists():
                    return str(p.resolve())

            # Fallback to outputs/<video> audio_outputs/<name>.mp3
            base = key.replace('_audio', '')
            candidate = Path(output_dir) / 'audio_outputs' / f"{base}.mp3"
            if candidate.exists():
                return str(candidate.resolve())

            return None

        screenplay_audio = _resolve_audio_key("screenplay_audio")
        narrative_audio = _resolve_audio_key("narrative_audio")
        subtitles_audio = _resolve_audio_key("subtitles_audio")
        transcript_audio = _resolve_audio_key("transcript_audio")
        
        return (
            detailed_summary,
            screenplay,
            narrative,
            transcript,
            scenes_text,
            dialogue_text,
            entity_text,
            analysis_json,
            graph_summary,
            graph_img,
            screenplay_audio,
            narrative_audio,
            subtitles_audio,
            transcript_audio,
            status
        )
    
    except Exception as e:
        import traceback
        error_msg = f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"
        return (error_msg,) * 9 + (None, None, None, None, None, error_msg)


def _format_scenes(scenes) -> str:
    """Format scene information"""
    if not scenes:
        return "No scenes detected."
    
    lines = [f"🎬 SCENE ANALYSIS ({len(scenes)} scenes)\n{'='*50}\n"]
    
    for scene in scenes:
        lines.append(f"""
📍 SCENE {scene.scene_number}
   Time: {format_time(scene.start_time)} → {format_time(scene.end_time)} ({scene.end_time - scene.start_time:.1f}s)
   Location: {scene.location or 'Unknown'}
   Environment: {'Indoor' if scene.is_indoor else 'Outdoor'}
   Lighting: {scene.lighting or 'Normal'}
   Mood: {scene.mood or 'Neutral'}
   Objects: {', '.join(scene.objects[:10]) if scene.objects else 'None detected'}
   Actions: {', '.join(scene.actions[:5]) if scene.actions else 'None detected'}
   People: {len(scene.people)} detected
""")
    
    return "\n".join(lines)


def _format_dialogues(scenes) -> str:
    """Format all dialogues"""
    if not scenes:
        return "No dialogue detected."
    
    lines = ["💬 DIALOGUE TRANSCRIPT\n" + "="*50 + "\n"]
    
    dialogue_count = 0
    for scene in scenes:
        if scene.dialogues:
            lines.append(f"\n--- Scene {scene.scene_number} ({scene.location or 'Unknown'}) ---\n")
            for dlg in scene.dialogues:
                dialogue_count += 1
                timestamp = format_time(dlg.start_time)
                lines.append(f"[{timestamp}] {dlg.speaker}: {dlg.text}")
    
    if dialogue_count == 0:
        return "No dialogue detected in video."
    
    lines.insert(1, f"Total: {dialogue_count} dialogue exchanges\n")
    return "\n".join(lines)


def _format_entities(analysis) -> str:
    """Format entity and object information"""
    lines = ["🔍 ENTITY & OBJECT ANALYSIS\n" + "="*50 + "\n"]
    
    lines.append(f"\n👥 SPEAKERS ({len(analysis.all_speakers)})")
    for speaker in analysis.all_speakers:
        lines.append(f"   • {speaker}")
    
    lines.append(f"\n📦 OBJECTS DETECTED ({len(analysis.all_objects)})")
    for obj in sorted(analysis.all_objects)[:20]:
        lines.append(f"   • {obj}")
    if len(analysis.all_objects) > 20:
        lines.append(f"   ... and {len(analysis.all_objects) - 20} more")
    
    lines.append(f"\n🎭 ACTIONS DETECTED ({len(analysis.all_actions)})")
    for action in analysis.all_actions[:10]:
        lines.append(f"   • {action}")
    
    lines.append("\n👤 PEOPLE PER SCENE")
    for scene in analysis.scenes:
        people_count = len(scene.people)
        lines.append(f"   Scene {scene.scene_number}: {people_count} people")
    
    return "\n".join(lines)


def _format_graph_summary(graph_data: Dict[str, Any]) -> str:
    """Format GNN graph summary for display"""
    summary = graph_data.get('summary', {})
    important = graph_data.get('important_entities', [])
    
    lines = ["🔗 ENTITY RELATIONSHIP GRAPH\n" + "="*50 + "\n"]
    
    # Overall stats
    lines.append("📊 GRAPH STATISTICS")
    lines.append(f"   Total Entities: {summary.get('total_entities', 0)}")
    lines.append(f"   Total Relationships: {summary.get('total_relationships', 0)}")
    lines.append(f"   Graph Density: {summary.get('graph_density', 0):.4f}")
    
    # Entity type breakdown
    entity_types = summary.get('entity_types', {})
    lines.append("\n📌 ENTITY TYPES")
    for etype, count in entity_types.items():
        icon = {'speakers': '🎤', 'people': '👤', 'objects': '📦', 'locations': '📍', 'actions': '⚡'}.get(etype, '•')
        lines.append(f"   {icon} {etype.title()}: {count}")
    
    # Relationship types
    rel_types = summary.get('relationship_types', {})
    lines.append("\n🔗 RELATIONSHIP TYPES")
    for rtype, count in rel_types.items():
        lines.append(f"   • {rtype.replace('_', ' ').title()}: {count}")
    
    # Most important entities
    if important:
        lines.append("\n⭐ MOST IMPORTANT ENTITIES (by PageRank)")
        for i, entity in enumerate(important[:10], 1):
            lines.append(f"   {i}. {entity['name']} ({entity['type']}) - importance: {entity['importance']:.4f}, connections: {entity['connections']}")
    
    # Edge list preview
    edges = graph_data.get('edges', [])
    if edges:
        lines.append(f"\n📋 RELATIONSHIPS ({len(edges)} total)")
        for edge in edges[:15]:
            src_name = edge.get('source', '').replace('speaker_', '').replace('object_', '').replace('_', ' ')
            tgt_name = edge.get('target', '').replace('speaker_', '').replace('object_', '').replace('_', ' ')
            rel = edge.get('relation', '').replace('_', ' ')
            lines.append(f"   [{src_name}] --{rel}--> [{tgt_name}]")
        if len(edges) > 15:
            lines.append(f"   ... and {len(edges) - 15} more relationships")
    
    return "\n".join(lines)


def download_all_outputs():
    """Create download of all output files"""
    global last_results
    
    if not last_results or 'output_dir' not in last_results:
        return None
    
    output_dir = last_results['output_dir']
    if not Path(output_dir).exists():
        return None
    
    zip_path = shutil.make_archive(
        output_dir,
        'zip',
        root_dir=str(Path(output_dir).parent),
        base_dir=Path(output_dir).name
    )
    
    return zip_path


def quick_transcribe(video_file: str, progress=gr.Progress()) -> Tuple[str, str, str]:
    """Quick transcription only"""
    if video_file is None:
        return "", "", "Please upload a video file."
    
    try:
        from src.video_processor import VideoProcessor
        from src.speech_recognition import SpeechRecognizer
        
        progress(0.2, desc="Extracting audio...")
        config = Config.load()
        processor = VideoProcessor(config)
        recognizer = SpeechRecognizer(config)
        
        audio_path = processor.extract_audio(video_file)
        
        progress(0.5, desc="Transcribing...")
        result = recognizer.transcribe(audio_path)
        
        progress(0.9, desc="Generating SRT...")
        srt = recognizer.to_srt(result)
        
        processor.cleanup()
        
        status = f"✅ Transcribed {result.word_count} words in {len(result.segments)} segments"
        return result.text, srt, status
    
    except Exception as e:
        return "", "", f"❌ Error: {str(e)}"


def quick_scene_detect(video_file: str, progress=gr.Progress()) -> Tuple[str, str, str]:
    """Quick scene detection only"""
    if video_file is None:
        return "", "", "Please upload a video file."
    
    try:
        from src.scene_detection import SceneDetector
        
        progress(0.3, desc="Analyzing video...")
        config = Config.load()
        detector = SceneDetector(config)
        
        progress(0.6, desc="Detecting scenes...")
        result = detector.detect_scenes(video_file)
        
        scenes_text = f"🎬 Found {result.total_scenes} scenes:\n\n"
        for scene in result.scenes:
            scenes_text += (
                f"Scene {scene.scene_number}: "
                f"{format_time(scene.start_time)} → {format_time(scene.end_time)} "
                f"(duration: {scene.duration:.1f}s)\n"
            )
        
        scenes_json = json.dumps([{
            'scene': s.scene_number,
            'start': s.start_time,
            'end': s.end_time,
            'duration': s.duration
        } for s in result.scenes], indent=2)
        
        status = f"✅ Detected {result.total_scenes} scenes"
        return scenes_text, scenes_json, status
    
    except Exception as e:
        return "", "", f"❌ Error: {str(e)}"


def create_app() -> gr.Blocks:
    """Create the Gradio application"""
    
    has_groq_key = bool(os.getenv("GROQ_API_KEY"))
    
    with gr.Blocks(title="Video Scene Understanding - AI Video Analysis") as app:
        
        gr.Markdown("""
        # 🎬 Video Scene Understanding
        ### Multimodal AI Video Analysis System
        
        Upload a video to get comprehensive analysis including:
        **Speech Transcription** • **Scene Detection** • **Object Recognition** • **Face Detection** • **Action Recognition** • **AI-Generated Summaries**
        """)
        
        with gr.Tabs():
            # Full Analysis Tab
            with gr.TabItem("🔬 Full Analysis"):
                with gr.Row():
                    with gr.Column(scale=1):
                        video_input = gr.Video(label="📹 Upload Video")
                        
                        gr.Markdown("### ⚙️ Analysis Options")
                        
                        groq_info = " ✓ API key found" if has_groq_key else " ⚠️ Set GROQ_API_KEY in .env"
                        use_groq_cb = gr.Checkbox(
                            label="🤖 Use Groq AI (llama-3.3-70b-versatile)",
                            value=has_groq_key,
                            info="Generates detailed AI summaries from screenplay" + groq_info
                        )
                        
                        with gr.Accordion("Advanced Options", open=False):
                            skip_faces_cb = gr.Checkbox(
                                label="Skip Face Detection",
                                value=False,
                                info="Faster processing"
                            )
                            skip_actions_cb = gr.Checkbox(
                                label="Skip Action Recognition",
                                value=False,
                                info="Faster processing"
                            )
                            speaker_map_input = gr.Textbox(
                                label="Speaker Names (JSON)",
                                placeholder='{"SPEAKER_00": "John"}',
                                info="Optional speaker name mapping"
                            )

                        with gr.Accordion("🔊 Text to Audio Options", open=False):
                            audio_screenplay_cb = gr.Checkbox(
                                label="Convert Screenplay to Audio",
                                value=False
                            )
                            audio_narrative_cb = gr.Checkbox(
                                label="Convert Narrative to Audio",
                                value=False
                            )
                            audio_subtitles_cb = gr.Checkbox(
                                label="Convert Subtitles (SRT) to Audio",
                                value=False
                            )
                            audio_transcript_cb = gr.Checkbox(
                                label="Convert Transcript to Audio",
                                value=False
                            )
                        
                        analyze_btn = gr.Button("🚀 Analyze Video", variant="primary", size="lg")
                        
                        status_output = gr.Textbox(label="📊 Status", lines=10)
                        
                        download_btn = gr.Button("📥 Download All Outputs")
                        download_file = gr.File(label="Download")
                    
                    with gr.Column(scale=2):
                        with gr.Tabs():
                            with gr.TabItem("🤖 AI Summary"):
                                gr.Markdown("*Detailed AI-generated summary of the entire video*")
                                summary_output = gr.Textbox(
                                    label="Detailed Video Summary",
                                    lines=25,
                                    placeholder="AI-generated comprehensive summary will appear here..."
                                )
                            
                            with gr.TabItem("🎭 Screenplay"):
                                gr.Markdown("*Professional screenplay format output*")
                                screenplay_output = gr.Textbox(label="Screenplay", lines=25)
                            
                            with gr.TabItem("📖 Narrative"):
                                gr.Markdown("*AI-generated narrative description*")
                                narrative_output = gr.Textbox(label="Narrative", lines=25)
                            
                            with gr.TabItem("📝 Transcript"):
                                gr.Markdown("*Full speech transcription*")
                                transcript_output = gr.Textbox(label="Transcript", lines=25)
                            
                            with gr.TabItem("🎬 Scenes"):
                                gr.Markdown("*Scene-by-scene breakdown*")
                                scenes_output = gr.Textbox(label="Scene Analysis", lines=25)
                            
                            with gr.TabItem("💬 Dialogue"):
                                gr.Markdown("*All dialogue with timestamps*")
                                dialogue_output = gr.Textbox(label="Dialogue", lines=25)
                            
                            with gr.TabItem("🔍 Entities"):
                                gr.Markdown("*People, objects, and actions detected*")
                                entity_output = gr.Textbox(label="Entity Analysis", lines=25)
                            
                            with gr.TabItem("📊 JSON Data"):
                                gr.Markdown("*Complete analysis data in JSON format*")
                                json_output = gr.Code(label="Analysis JSON", language="json", lines=25)
                            
                            with gr.TabItem("🔗 Relationships"):
                                gr.Markdown("*Entity relationship graph powered by GNN*")
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        graph_summary_output = gr.Textbox(
                                            label="Graph Analysis",
                                            lines=20,
                                            placeholder="Entity relationship analysis will appear here..."
                                        )
                                    with gr.Column(scale=2):
                                        graph_image_output = gr.Image(
                                            label="Entity Relationship Graph",
                                            type="filepath",
                                            height=500
                                        )

                            with gr.TabItem("🔊 Audio Outputs"):
                                gr.Markdown("*Generated audio files from screenplay, narrative, subtitles, and transcript*")
                                with gr.Row():
                                    with gr.Column():
                                        screenplay_audio_output = gr.Audio(
                                            label="🎭 Screenplay Audio",
                                            type="filepath"
                                        )
                                        narrative_audio_output = gr.Audio(
                                            label="📖 Narrative Audio",
                                            type="filepath"
                                        )
                                    with gr.Column():
                                        subtitles_audio_output = gr.Audio(
                                            label="💬 Subtitles Audio",
                                            type="filepath"
                                        )
                                        transcript_audio_output = gr.Audio(
                                            label="📝 Transcript Audio",
                                            type="filepath"
                                        )
                
                analyze_btn.click(
                    fn=analyze_video,
                    inputs=[
                        video_input,
                        use_groq_cb,
                        skip_faces_cb,
                        skip_actions_cb,
                        audio_screenplay_cb,
                        audio_narrative_cb,
                        audio_subtitles_cb,
                        audio_transcript_cb,
                        speaker_map_input,
                    ],
                    outputs=[summary_output, screenplay_output, narrative_output, transcript_output,
                             scenes_output, dialogue_output, entity_output, json_output,
                             graph_summary_output, graph_image_output,
                             screenplay_audio_output, narrative_audio_output,
                             subtitles_audio_output, transcript_audio_output,
                             status_output]
                )
                
                download_btn.click(fn=download_all_outputs, inputs=[], outputs=[download_file])
            
            # Quick Transcribe Tab
            with gr.TabItem("🎤 Quick Transcribe"):
                gr.Markdown("### Fast Speech-to-Text (No full analysis)")
                
                with gr.Row():
                    with gr.Column():
                        trans_video = gr.Video(label="Upload Video")
                        trans_btn = gr.Button("🎤 Transcribe", variant="primary")
                        trans_status = gr.Textbox(label="Status", lines=2)
                    
                    with gr.Column():
                        trans_text = gr.Textbox(label="Transcript", lines=12)
                        trans_srt = gr.Textbox(label="SRT Subtitles", lines=12)
                
                trans_btn.click(
                    fn=quick_transcribe,
                    inputs=[trans_video],
                    outputs=[trans_text, trans_srt, trans_status]
                )
            
            # Scene Detection Tab
            with gr.TabItem("🎬 Scene Detection"):
                gr.Markdown("### Detect Scene Changes (No full analysis)")
                
                with gr.Row():
                    with gr.Column():
                        scene_video = gr.Video(label="Upload Video")
                        scene_btn = gr.Button("🎬 Detect Scenes", variant="primary")
                        scene_status = gr.Textbox(label="Status", lines=2)
                    
                    with gr.Column():
                        scene_text = gr.Textbox(label="Detected Scenes", lines=12)
                        scene_json = gr.Code(label="Scene Data (JSON)", language="json", lines=12)
                
                scene_btn.click(
                    fn=quick_scene_detect,
                    inputs=[scene_video],
                    outputs=[scene_text, scene_json, scene_status]
                )
            
            # Settings Tab
            with gr.TabItem("⚙️ Settings & Help"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("""
                        ### 🔑 API Keys Configuration
                        
                        Create a `.env` file in the project root with:
                        
                        ```
                        # Required for AI summaries
                        GROQ_API_KEY=your_groq_api_key_here
                        
                        # Optional for speaker diarization
                        HF_TOKEN=your_huggingface_token_here
                        ```
                        
                        **Get your free Groq API key at:** [console.groq.com](https://console.groq.com)
                        
                        ---
                        
                        ### 📁 Output Files Generated
                        
                        | File | Description |
                        |------|-------------|
                        | `screenplay.txt` | Professional screenplay format |
                        | `narrative.txt` | AI-generated narrative |
                        | `detailed_summary.txt` | Comprehensive Groq AI summary |
                        | `transcript.txt` | Full speech transcription |
                        | `analysis.json` | Complete analysis data |
                        | `subtitles.srt` | SRT subtitle file |
                        | `metrics.json` | Processing metrics |
                        """)
                    
                    with gr.Column():
                        gr.Markdown("""
                        ### 🎯 Features
                        
                        - **Speech Recognition**: Whisper-powered transcription
                        - **Speaker Diarization**: Identify who's speaking
                        - **Scene Detection**: Automatic scene boundary detection
                        - **Object Detection**: YOLOv8-powered object recognition
                        - **Face Detection**: MTCNN face detection
                        - **Action Recognition**: Activity classification
                        - **AI Summaries**: Groq llama-3.3-70b narratives
                        
                        ---
                        
                        ### 💻 Hardware Requirements
                        
                        - **Minimum**: 8GB RAM, Any modern CPU
                        - **Recommended**: 16GB RAM, NVIDIA GPU (6GB+ VRAM)
                        """)
                        
                        groq_status = "✅ Configured" if has_groq_key else "❌ Not set"
                        hf_status = "✅ Configured" if os.getenv("HF_TOKEN") else "❌ Not set"
                        
                        gr.Markdown(f"""
                        ---
                        ### 📊 Current Configuration
                        
                        | Setting | Status |
                        |---------|--------|
                        | GROQ_API_KEY | {groq_status} |
                        | HF_TOKEN | {hf_status} |
                        """)
                        # Output folders dropdown + generate audio
                        outputs_root = Path('outputs')
                        available_outputs = [d.name for d in outputs_root.iterdir() if d.is_dir()] if outputs_root.exists() else []
                        outputs_dropdown = gr.Dropdown(label="Select outputs folder", choices=available_outputs, value=available_outputs[0] if available_outputs else None)
                        gen_audio_btn = gr.Button("🔊 Generate Audio for Folder")
                        gen_status = gr.Textbox(label="Generation Status", lines=2)

                        with gr.Row():
                            with gr.Column():
                                settings_screenplay_audio = gr.Audio(label="Screenplay Audio (folder)", type="filepath")
                                settings_narrative_audio = gr.Audio(label="Narrative Audio (folder)", type="filepath")
                            with gr.Column():
                                settings_subtitles_audio = gr.Audio(label="Subtitles Audio (folder)", type="filepath")
                                settings_transcript_audio = gr.Audio(label="Transcript Audio (folder)", type="filepath")

                        def _generate_audio(folder_name: Optional[str]):
                            if not folder_name:
                                return "No folder selected", None, None, None, None, None, None, None, None
                            out_dir = Path('outputs') / folder_name
                            if not out_dir.exists():
                                return f"Folder not found: {folder_name}", None, None, None, None, None, None, None, None
                            tts = TextToAudioGenerator()
                            audio_dir = out_dir / 'audio_outputs'
                            audio_dir.mkdir(parents=True, exist_ok=True)
                            mapping = {}
                            # mapping of source files to target names
                            sources = {
                                'screenplay': out_dir / 'screenplay.txt',
                                'narrative': out_dir / 'narrative.txt',
                                'transcript': out_dir / 'transcript.txt',
                                'subtitles': out_dir / 'subtitles.srt'
                            }
                            created = []
                            for key, src in sources.items():
                                target = audio_dir / f"{key}.mp3"
                                if not src.exists():
                                    mapping[key] = None
                                    continue
                                try:
                                    if key == 'subtitles':
                                        text = src.read_text(encoding='utf-8')
                                        tts.srt_to_mp3(text, str(target))
                                    else:
                                        text = src.read_text(encoding='utf-8')
                                        tts.text_to_mp3(text, str(target))
                                    mapping[key] = str(target.resolve())
                                    created.append(key)
                                except Exception as e:
                                    mapping[key] = None
                            status_text = f"Created: {', '.join(created)}" if created else "No audio created (missing source files or TTS error)"

                            screenplay_path = mapping.get('screenplay')
                            narrative_path = mapping.get('narrative')
                            subtitles_path = mapping.get('subtitles')
                            transcript_path = mapping.get('transcript')

                            # Return for both Settings tab players and Audio Outputs tab players
                            return (
                                status_text,
                                screenplay_path,
                                narrative_path,
                                subtitles_path,
                                transcript_path,
                                screenplay_path,
                                narrative_path,
                                subtitles_path,
                                transcript_path,
                            )

                        gen_audio_btn.click(
                            fn=_generate_audio,
                            inputs=[outputs_dropdown],
                            outputs=[
                                gen_status,
                                settings_screenplay_audio,
                                settings_narrative_audio,
                                settings_subtitles_audio,
                                settings_transcript_audio,
                                screenplay_audio_output,
                                narrative_audio_output,
                                subtitles_audio_output,
                                transcript_audio_output,
                            ]
                        )
        
        gr.Markdown("""
        ---
        **🎬 Video Scene Understanding** | Multimodal AI Analysis Pipeline | Built with OpenCV • Whisper • YOLOv8 • Groq • Gradio
        """)
    
    return app


def main():
    """Run the web application"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Video Scene Understanding Web Interface")
    parser.add_argument("--port", type=int, default=7860, help="Port number")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🎬 Video Scene Understanding - Web Interface")
    print("="*60)
    
    if os.getenv("GROQ_API_KEY"):
        print("✅ GROQ_API_KEY found - AI summaries enabled")
    else:
        print("⚠️  GROQ_API_KEY not set - Add to .env for AI summaries")
    
    if os.getenv("HF_TOKEN"):
        print("✅ HF_TOKEN found - Speaker diarization enabled")
    else:
        print("⚠️  HF_TOKEN not set - Speaker diarization may be limited")
    
    print("="*60 + "\n")
    
    app = create_app()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share
    )


if __name__ == "__main__":
    main()
