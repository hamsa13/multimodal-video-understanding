"""
Video Analysis Pipeline
Main orchestrator that runs all analysis modules and produces final output
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
import time

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel

from .config import Config
from .video_processor import VideoProcessor, VideoMetadata
from .speech_recognition import SpeechRecognizer, TranscriptionResult
from .speaker_diarization import SpeakerDiarizer, DiarizationResult
from .face_recognition_module import FaceProcessor, FrameFaces
from .scene_detection import SceneDetector, SceneDetectionResult
from .object_detection import ObjectDetector, FrameDetections
from .action_recognition import ActionRecognizer, ClipActions
from .multimodal_fusion import MultimodalFusion, VideoAnalysis
from .text_generator import TextGenerator
from .accuracy_metrics import MetricsCollector, PipelineMetrics
from .summary_generator import DetailedSummaryGenerator
from .groq_summary import GroqSummaryGenerator
from .text_to_audio import TextToAudioGenerator

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class PipelineResult:
    """Complete pipeline result"""
    video_path: str
    analysis: VideoAnalysis
    generated_narrative: str
    generated_screenplay: str
    processing_time: float
    output_files: Dict[str, str] = field(default_factory=dict)
    generated_audio_files: Dict[str, str] = field(default_factory=dict)
    metrics: Optional[PipelineMetrics] = None
    detailed_summary: str = ""
    
    def save_all(self, output_dir: str) -> Dict[str, str]:
        """Save all outputs to directory"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON analysis
        json_path = output_dir / "analysis.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.analysis.to_dict(), f, indent=2, ensure_ascii=False)
        self.output_files['json'] = str(json_path)
        
        # Save narrative
        narrative_path = output_dir / "narrative.txt"
        with open(narrative_path, 'w', encoding='utf-8') as f:
            f.write(self.generated_narrative)
        self.output_files['narrative'] = str(narrative_path)
        
        # Save screenplay
        screenplay_path = output_dir / "screenplay.txt"
        with open(screenplay_path, 'w', encoding='utf-8') as f:
            f.write(self.generated_screenplay)
        self.output_files['screenplay'] = str(screenplay_path)
        
        # Save transcript
        transcript_path = output_dir / "transcript.txt"
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write(self.analysis.full_transcript)
        self.output_files['transcript'] = str(transcript_path)
        
        # Save SRT subtitles
        srt_path = output_dir / "subtitles.srt"
        with open(srt_path, 'w', encoding='utf-8') as f:
            f.write(self._generate_srt())
        self.output_files['srt'] = str(srt_path)

        # Save generated TTS audio references (if created during process)
        for key, path in self.generated_audio_files.items():
            self.output_files[key] = path
        
        # Save metrics if available
        if self.metrics is not None:
            metrics_json_path = output_dir / "metrics.json"
            with open(metrics_json_path, 'w', encoding='utf-8') as f:
                json.dump(self.metrics.to_dict(), f, indent=2, ensure_ascii=False)
            self.output_files['metrics_json'] = str(metrics_json_path)
            
            metrics_report_path = output_dir / "metrics_report.txt"
            with open(metrics_report_path, 'w', encoding='utf-8') as f:
                f.write(self.metrics.generate_report())
            self.output_files['metrics_report'] = str(metrics_report_path)
        
        # Save detailed summary if available
        if self.detailed_summary:
            summary_path = output_dir / "detailed_summary.txt"
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(self.detailed_summary)
            self.output_files['detailed_summary'] = str(summary_path)
        
        logger.info(f"Saved all outputs to {output_dir}")
        return self.output_files
    
    def _generate_srt(self) -> str:
        """Generate SRT subtitle file"""
        lines = []
        counter = 1
        
        for scene in self.analysis.scenes:
            for dialogue in scene.dialogues:
                start = self._format_srt_time(dialogue.start_time)
                end = self._format_srt_time(dialogue.end_time)
                
                lines.append(str(counter))
                lines.append(f"{start} --> {end}")
                lines.append(f"[{dialogue.speaker}]: {dialogue.text}")
                lines.append("")
                counter += 1
        
        return "\n".join(lines)
    
    @staticmethod
    def _format_srt_time(seconds: float) -> str:
        """Format time as SRT timestamp"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"




class VideoPipeline:
    """
    Main video analysis pipeline
    Orchestrates all processing modules
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the pipeline
        
        Args:
            config: Configuration object (loads default if None)
        """
        self.config = config or Config.load()
        self.config.create_dirs()
        
        # Setup logging
        self._setup_logging()
        
        # Initialize modules (lazy loading)
        self._video_processor = None
        self._speech_recognizer = None
        self._speaker_diarizer = None
        self._face_processor = None
        self._scene_detector = None
        self._object_detector = None
        self._action_recognizer = None
        self._fusion = None
        self._text_generator = None
        
        logger.info("Video pipeline initialized")
    
    def _setup_logging(self) -> None:
        """Setup logging configuration"""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Lazy loading properties
    @property
    def video_processor(self) -> VideoProcessor:
        if self._video_processor is None:
            self._video_processor = VideoProcessor(self.config)
        return self._video_processor
    
    @property
    def speech_recognizer(self) -> SpeechRecognizer:
        if self._speech_recognizer is None:
            self._speech_recognizer = SpeechRecognizer(self.config)
        return self._speech_recognizer
    
    @property
    def speaker_diarizer(self) -> SpeakerDiarizer:
        if self._speaker_diarizer is None:
            self._speaker_diarizer = SpeakerDiarizer(self.config)
        return self._speaker_diarizer
    
    @property
    def face_processor(self) -> FaceProcessor:
        if self._face_processor is None:
            self._face_processor = FaceProcessor(self.config)
        return self._face_processor
    
    @property
    def scene_detector(self) -> SceneDetector:
        if self._scene_detector is None:
            self._scene_detector = SceneDetector(self.config)
        return self._scene_detector
    
    @property
    def object_detector(self) -> ObjectDetector:
        if self._object_detector is None:
            self._object_detector = ObjectDetector(self.config)
        return self._object_detector
    
    @property
    def action_recognizer(self) -> ActionRecognizer:
        if self._action_recognizer is None:
            self._action_recognizer = ActionRecognizer(self.config)
        return self._action_recognizer
    
    @property
    def fusion(self) -> MultimodalFusion:
        if self._fusion is None:
            self._fusion = MultimodalFusion(self.config)
        return self._fusion
    
    @property
    def text_generator(self) -> TextGenerator:
        if self._text_generator is None:
            self._text_generator = TextGenerator(self.config)
        return self._text_generator
    
    def process(
        self,
        video_path: str,
        output_dir: Optional[str] = None,
        speaker_names: Optional[Dict[str, str]] = None,
        skip_faces: bool = False,
        skip_actions: bool = False,
        generate_text: bool = True,
        use_groq: bool = True,
        groq_api_key: Optional[str] = None,
        audio_outputs: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> PipelineResult:
        """
        Process a video file through the complete pipeline
        
        Args:
            video_path: Path to video file
            output_dir: Output directory (default: outputs/<video_name>)
            speaker_names: Optional mapping of speaker IDs to names
            skip_faces: Skip face detection (faster processing)
            skip_actions: Skip action recognition (faster processing)
            generate_text: Whether to generate LLM narratives
            use_groq: Use Groq LLM (llama-3.3-70b-versatile) for detailed summary
            groq_api_key: Groq API key (or set GROQ_API_KEY env var)
            audio_outputs: Optional list of text artifacts to convert to audio.
                Supported values: screenplay, narrative, transcript, subtitles
            progress_callback: Optional callback for progress updates
        
        Returns:
            PipelineResult with all analysis and generated content
        """
        start_time = time.time()
        video_path = str(Path(video_path).resolve())
        
        if output_dir is None:
            video_name = Path(video_path).stem
            output_dir = Path(self.config.output_dir) / video_name
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics collector
        metrics_collector = MetricsCollector()
        metrics_collector.set_video_path(video_path)
        
        console.print(Panel.fit(
            f"[bold blue]Processing Video[/bold blue]\n{video_path}",
            title="Video Scene Understanding"
        ))
        
        def update_progress(stage: str, progress: float = 0):
            if progress_callback:
                progress_callback(stage, progress)
            console.print(f"[yellow]→[/yellow] {stage}")
        
        # Step 1: Video Metadata
        update_progress("Extracting video metadata...")
        metadata = self.video_processor.get_metadata(video_path)
        self._print_metadata(metadata)
        metrics_collector.update_video_metrics(metadata, 0)
        
        # Step 2: Extract Audio
        update_progress("Extracting audio...")
        audio_path = self.video_processor.extract_audio(
            video_path,
            output_path=str(output_dir / "audio.wav")
        )
        
        # Step 3: Speech Recognition
        update_progress("Transcribing speech...")
        transcription = self.speech_recognizer.transcribe(audio_path)
        console.print(f"  [green]✓[/green] Transcribed {transcription.word_count} words")
        metrics_collector.update_audio_metrics(transcription, metadata.duration)
        
        # Step 4: Speaker Diarization
        update_progress("Identifying speakers...")
        diarization = None
        if self.config.diarization.enabled:
            diarization = self.speaker_diarizer.diarize(audio_path)
            console.print(f"  [green]✓[/green] Found {diarization.num_speakers} speakers")
        
        # Step 5: Scene Detection
        update_progress("Detecting scenes...")
        scene_result = self.scene_detector.detect_scenes(video_path)
        console.print(f"  [green]✓[/green] Detected {scene_result.total_scenes} scenes")
        
        # Save scene thumbnails
        scenes_with_thumbs = self.scene_detector.save_scene_thumbnails(
            video_path, scene_result.scenes, str(output_dir / "thumbnails")
        )
        
        # Step 6: Extract Frames for Analysis
        update_progress("Extracting frames...")
        frames = self.video_processor.extract_frames(
            video_path,
            output_dir=str(output_dir / "frames"),
            return_images=True
        )
        console.print(f"  [green]✓[/green] Extracted {len(frames)} frames")
        metrics_collector.update_video_metrics(metadata, len(frames))
        
        # Step 7: Object Detection
        update_progress("Detecting objects...")
        object_results, object_tracks = self.object_detector.detect_and_track(video_path)
        obj_summary = self.object_detector.summarize_detections(object_results)
        console.print(f"  [green]✓[/green] Found {obj_summary['unique_objects']} object types")
        metrics_collector.update_object_metrics(object_results, len(frames))
        
        # Step 8: Face Detection & Recognition
        face_results = []
        if not skip_faces:
            update_progress("Processing faces...")
            face_results = self._process_faces(frames)
            console.print(f"  [green]✓[/green] Processed faces in {len(face_results)} frames")
            metrics_collector.update_face_metrics(face_results, len(frames))
        
        # Step 9: Action Recognition
        action_results = []
        if not skip_actions:
            update_progress("Recognizing actions...")
            action_results = self.action_recognizer.process_video(video_path)
            action_summary = self.action_recognizer.summarize_actions(action_results)
            console.print(f"  [green]✓[/green] Recognized {action_summary['unique_actions']} actions")
            metrics_collector.update_action_metrics(action_results)
        
        # Step 10: Analyze Scenes
        update_progress("Analyzing scenes...")
        analyzed_scenes = self._analyze_scenes(
            frames, scenes_with_thumbs, metadata
        )
        
        # Step 11: Multimodal Fusion
        update_progress("Fusing multimodal data...")
        video_analysis = self.fusion.fuse(
            video_metadata={
                'path': video_path,
                'duration': metadata.duration,
                'fps': metadata.fps,
                'width': metadata.width,
                'height': metadata.height
            },
            scenes=analyzed_scenes,
            transcription={
                'text': transcription.text,
                'segments': [
                    {'start': s.start, 'end': s.end, 'text': s.text, 'speaker': s.speaker}
                    for s in transcription.segments
                ]
            },
            diarization={
                'segments': [
                    {'start': s.start, 'end': s.end, 'speaker': s.speaker}
                    for s in diarization.segments
                ] if diarization else []
            } if diarization else None,
            face_results=[
                {
                    'timestamp': f.timestamp,
                    'detections': [{'bbox': (d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2)} 
                                  for d in f.detections],
                    'matches': [{'identity': m.identity, 'confidence': m.confidence} 
                               for m in f.matches]
                }
                for f in face_results
            ],
            object_results=[
                {
                    'timestamp': f.timestamp,
                    'detections': [{'class_name': d.class_name, 'confidence': d.confidence}
                                  for d in f.detections]
                }
                for f in object_results
            ],
            action_results=[
                {
                    'start_time': a.start_time,
                    'end_time': a.end_time,
                    'predictions': [{'action': p.action, 'confidence': p.confidence}
                                   for p in a.predictions]
                }
                for a in action_results
            ],
            speaker_names=speaker_names
        )

        # Estimate speaker gender from audio (heuristic pitch-based)
        update_progress("Analyzing voice characteristics...")
        try:
            self._estimate_voice_genders(str(audio_path), video_analysis)
            self._extract_names_and_label_speakers(video_analysis)
            # Rebuild transcript with updated speaker labels
            video_analysis.rebuild_transcript()
            console.print(f"  [green]✓[/green] Analyzed voice characteristics")
        except Exception as e:
            logger.warning(f"Voice analysis failed: {e}")
        
        # Step 12: Generate Text
        narrative = ""
        screenplay = ""
        groq_generator = None
        if generate_text:
            update_progress("Generating screenplay...")
            screenplay = self.text_generator.generate_screenplay_format(video_analysis.to_dict())

            # Use Groq to generate narrative from screenplay when enabled
            if use_groq:
                update_progress("Generating narrative from screenplay with Groq (llama-3.3-70b)...")
                groq_generator = GroqSummaryGenerator(api_key=groq_api_key)
                narrative = groq_generator.generate_narrative(screenplay, video_analysis.to_dict())
                console.print(f"  [green]✓[/green] Generated Groq narrative from screenplay")
            else:
                update_progress("Generating narrative...")
                narrative = self.text_generator.generate_full_narrative(video_analysis.to_dict())
                console.print(f"  [green]✓[/green] Generated narrative text")
        
        # Step 13: Update final metrics
        update_progress("Calculating accuracy metrics...")
        metrics_collector.update_scene_metrics(video_analysis.scenes, metadata.duration)
        metrics_collector.update_speaker_metrics(video_analysis)
        
        # Step 14: Generate detailed summary
        update_progress("Generating detailed summary...")

        # Use Groq LLM for detailed summary from generated screenplay if enabled
        if use_groq:
            update_progress("Generating AI-powered summary from screenplay with Groq (llama-3.3-70b)...")
            if groq_generator is None:
                groq_generator = GroqSummaryGenerator(api_key=groq_api_key)

            detailed_summary = groq_generator.generate_detailed_summary(
                video_analysis.to_dict(),
                screenplay,
                video_analysis.full_transcript
            )
            console.print(f"  [green]✓[/green] Generated Groq AI summary from screenplay")
        else:
            summary_generator = DetailedSummaryGenerator()
            detailed_summary = summary_generator.generate_summary(
                video_analysis.to_dict(),
                screenplay,
                metrics_collector.get_metrics().to_dict()
            )
            console.print(f"  [green]✓[/green] Generated detailed summary")
        
        # Create result
        processing_time = time.time() - start_time
        metrics_collector.set_processing_time(processing_time)
        
        result = PipelineResult(
            video_path=video_path,
            analysis=video_analysis,
            generated_narrative=narrative,
            generated_screenplay=screenplay,
            processing_time=processing_time,
            metrics=metrics_collector.get_metrics(),
            detailed_summary=detailed_summary
        )

        # Generate audio from text outputs:
        # - If caller requested specific `audio_outputs`, use those.
        # - Otherwise, if we generated text (screenplay/narrative/transcript/subtitles),
        #   auto-generate audio for the common artifacts so the frontend always has audio files.
        selected_audio_outputs = set(audio_outputs or [])
        if not selected_audio_outputs and generate_text:
            # default auto-generation for all produced text artifacts
            selected_audio_outputs = {"screenplay", "narrative", "transcript", "subtitles"}

        if selected_audio_outputs:
            update_progress("Converting text outputs to audio...")
            result.generated_audio_files = self._generate_text_audios(
                result=result,
                output_dir=output_dir,
                selected_outputs=selected_audio_outputs
            )
        
        # Save outputs
        update_progress("Saving outputs...")
        result.save_all(str(output_dir))
        
        # Print summary
        self._print_summary(result)
        
        return result

    def _generate_text_audios(
        self,
        result: PipelineResult,
        output_dir: Path,
        selected_outputs: set
    ) -> Dict[str, str]:
        """Generate MP3 files for selected text artifacts."""
        generator = TextToAudioGenerator()
        audio_dir = output_dir / "audio_outputs"
        audio_dir.mkdir(parents=True, exist_ok=True)

        created: Dict[str, str] = {}

        def safe_create(label: str, creator) -> None:
            try:
                path = creator()
                created[label] = path
            except Exception as exc:
                logger.warning(f"Failed to generate {label}: {exc}")

        if "screenplay" in selected_outputs:
            safe_create(
                "screenplay_audio",
                lambda: generator.text_to_mp3(
                    result.generated_screenplay,
                    str(audio_dir / "screenplay.mp3")
                )
            )

        if "narrative" in selected_outputs:
            safe_create(
                "narrative_audio",
                lambda: generator.text_to_mp3(
                    result.generated_narrative,
                    str(audio_dir / "narrative.mp3")
                )
            )

        if "transcript" in selected_outputs:
            safe_create(
                "transcript_audio",
                lambda: generator.text_to_mp3(
                    result.analysis.full_transcript,
                    str(audio_dir / "transcript.mp3")
                )
            )

        if "subtitles" in selected_outputs:
            safe_create(
                "subtitles_audio",
                lambda: generator.srt_to_mp3(
                    result._generate_srt(),
                    str(audio_dir / "subtitles.mp3")
                )
            )

        return created
    
    def _process_faces(self, frames: List) -> List[FrameFaces]:
        """Process all frames for face detection"""
        results = []
        
        for frame in frames:
            if frame.image is not None:
                face_result = self.face_processor.process_frame(
                    frame=frame.image,
                    frame_number=frame.frame_number,
                    timestamp=frame.timestamp
                )
                results.append(face_result)
        
        return results
    
    def _analyze_scenes(
        self,
        frames: List,
        scenes: List,
        metadata: VideoMetadata
    ) -> List[Dict[str, Any]]:
        """Analyze each scene with environment classification"""
        analyzed = []
        
        for scene in scenes:
            # Get frames for this scene
            scene_frames = [
                f.image for f in frames
                if f.image is not None and
                scene.start_time <= f.timestamp <= scene.end_time
            ]
            
            # Analyze scene
            if scene_frames:
                scene = self.scene_detector.analyze_scene(scene_frames, scene)
            
            analyzed.append({
                'scene_number': scene.scene_number,
                'start_time': scene.start_time,
                'end_time': scene.end_time,
                'duration': scene.duration,
                'environment': scene.environment,
                'is_indoor': scene.is_indoor,
                'brightness': scene.brightness,
                'thumbnail_path': scene.thumbnail_path
            })
        
        return analyzed
    
    def _print_metadata(self, metadata: VideoMetadata) -> None:
        """Print video metadata"""
        table = Table(title="Video Information")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Duration", f"{metadata.duration:.2f} seconds")
        table.add_row("Resolution", f"{metadata.width}x{metadata.height}")
        table.add_row("FPS", f"{metadata.fps:.2f}")
        table.add_row("Total Frames", str(metadata.total_frames))
        table.add_row("Has Audio", "Yes" if metadata.has_audio else "No")
        
        console.print(table)
    
    def _print_summary(self, result: PipelineResult) -> None:
        """Print processing summary"""
        console.print()
        
        summary_lines = [
            f"[bold green]Processing Complete![/bold green]\n",
            f"Time: {result.processing_time:.2f} seconds",
            f"Scenes: {len(result.analysis.scenes)}",
            f"Speakers: {len(result.analysis.all_speakers)}",
            f"People Recognized: {len(result.analysis.all_people)}",
            f"Objects Detected: {len(result.analysis.all_objects)}",
            f"Actions Recognized: {len(result.analysis.all_actions)}"
        ]
        
        if result.metrics:
            confidence = result.metrics.overall_confidence * 100
            quality = result.metrics.data_quality_score
            summary_lines.append("")
            summary_lines.append(f"[cyan]Quality Metrics:[/cyan]")
            summary_lines.append(f"  Confidence: {confidence:.1f}%")
            summary_lines.append(f"  Data Quality: {quality:.1f}%")
        
        console.print(Panel.fit("\n".join(summary_lines), title="Summary"))
        
        console.print("\n[bold]Output Files:[/bold]")
        for name, path in result.output_files.items():
            console.print(f"  [blue]•[/blue] {name}: {path}")
    
    def cleanup(self) -> None:
        """Clean up temporary files"""
        self.video_processor.cleanup()
        logger.info("Cleanup complete")

    def _estimate_voice_genders(self, audio_path: str, video_analysis) -> None:
        """Estimate voice gender for dialogue turns using robust pitch heuristics.

        Uses voiced-frame checks and conservative thresholds. If confidence is low,
        speaker stays unlabeled to avoid incorrect gender assignments.
        """
        try:
            import librosa
            import numpy as np
        except Exception:
            logger.warning("librosa not available; skipping voice gender estimation")
            return

        # Conservative pitch bands (Hz)
        MALE_UPPER = 155.0
        FEMALE_LOWER = 185.0

        # Load full audio
        try:
            y, sr = librosa.load(audio_path, sr=None)
        except Exception as e:
            logger.warning(f"Failed to load audio for gender estimation: {e}")
            return

        logger.info("Estimating voice gender for dialogue segments...")

        for scene in video_analysis.scenes:
            for dlg in scene.dialogues:
                # Extract center region of dialogue segment
                start = max(0.0, float(dlg.start_time) - 0.05)
                end = float(dlg.end_time) + 0.05
                s_idx = int(start * sr)
                e_idx = int(end * sr)
                segment = y[s_idx:e_idx]
                duration = max(0.0, end - start)
                if segment.size < 2048 or duration < 0.8:
                    # too short to analyze
                    continue

                try:
                    # Use YIN to estimate fundamental frequencies
                    f0 = librosa.yin(segment, fmin=50, fmax=500, sr=sr)
                    f0_clean = f0[(~np.isnan(f0)) & (f0 > 0)]

                    if f0_clean.size < 10:
                        continue

                    voiced_ratio = float(f0_clean.size) / float(f0.size if f0.size > 0 else 1)
                    if voiced_ratio < 0.35:
                        continue

                    p25 = float(np.percentile(f0_clean, 25))
                    p50 = float(np.percentile(f0_clean, 50))
                    p75 = float(np.percentile(f0_clean, 75))

                    gender = None
                    confidence = 0.0

                    # Male if entire central range stays in lower band
                    if p50 <= MALE_UPPER and p75 <= 185.0:
                        gender = 'Male'
                        confidence = min(0.95, 0.55 + (MALE_UPPER - p50) / 120.0 + voiced_ratio * 0.2)

                    # Female if central range stays in higher band
                    elif p50 >= FEMALE_LOWER and p25 >= 155.0:
                        gender = 'Female'
                        confidence = min(0.95, 0.55 + (p50 - FEMALE_LOWER) / 140.0 + voiced_ratio * 0.2)

                    # Ambiguous range: do not force classification
                    if gender and confidence >= 0.62:
                        dlg.speaker = f"Speaker ({gender})"

                except Exception as e:
                    logger.debug(f"Gender estimation failed for segment: {e}")
                    continue

    def _extract_names_and_label_speakers(self, video_analysis) -> None:
        """Label speakers with numbered identifiers based on gender.
        
        Since we can't reliably extract speaker names from dialogue content
        (names mentioned are often about other things, not speakers), 
        we use gender-based labels like 'Woman', 'Man', etc.
        """
        # Track unique speakers by gender
        male_count = 0
        female_count = 0
        speaker_map = {}  # original label -> new label
        
        for scene in video_analysis.scenes:
            for dlg in scene.dialogues:
                original = dlg.speaker or 'Speaker'
                
                if original in speaker_map:
                    dlg.speaker = speaker_map[original]
                    continue
                
                # Determine new label based on gender
                if '(Female)' in original:
                    female_count += 1
                    new_label = f"Woman {female_count}" if female_count > 1 else "Woman"
                    speaker_map[original] = new_label
                    dlg.speaker = new_label
                elif '(Male)' in original:
                    male_count += 1
                    new_label = f"Man {male_count}" if male_count > 1 else "Man"
                    speaker_map[original] = new_label
                    dlg.speaker = new_label
                else:
                    # No reliable gender detected
                    dlg.speaker = 'Speaker'
        
        logger.info(f"Labeled {male_count} male and {female_count} female speakers")


def quick_analyze(
    video_path: str,
    output_dir: Optional[str] = None,
    config_path: Optional[str] = None
) -> PipelineResult:
    """
    Quick analysis function for simple usage
    
    Args:
        video_path: Path to video file
        output_dir: Output directory
        config_path: Optional config file path
    
    Returns:
        PipelineResult
    """
    config = Config.load(config_path)
    pipeline = VideoPipeline(config)
    
    try:
        result = pipeline.process(video_path, output_dir)
        return result
    finally:
        pipeline.cleanup()
