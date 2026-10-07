#!/usr/bin/env python3
"""
Video Scene Understanding - Command Line Interface
"""

import os
import sys
from pathlib import Path

import click
from rich.console import Console

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.pipeline import VideoPipeline, quick_analyze
from src.text_to_audio import TextToAudioGenerator

console = Console()


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """Video Scene Understanding - Multimodal Video Analysis Tool"""
    pass


@cli.command()
@click.argument('video_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output directory')
@click.option('--config', '-c', type=click.Path(exists=True), help='Config file path')
@click.option('--skip-faces', is_flag=True, help='Skip face detection')
@click.option('--skip-actions', is_flag=True, help='Skip action recognition')
@click.option('--no-llm', is_flag=True, help='Skip LLM text generation')
@click.option('--speaker-map', type=str, help='JSON speaker name mapping')
@click.option('--audio-screenplay', is_flag=True, help='Generate audio from screenplay')
@click.option('--audio-narrative', is_flag=True, help='Generate audio from narrative')
@click.option('--audio-transcript', is_flag=True, help='Generate audio from transcript')
@click.option('--audio-subtitles', is_flag=True, help='Generate audio from subtitles (SRT)')
def analyze(
    video_path,
    output,
    config,
    skip_faces,
    skip_actions,
    no_llm,
    speaker_map,
    audio_screenplay,
    audio_narrative,
    audio_transcript,
    audio_subtitles,
):
    """
    Analyze a video file and generate scene descriptions.
    
    Example:
        python -m cli analyze video.mp4 -o output/
    """
    console.print("[bold blue]Video Scene Understanding[/bold blue]")
    console.print(f"Analyzing: {video_path}")
    
    # Parse speaker mapping
    speaker_names = None
    if speaker_map:
        import json
        speaker_names = json.loads(speaker_map)
    
    # Load config
    cfg = Config.load(config)

    audio_outputs = []
    if audio_screenplay:
        audio_outputs.append('screenplay')
    if audio_narrative:
        audio_outputs.append('narrative')
    if audio_transcript:
        audio_outputs.append('transcript')
    if audio_subtitles:
        audio_outputs.append('subtitles')
    
    # Create pipeline
    pipeline = VideoPipeline(cfg)
    
    try:
        result = pipeline.process(
            video_path=video_path,
            output_dir=output,
            speaker_names=speaker_names,
            skip_faces=skip_faces,
            skip_actions=skip_actions,
            generate_text=not no_llm,
            audio_outputs=audio_outputs
        )
        
        console.print(f"\n[green]✓ Analysis complete![/green]")
        console.print(f"Output saved to: {output or cfg.output_dir}")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise click.Abort()
    finally:
        pipeline.cleanup()


@cli.command()
@click.argument('video_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), default='transcript.txt', help='Output file')
def transcribe(video_path, output):
    """
    Transcribe audio from a video file.
    
    Example:
        python -m cli transcribe video.mp4 -o transcript.txt
    """
    from src.video_processor import VideoProcessor
    from src.speech_recognition import SpeechRecognizer
    
    config = Config.load()
    
    console.print(f"Transcribing: {video_path}")
    
    # Extract audio
    processor = VideoProcessor(config)
    audio_path = processor.extract_audio(video_path)
    
    # Transcribe
    recognizer = SpeechRecognizer(config)
    result = recognizer.transcribe(audio_path)
    
    # Save
    with open(output, 'w') as f:
        f.write(result.text)
    
    # Also save SRT
    srt_path = Path(output).with_suffix('.srt')
    with open(srt_path, 'w') as f:
        f.write(recognizer.to_srt(result))
    
    console.print(f"[green]✓ Transcript saved to: {output}[/green]")
    console.print(f"[green]✓ Subtitles saved to: {srt_path}[/green]")
    
    processor.cleanup()


@cli.command()
@click.argument('video_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output directory for thumbnails')
def detect_scenes(video_path, output):
    """
    Detect scene changes in a video.
    
    Example:
        python -m cli detect-scenes video.mp4 -o scenes/
    """
    from src.scene_detection import SceneDetector
    
    config = Config.load()
    detector = SceneDetector(config)
    
    console.print(f"Detecting scenes in: {video_path}")
    
    result = detector.detect_scenes(video_path)
    
    console.print(f"\n[green]Found {result.total_scenes} scenes[/green]")
    
    for scene in result.scenes:
        console.print(
            f"  Scene {scene.scene_number}: "
            f"{scene.start_time:.2f}s - {scene.end_time:.2f}s "
            f"({scene.duration:.2f}s)"
        )
    
    if output:
        detector.save_scene_thumbnails(video_path, result.scenes, output)
        console.print(f"\nThumbnails saved to: {output}")


@cli.command()
@click.argument('video_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), default='objects.json', help='Output JSON file')
def detect_objects(video_path, output):
    """
    Detect objects in a video.
    
    Example:
        python -m cli detect-objects video.mp4 -o objects.json
    """
    import json
    from src.object_detection import ObjectDetector
    
    config = Config.load()
    detector = ObjectDetector(config)
    
    console.print(f"Detecting objects in: {video_path}")
    
    frame_results, tracks = detector.detect_and_track(video_path)
    summary = detector.summarize_detections(frame_results)
    
    console.print(f"\n[green]Found {summary['unique_objects']} object types[/green]")
    console.print("\nTop objects:")
    for obj, count in list(summary['object_counts'].items())[:10]:
        console.print(f"  • {obj}: {count}")
    
    # Save results
    with open(output, 'w') as f:
        json.dump(summary, f, indent=2)
    
    console.print(f"\nResults saved to: {output}")


@cli.command()
@click.argument('image_path', type=click.Path(exists=True))
@click.option('--name', '-n', type=str, required=True, help='Person name')
def add_face(image_path, name):
    """
    Add a known face for recognition.
    
    Example:
        python -m cli add-face john.jpg -n "John Smith"
    """
    from src.face_recognition_module import FaceProcessor
    
    config = Config.load()
    processor = FaceProcessor(config)
    
    console.print(f"Adding face for: {name}")
    
    success = processor.add_known_face(name, image_path)
    
    if success:
        console.print(f"[green]✓ Face added for {name}[/green]")
    else:
        console.print(f"[red]Failed to add face. Check the image.[/red]")


@cli.command()
@click.option('--port', '-p', type=int, default=7860, help='Port number')
@click.option('--share', is_flag=True, help='Create public link')
def web(port, share):
    """
    Launch the web interface.
    
    Example:
        python -m cli web --port 7860
    """
    console.print("[bold blue]Launching Web Interface...[/bold blue]")
    
    # Import and run web app
    from web.app import create_app
    
    app = create_app()
    app.launch(server_port=port, share=share)


@cli.command()
@click.argument('output_dir', type=click.Path(exists=True))
@click.option('--screenplay', is_flag=True, help='Generate audio for screenplay.txt')
@click.option('--narrative', is_flag=True, help='Generate audio for narrative.txt')
@click.option('--transcript', is_flag=True, help='Generate audio for transcript.txt')
@click.option('--subtitles', is_flag=True, help='Generate audio for subtitles.srt')
@click.option('--all', 'all_outputs', is_flag=True, help='Generate audio for all supported outputs')
def generate_audio(output_dir, screenplay, narrative, transcript, subtitles, all_outputs):
    """Generate MP3 audio files from existing text outputs in an analysis folder.

    Example:
        python -m cli generate-audio outputs/my_run --all
    """
    output_path = Path(output_dir)
    audio_dir = output_path / 'audio_outputs'
    audio_dir.mkdir(parents=True, exist_ok=True)

    gen = TextToAudioGenerator()

    targets = []
    if all_outputs or screenplay:
        targets.append('screenplay')
    if all_outputs or narrative:
        targets.append('narrative')
    if all_outputs or transcript:
        targets.append('transcript')
    if all_outputs or subtitles:
        targets.append('subtitles')

    if not targets:
        console.print('[yellow]No targets selected. Use --all or one of the flags.[/yellow]')
        return

    console.print(f'[blue]Generating audio for:[/blue] {targets}')

    for t in targets:
        try:
            if t == 'screenplay':
                text = (output_path / 'screenplay.txt').read_text(encoding='utf-8')
                out = audio_dir / 'screenplay.mp3'
                gen.text_to_mp3(text, str(out))
            elif t == 'narrative':
                text = (output_path / 'narrative.txt').read_text(encoding='utf-8')
                out = audio_dir / 'narrative.mp3'
                gen.text_to_mp3(text, str(out))
            elif t == 'transcript':
                text = (output_path / 'transcript.txt').read_text(encoding='utf-8')
                out = audio_dir / 'transcript.mp3'
                gen.text_to_mp3(text, str(out))
            elif t == 'subtitles':
                srt = (output_path / 'subtitles.srt').read_text(encoding='utf-8')
                out = audio_dir / 'subtitles.mp3'
                gen.srt_to_mp3(srt, str(out))

            console.print(f'[green]✓ Generated:[/green] {out}')
        except Exception as e:
            console.print(f'[red]Failed to generate {t}: {e}[/red]')


@cli.command()
def info():
    """Show system information and capabilities."""
    import torch
    
    console.print("[bold blue]System Information[/bold blue]\n")
    
    # Python
    console.print(f"Python: {sys.version}")
    
    # PyTorch
    console.print(f"PyTorch: {torch.__version__}")
    console.print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        console.print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        console.print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Check modules
    console.print("\n[bold]Module Status:[/bold]")
    
    modules = [
        ('openai-whisper', 'whisper'),
        ('faster-whisper', 'faster_whisper'),
        ('pyannote.audio', 'pyannote.audio'),
        ('ultralytics', 'ultralytics'),
        ('face_recognition', 'face_recognition'),
        ('scenedetect', 'scenedetect'),
        ('gradio', 'gradio'),
    ]
    
    for name, import_name in modules:
        try:
            __import__(import_name)
            console.print(f"  [green]✓[/green] {name}")
        except ImportError:
            console.print(f"  [red]✗[/red] {name} (not installed)")


if __name__ == '__main__':
    cli()
