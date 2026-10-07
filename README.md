# 🎬 Video Scene Understanding

**Multimodal Video Understanding System** - Complete pipeline for extracting structured information from videos and generating cinematic text descriptions.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![CUDA](https://img.shields.io/badge/CUDA-Optional-orange.svg)

> **Team Ashwatthama** | CRRAO Internal Hackathon 2026

## Overview

This system processes video content through multiple AI modules to:
- 🎤 **Transcribe speech** with word-level timestamps
- 👥 **Identify speakers** (who said what when)
- 👤 **Detect and recognize faces** in frames
- 🎬 **Detect scene changes** automatically
- 📦 **Detect and track objects** using YOLO
- 🏃 **Recognize human actions** (running, talking, walking, etc.)
- 📝 **Generate narrative text** using LLMs (GPT-4, Claude, Gemini)

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                       VIDEO INPUT                              │
└─────────────────────────┬──────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
┌─────────────────┐ ┌───────────┐ ┌──────────────────┐
│ Frame Extraction│ │   Audio   │ │  Scene Detection │
│  (OpenCV/FFmpeg)│ │ Extraction│ │   (PySceneDetect)│
└────────┬────────┘ └─────┬─────┘ └────────┬─────────┘
         │                │                │
    ┌────┴────┐      ┌────┴────┐           │
    ▼         ▼      ▼         ▼           │
┌───────┐ ┌──────┐ ┌──────┐ ┌─────────┐    │
│ Object│ │ Face │ │Speech│ │ Speaker │    │
│  Det. │ │ Det. │ │ Rec. │ │  Diar.  │    │
│(YOLO) │ │(MTCNN│ │(Whis)│ │(pyannote│    │
└───┬───┘ └──┬───┘ └──┬───┘ └────┬────┘    │
    │        │        │          │         │
    └────────┴────────┴──────────┴─────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │   Multimodal Fusion    │
         │  (Temporal Alignment)  │
         └───────────┬────────────┘
                     │
                     ▼
         ┌────────────────────────┐
         │   LLM Text Generator   │
         │ (GPT-4/Claude/Gemini)  │
         └───────────┬────────────┘
                     │
                     ▼
         ┌────────────────────────┐
         │   Structured Output    │
         │ (JSON, SRT, Narrative) │
         └────────────────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
# Clone the repository
git clone https://github.com/yourusername/video-scene-understanding.git
cd video-scene-understanding

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Set Up API Keys

Create a `.env` file from the example:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```env
# Required for LLM text generation
OPENAI_API_KEY=sk-your-openai-key

# Required for speaker diarization
HF_TOKEN=hf_your_huggingface_token

# Optional
ANTHROPIC_API_KEY=sk-ant-your-key
GOOGLE_API_KEY=your-google-key
```

**Getting API Keys:**
- OpenAI: https://platform.openai.com/api-keys
- HuggingFace: https://huggingface.co/settings/tokens (accept pyannote terms)
- Anthropic: https://console.anthropic.com/
- Google: https://makersuite.google.com/app/apikey

### 3. Run Analysis

**CLI:**
```bash
# Full analysis
python cli.py analyze path/to/video.mp4

# Quick transcription only
python cli.py transcribe path/to/video.mp4

# Scene detection only
python cli.py detect-scenes path/to/video.mp4
```

**Web Interface:**
```bash
python cli.py web
# Open http://localhost:7860
```

## Usage Guide

### Command Line Interface

```bash
# Full analysis with all features
python cli.py analyze video.mp4 -o output/

# Generate audio from selected text outputs
python cli.py analyze video.mp4 \
  --audio-screenplay \
  --audio-narrative \
  --audio-subtitles \
  --audio-transcript

# Skip face detection (faster)
python cli.py analyze video.mp4 --skip-faces

# Skip action recognition (faster)
python cli.py analyze video.mp4 --skip-actions

# Without LLM text generation
python cli.py analyze video.mp4 --no-text

# Map speaker IDs to names
python cli.py analyze video.mp4 \
  --speaker-names '{"SPEAKER_00":"Alice","SPEAKER_01":"Bob"}'

# Quick transcription
python cli.py transcribe video.mp4

# Scene detection
python cli.py detect-scenes video.mp4

# Object detection with tracking
python cli.py detect-objects video.mp4 --track

# Add known face for recognition
python cli.py add-face "John Doe" path/to/john.jpg

# Check system info
python cli.py info
```

### Web Interface

Launch the Gradio web interface:

```bash
python cli.py web --port 7860 --share
```

Features:
- Upload video files
- Configure analysis options
- View transcript, scene data, narrative
- Download results

### Python API

```python
from src.config import Config
from src.pipeline import VideoPipeline

# Load configuration
config = Config.load()

# Create pipeline
pipeline = VideoPipeline(config)

# Process video
result = pipeline.process(
    video_path="video.mp4",
    output_dir="output/",
    speaker_names={"SPEAKER_00": "Alice"},
    skip_faces=False,
    skip_actions=False,
    generate_text=True
)

# Access results
print(result.analysis.full_transcript)
print(result.generated_narrative)

# Get scene data
for scene in result.analysis.scenes:
    print(f"Scene {scene.scene_number}: {scene.start_time}s - {scene.end_time}s")
    print(f"  Speakers: {scene.speakers}")
    print(f"  Objects: {scene.objects}")
```

### Known Faces

To recognize specific people in videos:

```bash
# Create directory structure
mkdir -p known_faces/JohnDoe
mkdir -p known_faces/JaneSmith

# Add face images (clear frontal shots work best)
cp john_photo1.jpg known_faces/JohnDoe/
cp john_photo2.jpg known_faces/JohnDoe/
cp jane_photo.jpg known_faces/JaneSmith/

# Or use the CLI
python cli.py add-face "John Doe" john_photo.jpg
```

## Configuration

Edit `config/config.yaml` to customize:

```yaml
# Video processing
video:
  fps: 1              # Frames per second to extract
  max_dimension: 1920 # Resize large videos
  format: jpg         # Frame format

# Speech recognition
speech:
  model: base         # tiny, base, small, medium, large
  language: null      # Auto-detect or specify: en, es, etc.

# Object detection
object_detection:
  model: yolov8n      # yolov8n, yolov8s, yolov8m, yolov8l
  confidence: 0.5     # Detection threshold
  tracking: true      # Enable object tracking

# LLM text generation
llm:
  provider: openai
  model: gpt-4o-mini  # or gpt-4, claude-3-opus, etc.
  temperature: 0.7
```

## Output Formats

### JSON Structure

```json
{
  "duration": 120.5,
  "scenes": [
    {
      "scene_number": 1,
      "start_time": 0.0,
      "end_time": 25.3,
      "dialogue": [
        {
          "speaker": "Alice",
          "text": "Hello everyone!",
          "start": 1.2,
          "end": 2.5
        }
      ],
      "speakers": ["Alice", "Bob"],
      "objects": ["chair", "table", "laptop"],
      "faces": ["Alice", "Unknown_1"],
      "actions": ["talking", "sitting"]
    }
  ]
}
```

### SRT Subtitles

```srt
1
00:00:01,200 --> 00:00:02,500
[Alice] Hello everyone!

2
00:00:03,100 --> 00:00:05,800
[Bob] Welcome to the meeting.
```

### Generated Narrative

```
The scene opens in a modern office space. Alice sits at a conference 
table, her laptop open before her. "Hello everyone!" she says warmly,
looking around the room...
```

### Generated Audio

When audio options are enabled, the pipeline also creates:

- `audio_outputs/screenplay.mp3`
- `audio_outputs/narrative.mp3`
- `audio_outputs/transcript.mp3`
- `audio_outputs/subtitles.mp3`

## Models Used

| Component | Model | Size | Memory |
|-----------|-------|------|--------|
| Speech Recognition | Whisper base | 142MB | ~1GB VRAM |
| Speaker Diarization | pyannote 3.1 | 50MB | ~500MB VRAM |
| Face Detection | MTCNN | 2MB | ~100MB VRAM |
| Face Recognition | DeepFace (VGG-Face) | 500MB | ~500MB VRAM |
| Object Detection | YOLOv8n | 6MB | ~500MB VRAM |
| Action Recognition | SlowFast | 1GB | ~2GB VRAM |
| Text Generation | GPT-4o-mini | API | N/A |

**Total VRAM**: ~4-5GB for full pipeline (fits on RTX 3050 6GB)

## Hardware Requirements

**Minimum:**
- CPU: 4 cores
- RAM: 8GB
- Storage: 10GB
- GPU: Not required (CPU mode available)

**Recommended:**
- CPU: 8+ cores
- RAM: 16GB+
- Storage: 50GB+ SSD
- GPU: NVIDIA with 6GB+ VRAM (RTX 3050 or better)

**Optimal:**
- CPU: 8+ cores
- RAM: 32GB+
- GPU: NVIDIA RTX 3090/4080+ (24GB VRAM)

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Use smaller models in config.yaml
speech:
  model: tiny  # Instead of base/small

object_detection:
  model: yolov8n  # Smallest model
```

**2. pyannote Access Error**
- Go to https://huggingface.co/pyannote/speaker-diarization-3.1
- Accept the user agreement
- Set `HF_TOKEN` in your `.env`

**3. FFmpeg Not Found**
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows - download from ffmpeg.org
```

**4. Slow Processing**
- Enable GPU: Install PyTorch with CUDA
- Use faster models (whisper tiny, yolov8n)
- Skip unnecessary modules (--skip-faces, --skip-actions)

### Logs

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Project Structure

```
video-scene-understanding/
├── cli.py                 # Command-line interface
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── config/
│   └── config.yaml       # Configuration file
├── src/
│   ├── __init__.py
│   ├── config.py         # Configuration loader
│   ├── pipeline.py       # Main orchestrator
│   ├── video_processor.py    # Frame/audio extraction
│   ├── speech_recognition.py # Whisper integration
│   ├── speaker_diarization.py # Speaker identification
│   ├── face_recognition_module.py # Face detection
│   ├── scene_detection.py    # Scene changes
│   ├── object_detection.py   # YOLO detection
│   ├── action_recognition.py # Action recognition
│   ├── multimodal_fusion.py  # Data fusion
│   └── text_generator.py     # LLM integration
├── web/
│   ├── __init__.py
│   └── app.py            # Gradio web interface
├── known_faces/          # Face recognition database
└── output/               # Analysis results
```

## API Reference

### VideoPipeline

```python
class VideoPipeline:
    def process(
        self,
        video_path: str,
        output_dir: str = "output/",
        speaker_names: dict = None,
        skip_faces: bool = False,
        skip_actions: bool = False,
        generate_text: bool = True,
        audio_outputs: List[str] = None,
        progress_callback: callable = None
    ) -> PipelineResult
```

### VideoAnalysis

```python
@dataclass
class VideoAnalysis:
    duration: float
    scenes: List[SceneData]
    full_transcript: str
    all_speakers: List[str]
    all_objects: List[str]
    
    def to_dict(self) -> Dict
    def to_json(self) -> str
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `python -m pytest tests/`
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) - Speaker diarization
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - Object detection
- [DeepFace](https://github.com/serengil/deepface) - Face recognition
- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) - Scene detection
- [Gradio](https://gradio.app/) - Web interface

---

**Built with ❤️ by Team Ashwatthama**
