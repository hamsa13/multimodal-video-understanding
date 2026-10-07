Project Dependencies — What each package does

This file explains the purpose of each package listed in requirements.txt and how it is used in this video analysis project.

## Install

Install the project dependencies:

```bash
pip install -r requirements.txt
```

## Core

- **numpy**: Numerical arrays and fast vectorized operations used across video/frame processing and ML features.
- **opencv-python**: Read, manipulate and preprocess video frames; common image processing utilities (resizing, drawing, color conversions).
- **pillow**: Image I/O and small image manipulations (thumbnails, saving frames to disk).
- **tqdm**: Progress bars used in long-running loops (frame extraction, detection loops).

## Config & CLI

- **pyyaml**: Load/save project configuration files (YAML) such as model paths and thresholds.
- **python-dotenv**: Load `.env` environment variables (API keys like `GROQ_API_KEY`, `HF_TOKEN`).
- **click**: Lightweight CLI helpers to build command-line entrypoints.
- **rich**: Pretty console output (tables, progress, colors) used by the pipeline for readable logs.

## Video / Audio Processing

- **ffmpeg-python**: Programmatic FFmpeg wrapper used to extract audio from videos and transcode media.
- **moviepy**: High-level video processing utilities (clip trimming, saving thumbnails, composition).

## Speech Recognition

- **faster-whisper**: Local, fast Whisper implementation for offline speech-to-text transcription of audio tracks.

## Scene Detection

- **scenedetect[opencv]**: Detects shot/scene boundaries inside a video (used to split the video into scenes for per-scene analysis).

## Object Detection

- **ultralytics**: Provides YOLOv8 object detection models and APIs used to find and classify objects in frames.

## LLM & Tokenization

- **transformers**: Hugging Face transformers library for working with local or remote language models (fallback generation, prompt handling).
- **accelerate**: Utilities for efficient model execution and multi-GPU support.
- **sentencepiece**: Tokenizer library used by some Transformer models for tokenization.

## Groq LLM API

- **groq**: Official Groq SDK used to call the `llama-3.3-70b-versatile` model for high-quality AI summaries and narrative generation.

## Web Interface

- **gradio**: Simple web UI framework used to build the interactive demo (upload videos, show transcripts, summaries, and downloads).

## Entity Tracking

- **networkx**: Graph library used for representing relationships (entities, object co-occurrences, simple scene graphs) when analyzing spatial/temporal relations.

## Notes

- Most packages are free/open-source and run locally. `groq` is an API client (requires an API key set in `.env`).
- For production or large videos, a GPU is recommended for object detection and model acceleration.

If you want, I can also add a minimal `README.md` with usage examples and pinned versions in `requirements.txt` for reproducible installs.
