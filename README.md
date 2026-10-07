# VISTA — Multimodal Video Understanding

An end-to-end multimodal AI pipeline that transforms raw videos into structured, timestamped information by combining speech, speaker, scene, object, and action understanding.

Built as part of the **CRRAO Internal Hackathon 2026**.

---

## Overview

VISTA processes a video through multiple specialized AI components and combines their outputs along a shared timeline.

The system can:

- Transcribe speech with word-level timestamps
- Perform speaker diarization
- Detect scene boundaries
- Detect objects in video frames
- Recognize human actions
- Fuse multimodal information across timestamps
- Generate structured scene-level descriptions
- Produce transcripts, screenplays, and narrative summaries

The goal is to convert unstructured video into information that can be searched, analyzed, and interpreted programmatically.

---

## Pipeline

```text
                         Input Video
                              │
              ┌───────────────┴───────────────┐
              │                               │
           Video                           Audio
              │                               │
      ┌───────┼────────┐                Speech Recognition
      │       │        │                       │
   Scenes  Objects   Actions            Speaker Diarization
      │       │        │                       │
      └───────┴────────┴───────────────┬───────┘
                                       │
                              Multimodal Fusion
                                       │
                              Temporal Alignment
                                       │
                              Structured Scenes
                                       │
                              Llama 3.3 70B
                                 via Groq
                                       │
                         ┌─────────────┼─────────────┐
                         │             │             │
                    Transcript     Screenplay    Narrative
