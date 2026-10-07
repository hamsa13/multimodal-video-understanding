"""
Text-to-Audio utilities.

Converts generated text artifacts (screenplay, narrative, transcript, subtitles)
to MP3 audio files.
"""

from __future__ import annotations

import re
from pathlib import Path


class TextToAudioGenerator:
    """Generate MP3 files from text artifacts using Coqui/gTTS/espeak fallbacks.

    The offline fallback uses `espeak` -> `ffmpeg`. Parameters for voice, speed,
    pitch and bitrate are configurable via constructor arguments.
    """

    def __init__(
        self,
        lang: str = "en",
        tld: str = "com",
        # espeak tuning
        espeak_voice: str = "en-us+m3",
        espeak_speed: int = 140,
        espeak_pitch: int = 70,
        # ffmpeg/encoding
        ffmpeg_bitrate: str = "192k",
        ffmpeg_samplerate: int = 24000,
        coqui_model: str | None = None,
    ):
        self.lang = lang
        self.tld = tld
        self.espeak_voice = espeak_voice
        self.espeak_speed = espeak_speed
        self.espeak_pitch = espeak_pitch
        self.ffmpeg_bitrate = ffmpeg_bitrate
        self.ffmpeg_samplerate = ffmpeg_samplerate
        self.coqui_model = coqui_model or "tts_models/en/vctk/vits"

    def text_to_mp3(self, text: str, output_path: str) -> str:
        """Convert plain text to MP3.

        Args:
            text: Input text
            output_path: Target MP3 path

        Returns:
            Saved MP3 path as string
        """
        cleaned = self._normalize_text(text)
        if not cleaned:
            raise ValueError("No valid text content to convert to audio")

        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        # Prefer Coqui TTS (higher-quality, free) if available
        try:
            from TTS.api import TTS as CoquiTTS
            # Prefer a compact, high-quality English model; this will download on first use
            # Model choices: 'tts_models/en/vctk/vits' (multi-speaker, high quality)
            model_name = getattr(self, 'coqui_model', 'tts_models/en/vctk/vits')
            tts = CoquiTTS(model_name)
            # Synchronous write to file
            tts.tts_to_file(text=cleaned, file_path=str(target))
            return str(target)
        except Exception:
            # Next fallback: gTTS (online, usually better than espeak)
            try:
                from gtts import gTTS
                tts = gTTS(text=cleaned, lang=self.lang, tld=self.tld)
                tts.save(str(target))
                return str(target)
            except Exception:
                # Final fallback: system TTS via espeak + ffmpeg (offline, lower quality)
                try:
                    import subprocess

                    wav_tmp = str(target.with_suffix(".wav"))

                    espeak_cmd = [
                        "espeak",
                        "-w",
                        wav_tmp,
                        "-v",
                        str(self.espeak_voice),
                        "-s",
                        str(self.espeak_speed),
                        "-p",
                        str(self.espeak_pitch),
                    ]

                    # Use espeak via stdin to avoid shell length limits
                    subprocess.run(espeak_cmd, input=cleaned.encode("utf-8"), check=True)

                    # Convert to mp3 via ffmpeg with tuned bitrate/samplerate
                    ffmpeg_cmd = [
                        "ffmpeg",
                        "-y",
                        "-i",
                        wav_tmp,
                        "-ar",
                        str(self.ffmpeg_samplerate),
                        "-codec:a",
                        "libmp3lame",
                        "-b:a",
                        str(self.ffmpeg_bitrate),
                        str(target),
                    ]
                    subprocess.check_call(ffmpeg_cmd)

                    # remove wav
                    try:
                        Path(wav_tmp).unlink()
                    except Exception:
                        pass
                    return str(target)
                except Exception as e:
                    raise RuntimeError(
                        "No viable TTS backend available (Coqui TTS, gTTS, or espeak+ffmpeg required)"
                    ) from e

    def srt_to_mp3(self, srt_text: str, output_path: str) -> str:
        """Convert SRT subtitle content to MP3 by stripping timestamps/indexes."""
        plain_text = self._srt_to_plain_text(srt_text)
        return self.text_to_mp3(plain_text, output_path)

    def _normalize_text(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""

        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _srt_to_plain_text(self, srt_text: str) -> str:
        lines = (srt_text or "").splitlines()
        out = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            if stripped.isdigit():
                continue

            if "-->" in stripped:
                continue

            out.append(stripped)

        return " ".join(out)
