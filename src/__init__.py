"""
Video Scene Understanding System
A complete multimodal video analysis pipeline
"""

__version__ = "1.0.0"
__author__ = "Team Ashwatthama"

from .pipeline import VideoPipeline
from .config import Config

__all__ = ["VideoPipeline", "Config", "__version__"]
