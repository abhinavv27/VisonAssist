"""
Audio guidance package for VisionAssist.
Provides offline, non-blocking text-to-speech output.
"""

from .tts import TextToSpeechEngine
from .audio_stream_server import AudioStreamServer, notify_phone_audio

__all__ = ["TextToSpeechEngine", "AudioStreamServer", "notify_phone_audio"]
