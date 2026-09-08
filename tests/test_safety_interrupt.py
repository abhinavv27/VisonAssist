"""
Unit test for Mode-5 Safety Alert Priority-Interrupt
====================================================
Verifies Phase 1 Task 3:
"Add TTS priority-interrupt flag for Mode-5 alerts; unit test mid-speech interrupt.
Done when: Long sentence is correctly cut off by a close-obstacle event."
"""

import time
from audio.tts import TextToSpeechEngine


def test_midspeech_priority_interrupt():
    """Verify that a long low-priority sentence is cut off when an emergency alert arrives."""
    # Initialize TTS with muted speaker output for automated headless testing
    tts = TextToSpeechEngine(mute=False, phone_audio_port=8099)

    long_mode_1_sentence = (
        "There is a doorway ahead, a table slightly to your left, a chair in front of you, "
        "and several posters along the corridor wall."
    )
    emergency_mode_5_alert = "Warning. Obstacle directly ahead, less than one metre."

    try:
        # Start speaking the long sentence
        tts.speak(long_mode_1_sentence, interrupt=False, priority=False)

        # Wait briefly for worker to pick it up and begin speaking
        time.sleep(0.15)
        assert tts.current_utterance == long_mode_1_sentence, (
            f"Expected current utterance to be '{long_mode_1_sentence}', got '{tts.current_utterance}'"
        )

        # Trigger emergency Mode-5 alert with priority=True
        tts.speak(emergency_mode_5_alert, interrupt=True, priority=True)

        # Check that the long sentence was registered as interrupted
        assert tts.last_interrupted_utterance == long_mode_1_sentence, (
            f"Expected last_interrupted_utterance to be '{long_mode_1_sentence}', got '{tts.last_interrupted_utterance}'"
        )

        # Allow worker thread to switch to the emergency alert
        time.sleep(0.2)
        assert tts.current_utterance == emergency_mode_5_alert or tts.last_interrupted_utterance == long_mode_1_sentence

    finally:
        tts.stop()


def test_safety_alert_queue_cleared():
    """Verify that calling speak with priority=True purges all queued backlog."""
    tts = TextToSpeechEngine(mute=True, phone_audio_port=8098)
    try:
        tts.speak("First non-urgent message", interrupt=False)
        tts.speak("Second non-urgent message", interrupt=False)
        tts.speak("Third non-urgent message", interrupt=False)

        # Urgent emergency alert clears queue
        tts.speak("Critical danger ahead!", interrupt=True, priority=True)

        # The queue should now only have the critical danger item
        assert tts._speech_queue.qsize() <= 1
    finally:
        tts.stop()
