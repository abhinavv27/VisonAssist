"""
Phase 4: Demo & Presentation Readiness Test Suite
=================================================
Validates Phase 4 deliverables:
- Bilingual Response Generator (English / Hindi guidance).
- Streamlit Dashboard Section 20 styling and status badges.
- 6-Step Judge Demo Rehearsal Runner.
- Backup Demo Video generation pipeline.
"""

from pathlib import Path
import tempfile

from intelligence.response_generator import ResponseGenerator
from scripts.rehearse_demo import DemoRehearsalRunner
from scripts.record_backup_demo import generate_backup_demo


class TestPhase4MultilingualAndSpeech:
    """Validates multilingual translation and audio guidance payloads."""

    def test_multilingual_obstacle_guidance(self):
        gen = ResponseGenerator()
        item = {
            "type": "obstacle",
            "candidate": {
                "object": "stairs",
                "distance": 2.0,
                "position_desc": "ahead"
            },
            "is_critical": False,
            "priority": "HIGH"
        }

        # English
        res_en = gen.generate(item, language="en")
        assert "Stairs" in res_en["text"]
        assert "ahead" in res_en["text"]

        # Hindi
        res_hi = gen.generate(item, language="hi")
        assert "सीढ़ियाँ" in res_hi["text"]
        assert "मीटर" in res_hi["text"]

    def test_multilingual_safety_alert_collision(self):
        gen = ResponseGenerator()
        item = {
            "type": "safety_alert",
            "candidate": {
                "object": "chair",
                "distance": 0.8,
                "position_desc": "ahead"
            },
            "is_critical": True,
            "priority": "HIGH"
        }

        # English emergency alert
        res_en = gen.generate(item, language="en")
        assert "Warning" in res_en["text"]
        assert "less than one metre" in res_en["text"]

        # Hindi emergency alert
        res_hi = gen.generate(item, language="hi")
        assert "सावधान" in res_hi["text"]
        assert "कुर्सी" in res_hi["text"]

    def test_multilingual_quick_look(self):
        gen = ResponseGenerator()
        item = {
            "type": "quick_look",
            "primary": {"object": "chair", "position_desc": "ahead"},
            "secondary": {"object": "door", "position_desc": "on your right"}
        }

        res_en = gen.generate(item, language="en")
        assert "chair" in res_en["text"]
        assert "door" in res_en["text"]

        res_hi = gen.generate(item, language="hi")
        assert "कुर्सी" in res_hi["text"]
        assert "दरवाजा" in res_hi["text"]


class TestPhase4DemoAutomation:
    """Validates automated judging demo runner and backup recording."""

    def test_six_step_demo_rehearsal_execution(self):
        # Run silent, fast rehearsal
        runner = DemoRehearsalRunner(pace_factor=0.01, speak_audio=False)
        success = runner.run()

        assert success is True
        assert len(runner.stage_results) == 6
        # Confirm all 6 stages marked as passed
        for name, passed, notes in runner.stage_results:
            assert passed is True, f"Stage failed: {name} ({notes})"

    def test_backup_demo_video_rendering(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "test_backup.mp4"
            ok = generate_backup_demo(
                output_path=out_file,
                total_seconds=0.5,
                fps=10
            )
            assert ok is True
            assert out_file.exists()
            assert out_file.stat().st_size > 1000


class TestPhase4UIThemeSpecification:
    """Validates UI color definitions and Section 20 tokens."""

    def test_dashboard_tokens_and_status_strip(self):
        dashboard_code = Path("ui/dashboard.py").read_text(encoding="utf-8")

        # Report styling tokens
        assert "#101828" in dashboard_code
        assert "status-strip" in dashboard_code
        assert "audio-banner" in dashboard_code
        assert "pulse-red" in dashboard_code
        assert "card-high" in dashboard_code
        assert "card-med" in dashboard_code
        assert "card-low" in dashboard_code

        # Check status strip indicators
        assert "Camera:" in dashboard_code
        assert "Detection:" in dashboard_code
        assert "OCR:" in dashboard_code
        assert "TTS:" in dashboard_code
        assert "Multilingual Speech Guidance" in dashboard_code
