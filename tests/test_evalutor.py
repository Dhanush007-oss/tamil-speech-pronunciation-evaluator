import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import SAMPLE_RATE
from evaluator import (
    analyze_fluency,
    compare_acoustics,
    evaluate_pronunciation,
    evaluate_teacher,
    segment_tamil_graphemes,
    validate_tamil_script,
)

EXPECTED_VALIDATION_KEYS = {"is_valid", "purity", "has_latin", "message"}
EXPECTED_TEACHER_KEYS = {
    "transcription",
    "script_valid",
    "purity",
    "syllables",
    "status",
    "message",
    "pause_count",
    "speech_duration_s",
    "fluency_status",
}
EXPECTED_EVAL_KEYS = {
    "transcription",
    "target",
    "mode",
    "wording_match",
    "pronunciation_match",
    "similarity",
    "overall_score",
    "pronunciation_status",
    "characteristics",
    "solution",
    "coaching_steps",
    "syllable_breakdown",
    "confidence_score",
    "confidence_label",
    "script_valid",
    "acoustic_score",
    "acoustic_label",
    "pause_count",
    "speech_duration_s",
    "fluency_status",
}


def test_segment_tamil_graphemes():
    assert segment_tamil_graphemes("வாழைப்பழம்") == ["வா", "ழை", "ப்", "ப", "ழ", "ம்"]
    assert segment_tamil_graphemes("தமிழ்") == ["த", "மி", "ழ்"]
    assert segment_tamil_graphemes("   ") == []


def test_validate_tamil_script():
    pure = validate_tamil_script("வாழைப்பழம்")
    assert set(pure.keys()) == EXPECTED_VALIDATION_KEYS
    assert pure["is_valid"] is True
    assert pure["purity"] == 100.0
    assert pure["has_latin"] is False

    tanglish = validate_tamil_script("vazhaipazham")
    assert set(tanglish.keys()) == EXPECTED_VALIDATION_KEYS
    assert tanglish["is_valid"] is False
    assert tanglish["has_latin"] is True
    assert tanglish["purity"] == 0.0


def test_evaluate_teacher():
    valid = evaluate_teacher("வணக்கம்")
    assert set(valid.keys()) == EXPECTED_TEACHER_KEYS
    assert valid["script_valid"] is True
    assert valid["syllables"] == ["வ", "ண", "க்", "க", "ம்"]

    invalid = evaluate_teacher("")
    assert invalid["script_valid"] is False
    assert invalid["syllables"] == []


def test_evaluate_exact_and_spaced_match():
    exact = evaluate_pronunciation("வாழைப்பழம்", "வாழைப்பழம்")
    assert set(exact.keys()) == EXPECTED_EVAL_KEYS
    assert exact["wording_match"] is True
    assert exact["pronunciation_match"] is True
    assert exact["similarity"] == 100.0
    assert exact["pronunciation_status"] == "Accurate Pronunciation"
    assert exact["confidence_label"] == "High"

    spaced = evaluate_pronunciation("மர வெள்ளிக் கிழங்கு", "மரவெள்ளிக்கிழங்கு")
    assert spaced["wording_match"] is True
    assert spaced["pronunciation_match"] is True
    assert spaced["similarity"] == 100.0


def test_evaluate_accent_gap_on_matching_word():
    accented = evaluate_pronunciation(
        "வாழைப்பழம்",
        "வாழைப்பழம்",
        acoustic={"acoustic_score": 48.0, "acoustic_label": "Strong Accent Gap"},
    )
    assert set(accented.keys()) == EXPECTED_EVAL_KEYS
    assert accented["wording_match"] is True
    assert accented["pronunciation_match"] is False
    assert accented["overall_score"] == 68.8
    assert accented["pronunciation_status"] == "Right Word, Pronunciation Needs Work"
    assert accented["confidence_label"] == "Low"
    assert "accent" in accented["characteristics"].lower()
    assert len(accented["coaching_steps"]) >= 2
    assert "வா · ழை · ப் · ப · ழ · ம்" in accented["solution"]


def test_evaluate_phonetic_shift_with_solution():
    result = evaluate_pronunciation("வாளைப்பலம்", "வாழைப்பழம்")
    assert set(result.keys()) == EXPECTED_EVAL_KEYS
    assert result["script_valid"] is True
    assert result["wording_match"] is False
    assert result["pronunciation_match"] is False
    assert result["similarity"] == 73.3
    assert result["pronunciation_status"] == "Phonetic Shift Detected"
    assert "ல / ள / ழ" in result["characteristics"]
    assert "ல" in result["solution"] and "ழ" in result["solution"]


def test_evaluate_spoken_diglossia_mode():
    literary = evaluate_pronunciation("கட", "கடை", mode="literary")
    assert literary["wording_match"] is False
    assert literary["pronunciation_status"] == "Phonetic Shift Detected"

    spoken = evaluate_pronunciation("கட", "கடை", mode="spoken")
    assert spoken["wording_match"] is True
    assert spoken["pronunciation_match"] is True
    assert spoken["pronunciation_status"] == "Accurate Pronunciation (Spoken)"
    assert "diglossia" in spoken["characteristics"].lower()

    wrong_word = evaluate_pronunciation("வணக்கம்", "கடை", mode="spoken")
    assert wrong_word["wording_match"] is False
    assert wrong_word["pronunciation_status"] == "Word Mismatch"


def test_evaluate_different_word_mismatch():
    result = evaluate_pronunciation("வணக்கம்", "வாழைப்பழம்")
    assert result["wording_match"] is False
    assert result["pronunciation_match"] is False
    assert result["pronunciation_status"] == "Word Mismatch"
    assert "Different word spoken" in result["characteristics"]
    assert "வா · ழை · ப் · ப · ழ · ம்" in result["solution"]


def test_evaluate_shared_prefix_different_word():
    result = evaluate_pronunciation("வாழைக்காய்", "வாழைப் பழம்")
    assert result["wording_match"] is False
    assert result["pronunciation_match"] is False
    assert result["similarity"] == 48.2
    assert result["pronunciation_status"] == "Word Mismatch"
    assert result["confidence_label"] == "Low"
    assert "Different word spoken" in result["characteristics"]
    assert "ல / ள / ழ" not in result["characteristics"]


def test_evaluate_empty_and_tanglish():
    empty = evaluate_pronunciation("", "வாழைப்பழம்")
    assert empty["wording_match"] is False
    assert empty["pronunciation_match"] is False
    assert empty["confidence_score"] == 0.0
    assert empty["confidence_label"] == "Low"
    assert empty["pronunciation_status"] == "Invalid Input"

    tanglish = evaluate_pronunciation("vazhaipazham", "வாழைப்பழம்")
    assert tanglish["wording_match"] is False
    assert tanglish["pronunciation_match"] is False
    assert tanglish["confidence_score"] == 0.0
    assert tanglish["pronunciation_status"] == "Invalid Input"


def test_analyze_fluency_smooth_and_paused():
    t = np.linspace(0.0, 0.8, int(0.8 * SAMPLE_RATE), dtype=np.float32)
    smooth_wave = 0.5 * np.sin(2.0 * np.pi * 440.0 * t)
    smooth = analyze_fluency(smooth_wave)
    assert smooth["pause_count"] == 0
    assert smooth["fluency_status"] == "Smooth / Continuous"

    segment = 0.5 * np.sin(2.0 * np.pi * 440.0 * t[: int(0.35 * SAMPLE_RATE)])
    gap = np.zeros(int(0.35 * SAMPLE_RATE), dtype=np.float32)
    paused_wave = np.concatenate([segment, gap, segment])
    paused = analyze_fluency(paused_wave)
    assert paused["pause_count"] == 1


def test_compare_acoustics_same_and_different():
    t = np.linspace(0.0, 0.8, int(0.8 * SAMPLE_RATE), dtype=np.float32)
    chirp = 0.5 * np.sin(2.0 * np.pi * (200.0 + 1800.0 * t) * t).astype(np.float32)
    emb = np.eye(4, dtype=np.float32)

    identical = compare_acoustics(chirp, chirp.copy(), teacher_emb=emb, student_emb=emb.copy())
    assert identical["acoustic_score"] == 100.0
    assert identical["acoustic_label"] == "Native-Like Match"

    reverse_chirp = chirp[::-1].copy()
    different = compare_acoustics(chirp, reverse_chirp, teacher_emb=emb, student_emb=emb[::-1].copy())
    assert different["acoustic_score"] is not None
    assert different["acoustic_score"] < identical["acoustic_score"]