import io
import os
import subprocess
import tempfile
import numpy as np
import torch
from gtts import gTTS
from rapidfuzz import fuzz
from transformers import pipeline

from config import (
    CONFUSION_GROUPS,
    FFMPEG_TIMEOUT_S,
    HIGH_ACOUSTIC,
    HIGH_CONFIDENCE,
    MAX_AUDIO_SECONDS,
    MAX_DECODE_TOKENS,
    MAX_DTW_FRAMES,
    MIN_PAUSE_MS,
    MODERATE_ACOUSTIC,
    MODERATE_CONFIDENCE,
    SAMPLE_RATE,
    SPOKEN_DROPPED_ENDINGS,
    SPOKEN_WORD_MATCH_THRESHOLD,
    VOWEL_LENGTH_PAIRS,
    WORD_MATCH_THRESHOLD,
)
def generate_reference_audio(text: str) -> bytes:
    cleaned = text.strip()
    if not cleaned:
        return b""

    buffer = io.BytesIO()
    tts = gTTS(text=cleaned, lang="ta")
    tts.write_to_fp(buffer)
    return buffer.getvalue()
def load_asr_pipeline(model_name: str):
    device = 0 if torch.cuda.is_available() else -1
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model=model_name,
        device=device,
    )
    # Greedy decoding + token cap prevents multi-minute CPU stalls and repeat loops
    gen_cfg = getattr(asr_pipe.model, "generation_config", None)
    if gen_cfg is not None:
        gen_cfg.num_beams = 1
        gen_cfg.do_sample = False
        gen_cfg.max_length = None
        gen_cfg.max_new_tokens = MAX_DECODE_TOKENS
        gen_cfg.no_repeat_ngram_size = 6
    return asr_pipe
def _decode_file_pcm(path: str) -> np.ndarray:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        path,
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(SAMPLE_RATE),
        "-ac",
        "1",
        "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=True, timeout=FFMPEG_TIMEOUT_S)
    except subprocess.TimeoutExpired as err:
        raise RuntimeError("Audio decoding timed out. Please use a shorter clip.") from err
    except subprocess.CalledProcessError as err:
        detail = err.stderr.decode(errors="ignore").strip()
        raise RuntimeError(f"FFmpeg audio decode failed: {detail}") from err

    if not proc.stdout:
        return np.array([], dtype=np.float32)

    return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
def decode_audio_pcm(audio_bytes: bytes, suffix: str = ".wav") -> np.ndarray:
    if not audio_bytes:
        return np.array([], dtype=np.float32)

    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        return _decode_file_pcm(tmp_path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
def _frame_signal(samples: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    count = 1 + (samples.size - frame_len) // hop
    starts = np.arange(count) * hop
    indices = starts[:, None] + np.arange(frame_len)[None, :]
    return samples[indices]
def _trim_silence(
    samples: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    pad_s: float = 0.15,
) -> np.ndarray:
    frame_len = int(0.025 * sample_rate)
    hop = int(0.010 * sample_rate)
    if samples.size < frame_len:
        return samples

    frames = _frame_signal(samples, frame_len, hop)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    peak = float(np.max(rms))
    if peak < 0.008:
        return samples

    active_idx = np.flatnonzero(rms > max(0.008, peak * 0.12))
    if active_idx.size == 0:
        return samples

    pad = int(pad_s * sample_rate)
    start = max(0, int(active_idx[0]) * hop - pad)
    end = min(samples.size, int(active_idx[-1]) * hop + frame_len + pad)
    return samples[start:end]

def transcribe_audio(asr_pipe, audio_bytes: bytes, suffix: str = ".wav") -> str:
    if not audio_bytes:
        return ""

    samples = decode_audio_pcm(audio_bytes, suffix=suffix)
    if samples.size == 0:
        return ""

    trimmed = _trim_silence(samples)
    max_samples = MAX_AUDIO_SECONDS * SAMPLE_RATE
    if trimmed.size > max_samples:
        trimmed = trimmed[:max_samples]

    audio_array = np.ascontiguousarray(trimmed, dtype=np.float32)

    # Pass decoded PCM directly so Whisper runs once without a second FFmpeg call
    with torch.inference_mode():
        result = asr_pipe({"array": audio_array, "sampling_rate": SAMPLE_RATE})

    return " ".join(result.get("text", "").strip().split())
def analyze_fluency(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> dict:
    frame_len = int(0.025 * sample_rate)
    hop = int(0.010 * sample_rate)

    if samples.size < frame_len:
        return {
            "pause_count": 0,
            "speech_duration_s": 0.0,
            "fluency_status": "Too short to measure",
        }

    frames = _frame_signal(samples, frame_len, hop)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    peak = float(np.max(rms))

    if peak < 0.008:
        return {
            "pause_count": 0,
            "speech_duration_s": 0.0,
            "fluency_status": "Silent or very low volume",
        }

    active = rms > max(0.008, peak * 0.12)
    active_idx = np.flatnonzero(active)
    start, end = int(active_idx[0]), int(active_idx[-1])
    speech_duration_s = round(((end - start + 1) * hop) / sample_rate, 2)

    min_gap_frames = max(1, int(MIN_PAUSE_MS / 10))
    pause_count = 0
    silent_run = 0

    for is_speech in active[start : end + 1]:
        if is_speech:
            if silent_run >= min_gap_frames:
                pause_count += 1
            silent_run = 0
        else:
            silent_run += 1

    if pause_count == 0:
        status = "Smooth / Continuous"
    elif pause_count == 1:
        status = "1 mid-utterance pause detected"
    else:
        status = f"{pause_count} pauses detected (choppy pacing)"

    return {
        "pause_count": pause_count,
        "speech_duration_s": speech_duration_s,
        "fluency_status": status,
    }
def _build_mel_filterbank(sample_rate: int, n_fft: int, n_mels: int = 26) -> np.ndarray:
    low_mel = 2595.0 * np.log10(1.0 + 80.0 / 700.0)
    high_mel = 2595.0 * np.log10(1.0 + (sample_rate / 2.0) / 700.0)
    mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    bins = np.clip(np.floor((n_fft + 1) * hz_points / sample_rate).astype(int), 0, n_fft // 2)

    fbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = int(bins[m - 1]), int(bins[m]), int(bins[m + 1])
        if center > left:
            for k in range(left, center):
                fbank[m - 1, k] = (k - left) / (center - left)
        if right > center:
            for k in range(center, right):
                fbank[m - 1, k] = (right - k) / (right - center)
        if left == center == right:
            fbank[m - 1, center] = 1.0
    return fbank
def _extract_mfcc(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    frame_len = int(0.025 * sample_rate)
    hop = int(0.010 * sample_rate)
    n_fft = 512
    n_mels = 26

    if samples.size < frame_len:
        return np.empty((0, 12), dtype=np.float32)

    samples = _trim_silence(samples, sample_rate=sample_rate, pad_s=0.0)
    if samples.size < frame_len:
        return np.empty((0, 12), dtype=np.float32)

    emphasized = np.append(samples[0], samples[1:] - 0.97 * samples[:-1])
    frames = _frame_signal(emphasized, frame_len, hop) * np.hamming(frame_len)
    pow_spec = (np.abs(np.fft.rfft(frames, n=n_fft)) ** 2) / n_fft

    fbank = _build_mel_filterbank(sample_rate, n_fft, n_mels)
    log_mel = np.log(np.maximum(pow_spec @ fbank.T, 1e-10))

    # Drop c0 energy and keep MFCCs 1..12 so microphone loudness does not skew comparison
    n = np.arange(n_mels)[:, None]
    k = np.arange(1, 13)[None, :]
    dct_basis = np.cos(np.pi * (n + 0.5) * k / n_mels)
    mfcc = log_mel @ dct_basis

    if mfcc.shape[0] > 1:
        mean = np.mean(mfcc, axis=0, keepdims=True)
        std = np.std(mfcc, axis=0, keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)
        mfcc = (mfcc - mean) / std

    if mfcc.shape[0] > MAX_DTW_FRAMES:
        idx = np.linspace(0, mfcc.shape[0] - 1, MAX_DTW_FRAMES, dtype=int)
        mfcc = mfcc[idx]

    return mfcc.astype(np.float32)
def compare_acoustics(
    teacher_samples: np.ndarray,
    student_samples: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
) -> dict:
    t_mfcc = _extract_mfcc(teacher_samples, sample_rate)
    s_mfcc = _extract_mfcc(student_samples, sample_rate)

    if t_mfcc.shape[0] < 2 or s_mfcc.shape[0] < 2:
        return {"acoustic_score": None, "acoustic_label": "Unavailable"}

    t_norm = t_mfcc / (np.linalg.norm(t_mfcc, axis=1, keepdims=True) + 1e-8)
    s_norm = s_mfcc / (np.linalg.norm(s_mfcc, axis=1, keepdims=True) + 1e-8)
    cost = 1.0 - np.clip(t_norm @ s_norm.T, -1.0, 1.0)

    # Symmetric DTW (weights 1, 1, 2) normalized by n + m for path-length invariance
    n, m = cost.shape
    dp = np.full((n, m), np.inf, dtype=np.float64)
    dp[0, 0] = 2.0 * cost[0, 0]

    for i in range(1, n):
        dp[i, 0] = dp[i - 1, 0] + cost[i, 0]
    for j in range(1, m):
        dp[0, j] = dp[0, j - 1] + cost[0, j]

    for i in range(1, n):
        row_cost = cost[i]
        prev_dp = dp[i - 1]
        curr_dp = dp[i]
        for j in range(1, m):
            c = row_cost[j]
            v1 = prev_dp[j] + c
            v2 = curr_dp[j - 1] + c
            v3 = prev_dp[j - 1] + 2.0 * c
            curr_dp[j] = v1 if v1 <= v2 and v1 <= v3 else (v2 if v2 <= v3 else v3)

    mean_dist = float(dp[n - 1, m - 1] / (n + m))
    score = round(max(0.0, min(100.0, (1.0 - mean_dist) * 100.0)), 1)

    if score >= HIGH_ACOUSTIC:
        label = "Close Match"
    elif score >= MODERATE_ACOUSTIC:
        label = "Moderate Match"
    else:
        label = "Distant"

    return {"acoustic_score": score, "acoustic_label": label}
def validate_tamil_script(text: str) -> dict:
    cleaned = text.strip()
    if not cleaned:
        return {
            "is_valid": False,
            "purity": 0.0,
            "has_latin": False,
            "message": "No speech or Tamil text detected.",
        }
    tamil_count = 0
    latin_count = 0
    other_alpha = 0

    for ch in cleaned:
        if ch.isspace() or not ch.isalnum():
            continue
        if "\u0B80" <= ch <= "\u0BFF":
            tamil_count += 1
        elif ("A" <= ch <= "Z") or ("a" <= ch <= "z"):
            latin_count += 1
        else:
            other_alpha += 1

    total = tamil_count + latin_count + other_alpha
    if total == 0:
        return {
            "is_valid": False,
            "purity": 0.0,
            "has_latin": False,
            "message": "Input contains no recognizable Tamil characters.",
        }

    purity = round((tamil_count / total) * 100.0, 1)
    has_latin = latin_count > 0

    if has_latin:
        return {
            "is_valid": False,
            "purity": purity,
            "has_latin": True,
            "message": "Non-Tamil Latin characters detected; only pure Tamil script is accepted.",
        }

    is_valid = tamil_count > 0 and purity >= 80.0
    message = "Valid Tamil script." if is_valid else "Insufficient Tamil script purity."
    return {
        "is_valid": is_valid,
        "purity": purity,
        "has_latin": False,
        "message": message,
    }
def segment_tamil_graphemes(text: str) -> list[str]:
    # Keep Tamil combining vowel signs and pulli attached to their base consonant
    graphemes: list[str] = []
    for ch in text.strip():
        if ch.isspace():
            continue
        is_combining = ch == "\u0B82" or ("\u0BBE" <= ch <= "\u0BCD") or ch == "\u0BD7"
        if is_combining and graphemes:
            graphemes[-1] += ch
        else:
            graphemes.append(ch)
    return graphemes
def _compact_tamil(text: str) -> str:
    return "".join(ch for ch in text if not ch.isspace())

def _normalize_spoken_tamil(text: str) -> str:
    # Normalize common conversational Tamil word-final diglossia drops
    normalized_words: list[str] = []
    for word in text.strip().split():
        base = word
        for ending in SPOKEN_DROPPED_ENDINGS:
            if len(base) > len(ending) + 1 and base.endswith(ending):
                base = base[: -len(ending)]
                break
        normalized_words.append(base)
    return "".join(normalized_words)


def evaluate_teacher(teacher_text: str, fluency: dict | None = None) -> dict:
    cleaned = " ".join(teacher_text.strip().split())
    validation = validate_tamil_script(cleaned)
    syllables = segment_tamil_graphemes(cleaned) if validation["is_valid"] else []
    fluency_info = fluency or {
        "pause_count": 0,
        "speech_duration_s": 0.0,
        "fluency_status": "Not measured",
    }

    return {
        "transcription": cleaned,
        "script_valid": validation["is_valid"],
        "purity": validation["purity"],
        "syllables": syllables,
        "status": "Ready as Benchmark" if validation["is_valid"] else "Invalid Benchmark",
        "message": validation["message"],
        "pause_count": fluency_info["pause_count"],
        "speech_duration_s": fluency_info["speech_duration_s"],
        "fluency_status": fluency_info["fluency_status"],
    }


def _analyze_pronunciation(
    spoken: str,
    target: str,
    similarity: float,
    diglossia_accepted: bool,
    acoustic: dict | None,
    fluency: dict | None,
) -> tuple[str, str]:
    compact_spoken = _compact_tamil(spoken)
    compact_target = _compact_tamil(target)
    target_graphemes = segment_tamil_graphemes(compact_target)
    spoken_graphemes = segment_tamil_graphemes(compact_spoken)
    syllable_guide = " · ".join(target_graphemes)

    notes: list[str] = []
    fixes: list[str] = []

    if compact_spoken == compact_target:
        notes.append("Exact Tamil articulation matching the Teacher's syllable structure.")
    elif diglossia_accepted:
        notes.append("Natural spoken Tamil (diglossia) ending accepted in conversational mode.")
        fixes.append(
            f"Acceptable in spoken Tamil. For formal literary delivery, pronounce all syllables of '{target}': {syllable_guide}."
        )
    elif similarity < MODERATE_CONFIDENCE:
        notes.append(f"Word mismatch detected: Student said '{spoken}' instead of '{target}'.")
        fixes.append(
            f"Listen to the Teacher's clip again and practice '{target}' slowly syllable by syllable: {syllable_guide}."
        )
    else:
        for group in CONFUSION_GROUPS:
            chars = group["chars"]
            target_counts = {c: compact_target.count(c) for c in chars}
            spoken_counts = {c: compact_spoken.count(c) for c in chars}
            if target_counts != spoken_counts and any(target_counts.values()):
                notes.append(group["description"])
                fixes.append(group["solution"])

        for sign, description, solution in VOWEL_LENGTH_PAIRS:
            if compact_target.count(sign) != compact_spoken.count(sign):
                notes.append(description)
                fixes.append(solution)

        diff = len(spoken_graphemes) - len(target_graphemes)
        if diff < 0:
            notes.append("Shortened or incomplete syllable delivery (possible spoken diglossia drop).")
            fixes.append(
                f"Articulate all {len(target_graphemes)} syllables completely without dropping endings: {syllable_guide}."
            )
        elif diff > 0:
            notes.append("Extra syllables or trailing sounds detected.")
            fixes.append(
                f"Speak '{target}' smoothly in one breath without extra pauses or trailing sounds: {syllable_guide}."
            )

        if not notes:
            notes.append("Minor syllable variation compared to the Teacher's pronunciation.")
            fixes.append(f"Replay the Teacher's audio and repeat each syllable cleanly: {syllable_guide}.")

    if acoustic and acoustic.get("acoustic_score") is not None:
        if acoustic["acoustic_score"] < MODERATE_ACOUSTIC and similarity >= MODERATE_CONFIDENCE:
            notes.append("Acoustic delivery (tone/stress) differed noticeably from the Teacher's audio.")
            fixes.append("Match the Teacher's syllable stress and speaking rhythm more closely.")

    if fluency and fluency.get("pause_count", 0) >= 2:
        notes.append(f"Audio pacing was choppy ({fluency['pause_count']} pauses detected).")
        fixes.append(f"Try saying '{target}' continuously without breaking between syllables.")

    if not fixes:
        fixes.append("No correction needed. Keep maintaining this clarity and pace.")

    return "; ".join(notes), " ".join(fixes)


def evaluate_pronunciation(
    spoken_text: str,
    target_text: str,
    mode: str = "literary",
    acoustic: dict | None = None,
    fluency: dict | None = None,
) -> dict:
    spoken = " ".join(spoken_text.strip().split())
    target = " ".join(target_text.strip().split())
    eval_mode = "spoken" if mode == "spoken" else "literary"
    target_check = validate_tamil_script(target)
    spoken_check = validate_tamil_script(spoken)

    acoustic_score = acoustic["acoustic_score"] if acoustic else None
    acoustic_label = acoustic["acoustic_label"] if acoustic else "Unavailable"
    pause_count = fluency["pause_count"] if fluency else 0
    speech_duration_s = fluency["speech_duration_s"] if fluency else 0.0
    fluency_status = fluency["fluency_status"] if fluency else "Not measured"

    if not target_check["is_valid"]:
        return {
            "transcription": spoken,
            "target": target,
            "mode": eval_mode,
            "wording_match": False,
            "similarity": 0.0,
            "pronunciation_status": "Invalid Benchmark",
            "characteristics": "Teacher recording did not produce a valid Tamil benchmark.",
            "solution": "Record or confirm a valid Tamil word or phrase from the Teacher first.",
            "syllable_breakdown": "",
            "confidence_score": 0.0,
            "confidence_label": "Low",
            "script_valid": spoken_check["is_valid"],
            "acoustic_score": acoustic_score,
            "acoustic_label": acoustic_label,
            "pause_count": pause_count,
            "speech_duration_s": speech_duration_s,
            "fluency_status": fluency_status,
        }

    syllable_breakdown = " · ".join(segment_tamil_graphemes(target))

    if not spoken_check["is_valid"]:
        return {
            "transcription": spoken,
            "target": target,
            "mode": eval_mode,
            "wording_match": False,
            "similarity": 0.0,
            "pronunciation_status": "Invalid Input",
            "characteristics": spoken_check["message"],
            "solution": f"Re-record clearly in pure Tamil matching '{target}' ({syllable_breakdown}).",
            "syllable_breakdown": syllable_breakdown,
            "confidence_score": 0.0,
            "confidence_label": "Low",
            "script_valid": False,
            "acoustic_score": acoustic_score,
            "acoustic_label": acoustic_label,
            "pause_count": pause_count,
            "speech_duration_s": speech_duration_s,
            "fluency_status": fluency_status,
        }

    # Compare compact Tamil text so spurious ASR spaces do not penalize clean speech
    compact_spoken = _compact_tamil(spoken)
    compact_target = _compact_tamil(target)
    raw_similarity = round(float(fuzz.ratio(compact_spoken, compact_target)), 1)

    diglossia_accepted = False
    if eval_mode == "spoken":
        spoken_sim = round(
            float(fuzz.ratio(_normalize_spoken_tamil(spoken), _normalize_spoken_tamil(target))),
            1,
        )
        similarity = max(raw_similarity, spoken_sim)
        match_threshold = SPOKEN_WORD_MATCH_THRESHOLD
        diglossia_accepted = (spoken_sim > raw_similarity) and (spoken_sim >= match_threshold)
    else:
        similarity = raw_similarity
        match_threshold = WORD_MATCH_THRESHOLD

    wording_match = similarity >= match_threshold
    confidence_score = round(similarity * (spoken_check["purity"] / 100.0), 1)

    if confidence_score >= HIGH_CONFIDENCE or diglossia_accepted:
        confidence_label = "High"
        status = "Accurate (Spoken)" if diglossia_accepted else "Accurate"
    elif confidence_score >= MODERATE_CONFIDENCE:
        confidence_label = "Moderate"
        status = "Partial / Rough"
    else:
        confidence_label = "Low"
        status = "Mismatch"

    characteristics, solution = _analyze_pronunciation(
        spoken,
        target,
        similarity,
        diglossia_accepted,
        acoustic,
        fluency,
    )
    return {
        "transcription": spoken,
        "target": target,
        "mode": eval_mode,
        "wording_match": wording_match,
        "similarity": similarity,
        "pronunciation_status": status,
        "characteristics": characteristics,
        "solution": solution,
        "syllable_breakdown": syllable_breakdown,
        "confidence_score": confidence_score,
        "confidence_label": confidence_label,
        "script_valid": True,
        "acoustic_score": acoustic_score,
        "acoustic_label": acoustic_label,
        "pause_count": pause_count,
        "speech_duration_s": speech_duration_s,
        "fluency_status": fluency_status,
    }