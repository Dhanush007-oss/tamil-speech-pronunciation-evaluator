from difflib import SequenceMatcher
import io
import os
import subprocess
import tempfile
import unicodedata

import numpy as np
import torch
from gtts import gTTS
from rapidfuzz import fuzz
from transformers import pipeline

from config import (
    ASR_NUM_BEAMS,
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
    TAMIL_WORD_SPELLING_MAP,
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
    # Small beam search recovers soft/accented Tamil onsets while token cap prevents CPU stalls
    gen_cfg = getattr(asr_pipe.model, "generation_config", None)
    if gen_cfg is not None:
        gen_cfg.num_beams = ASR_NUM_BEAMS
        gen_cfg.early_stopping = True
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
    pad_s: float = 0.30,
) -> np.ndarray:
    frame_len = int(0.025 * sample_rate)
    hop = int(0.010 * sample_rate)
    if samples.size < frame_len:
        return samples

    frames = _frame_signal(samples, frame_len, hop)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    peak = float(np.max(rms))
    if peak < 0.004:
        return samples

    # Low onset threshold preserves soft Tamil openings like 'வ' and 'ம'
    active_idx = np.flatnonzero(rms > max(0.004, peak * 0.05))
    if active_idx.size == 0:
        return samples

    pad = int(pad_s * sample_rate)
    start = max(0, int(active_idx[0]) * hop - pad)
    end = min(samples.size, int(active_idx[-1]) * hop + frame_len + pad)
    return samples[start:end]


def _prepare_asr_audio(
    samples: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    cushion_s: float = 0.20,
) -> np.ndarray:
    trimmed = _trim_silence(samples, sample_rate=sample_rate, pad_s=0.30 if cushion_s > 0 else 0.05)
    cushion_len = int(cushion_s * sample_rate)
    max_speech_samples = max(1, MAX_AUDIO_SECONDS * sample_rate - 2 * cushion_len)
    if trimmed.size > max_speech_samples:
        trimmed = trimmed[:max_speech_samples]

    centered = trimmed - float(np.mean(trimmed))
    peak_abs = float(np.max(np.abs(centered)))
    if peak_abs < 0.004:
        return np.array([], dtype=np.float32)

    normalized = centered * (0.90 / peak_abs)
    if cushion_len == 0:
        return np.ascontiguousarray(normalized, dtype=np.float32)

    cushion = np.zeros(cushion_len, dtype=np.float32)
    return np.ascontiguousarray(np.concatenate([cushion, normalized, cushion]), dtype=np.float32)


def transcribe_audio(asr_pipe, audio_bytes: bytes, suffix: str = ".wav") -> str:
    if not audio_bytes:
        return ""

    samples = decode_audio_pcm(audio_bytes, suffix=suffix)
    if samples.size == 0:
        return ""

    audio_array = _prepare_asr_audio(samples, cushion_s=0.20)
    if audio_array.size == 0:
        return ""

    # Pass decoded PCM directly so Whisper runs once without a second FFmpeg call
    with torch.inference_mode():
        result = asr_pipe({"array": audio_array, "sampling_rate": SAMPLE_RATE})

    return " ".join(result.get("text", "").strip().split())


def extract_phonetic_embeddings(asr_pipe, samples: np.ndarray) -> np.ndarray | None:
    if asr_pipe is None or samples.size == 0:
        return None

    audio_array = _prepare_asr_audio(samples, cushion_s=0.0)
    if audio_array.size < int(0.05 * SAMPLE_RATE):
        return None

    feature_extractor = getattr(asr_pipe, "feature_extractor", None)
    model = getattr(asr_pipe, "model", None)
    if feature_extractor is None or model is None or not hasattr(model, "get_encoder"):
        return None

    try:
        inputs = feature_extractor(
            audio_array,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(device=model.device, dtype=model.dtype)
        with torch.inference_mode():
            encoder_out = model.get_encoder()(input_features)
        hidden = encoder_out.last_hidden_state[0].float().cpu().numpy()
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None

    # Whisper encoder emits 50 frames/sec (20ms stride); keep active speech frames only
    n_frames = max(2, min(hidden.shape[0], int(round((audio_array.size / SAMPLE_RATE) * 50))))
    active = hidden[:n_frames]
    if active.shape[0] > MAX_DTW_FRAMES:
        idx = np.linspace(0, active.shape[0] - 1, MAX_DTW_FRAMES, dtype=int)
        active = active[idx]
    return active.astype(np.float32)


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


def _delta_features(features: np.ndarray) -> np.ndarray:
    if features.shape[0] < 2:
        return np.zeros_like(features)
    padded = np.pad(features, ((1, 1), (0, 0)), mode="edge")
    return 0.5 * (padded[2:] - padded[:-2])


def _extract_mfcc(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    frame_len = int(0.025 * sample_rate)
    hop = int(0.010 * sample_rate)
    n_fft = 512
    n_mels = 26

    if samples.size < frame_len:
        return np.empty((0, 36), dtype=np.float32)

    samples = _trim_silence(samples, sample_rate=sample_rate, pad_s=0.0)
    if samples.size < frame_len:
        return np.empty((0, 36), dtype=np.float32)

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

    # Emphasize delta and delta-delta articulation dynamics to reduce speaker pitch bias
    delta1 = _delta_features(mfcc)
    delta2 = _delta_features(delta1)
    combined = np.concatenate([0.5 * mfcc, delta1, delta2], axis=1)

    if combined.shape[0] > MAX_DTW_FRAMES:
        idx = np.linspace(0, combined.shape[0] - 1, MAX_DTW_FRAMES, dtype=int)
        combined = combined[idx]

    return combined.astype(np.float32)


def _dtw_cosine_score(seq_a: np.ndarray, seq_b: np.ndarray) -> float | None:
    if seq_a.shape[0] < 2 or seq_b.shape[0] < 2:
        return None

    a_norm = seq_a / (np.linalg.norm(seq_a, axis=1, keepdims=True) + 1e-8)
    b_norm = seq_b / (np.linalg.norm(seq_b, axis=1, keepdims=True) + 1e-8)
    cost = 1.0 - np.clip(a_norm @ b_norm.T, -1.0, 1.0)

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
    return round(max(0.0, min(100.0, (1.0 - mean_dist) * 100.0)), 1)


def compare_acoustics(
    teacher_samples: np.ndarray,
    student_samples: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    teacher_emb: np.ndarray | None = None,
    student_emb: np.ndarray | None = None,
) -> dict:
    t_mfcc = _extract_mfcc(teacher_samples, sample_rate)
    s_mfcc = _extract_mfcc(student_samples, sample_rate)
    mfcc_score = _dtw_cosine_score(t_mfcc, s_mfcc)

    emb_score = None
    if (
        teacher_emb is not None
        and student_emb is not None
        and teacher_emb.shape[0] >= 2
        and student_emb.shape[0] >= 2
    ):
        # Partial temporal mean removal cancels speaker/mic bias without collapsing short vowels
        t_centered = teacher_emb - 0.70 * np.mean(teacher_emb, axis=0, keepdims=True)
        s_centered = student_emb - 0.70 * np.mean(student_emb, axis=0, keepdims=True)
        if np.linalg.norm(t_centered) > 1e-6 and np.linalg.norm(s_centered) > 1e-6:
            emb_score = _dtw_cosine_score(t_centered, s_centered)
        else:
            emb_score = _dtw_cosine_score(teacher_emb, student_emb)

    if emb_score is not None and mfcc_score is not None:
        score = round(0.65 * emb_score + 0.35 * mfcc_score, 1)
    elif emb_score is not None:
        score = emb_score
    elif mfcc_score is not None:
        score = mfcc_score
    else:
        return {"acoustic_score": None, "acoustic_label": "Unavailable"}

    if score >= HIGH_ACOUSTIC:
        label = "Native-Like Match"
    elif score >= MODERATE_ACOUSTIC:
        label = "Slight Accent Gap"
    else:
        label = "Strong Accent Gap"

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


def _is_tamil_combining(ch: str) -> bool:
    return ch == "\u0B82" or ("\u0BBE" <= ch <= "\u0BCD") or ch == "\u0BD7"


def _strip_punctuation(text: str) -> str:
    return "".join(ch for ch in text if not unicodedata.category(ch).startswith("P"))


def check_tamil_spelling(text: str) -> dict:
    cleaned = " ".join(_strip_punctuation(text).strip().split())
    if not cleaned:
        return {"spelling_valid": False, "suggested_text": "", "issue": "No Tamil text to verify."}

    for word in cleaned.split():
        if _is_tamil_combining(word[0]):
            return {
                "spelling_valid": False,
                "suggested_text": cleaned,
                "issue": f"Invalid Tamil character sequence at the start of '{word}'.",
            }
        if "\u0BCD\u0BCD" in word:
            return {
                "spelling_valid": False,
                "suggested_text": cleaned,
                "issue": f"Duplicate pulli mark detected in '{word}'.",
            }
        for prev, curr in zip(word, word[1:]):
            if ("\u0BBE" <= prev <= "\u0BCC") and ("\u0BBE" <= curr <= "\u0BCC"):
                return {
                    "spelling_valid": False,
                    "suggested_text": cleaned,
                    "issue": f"Malformed vowel sign sequence detected in '{word}'.",
                }

    compact = _compact_tamil(cleaned)
    if compact in TAMIL_WORD_SPELLING_MAP:
        corrected = TAMIL_WORD_SPELLING_MAP[compact]
        return {
            "spelling_valid": False,
            "suggested_text": corrected,
            "issue": f"Spelling or sandhi issue detected: '{cleaned}' should be '{corrected}'.",
        }

    words = cleaned.split()
    fixed_words: list[str] = []
    changed = False
    for word in words:
        if word in TAMIL_WORD_SPELLING_MAP:
            fixed_words.append(TAMIL_WORD_SPELLING_MAP[word])
            changed = True
        else:
            fixed_words.append(word)

    if changed:
        corrected = " ".join(fixed_words)
        return {
            "spelling_valid": False,
            "suggested_text": corrected,
            "issue": f"Spelling issue detected: '{cleaned}' should be '{corrected}'.",
        }

    return {"spelling_valid": True, "suggested_text": cleaned, "issue": ""}


def segment_tamil_graphemes(text: str) -> list[str]:
    # Keep Tamil combining vowel signs and pulli attached to their base consonant
    graphemes: list[str] = []
    for ch in _strip_punctuation(text).strip():
        if ch.isspace():
            continue
        if _is_tamil_combining(ch) and graphemes:
            graphemes[-1] += ch
        else:
            graphemes.append(ch)
    return graphemes


def _compact_tamil(text: str) -> str:
    return "".join(ch for ch in _strip_punctuation(text) if not ch.isspace())


def _tamil_similarity(spoken: str, target: str) -> float:
    compact_spoken = _compact_tamil(spoken)
    compact_target = _compact_tamil(target)
    char_sim = float(fuzz.ratio(compact_spoken, compact_target))
    grapheme_sim = (
        SequenceMatcher(
            None,
            segment_tamil_graphemes(compact_spoken),
            segment_tamil_graphemes(compact_target),
        ).ratio()
        * 100.0
    )
    # Blend codepoint and Tamil syllable similarity so shared pulli marks do not inflate wrong words
    return round((char_sim + grapheme_sim) / 2.0, 1)


def _normalize_spoken_tamil(text: str) -> str:
    # Normalize common conversational Tamil word-final diglossia drops
    normalized_words: list[str] = []
    for word in _strip_punctuation(text).strip().split():
        base = word
        for ending in SPOKEN_DROPPED_ENDINGS:
            if len(base) > len(ending) + 1 and base.endswith(ending):
                base = base[: -len(ending)]
                break
        normalized_words.append(base)
    return "".join(normalized_words)


def _target_accent_tips(compact_target: str, target: str, syllable_guide: str) -> list[str]:
    steps: list[str] = []
    for group in CONFUSION_GROUPS:
        if any(ch in compact_target for ch in group["chars"]):
            steps.append(group["solution"])
    for sign, _, solution in VOWEL_LENGTH_PAIRS:
        if sign in compact_target:
            steps.append(solution)
            break
    if not steps:
        steps.append("Keep each Tamil syllable crisp and evenly timed without heavy stress shifts.")
    steps.append(f"Say '{target}' syllable by syllable like the Teacher: {syllable_guide}.")
    return steps


def evaluate_teacher(teacher_text: str, fluency: dict | None = None) -> dict:
    cleaned = " ".join(teacher_text.strip().split())
    validation = validate_tamil_script(cleaned)
    syllables = segment_tamil_graphemes(cleaned) if validation["is_valid"] else []
    fluency_info = fluency or {
        "pause_count": 0,
        "speech_duration_s": 0.0,
        "fluency_status": "Not measured",
    }

    if not validation["is_valid"]:
        spelling_valid = False
        suggested_correction = ""
        status = "Invalid Benchmark"
        message = validation["message"]
    else:
        spelling = check_tamil_spelling(cleaned)
        spelling_valid = spelling["spelling_valid"]
        suggested_correction = "" if spelling_valid else spelling["suggested_text"]
        status = "Ready as Benchmark" if spelling_valid else "Spelling Check Needed"
        message = validation["message"] if spelling_valid else spelling["issue"]

    return {
        "transcription": cleaned,
        "script_valid": validation["is_valid"],
        "spelling_valid": spelling_valid,
        "suggested_correction": suggested_correction,
        "purity": validation["purity"],
        "syllables": syllables,
        "status": status,
        "message": message,
        "pause_count": fluency_info["pause_count"],
        "speech_duration_s": fluency_info["speech_duration_s"],
        "fluency_status": fluency_info["fluency_status"],
    }


def _analyze_pronunciation(
    spoken: str,
    target: str,
    similarity: float,
    wording_match: bool,
    diglossia_accepted: bool,
    acoustic: dict | None,
    fluency: dict | None,
) -> tuple[str, list[str]]:
    compact_spoken = _compact_tamil(spoken)
    compact_target = _compact_tamil(target)
    target_graphemes = segment_tamil_graphemes(compact_target)
    spoken_graphemes = segment_tamil_graphemes(compact_spoken)
    syllable_guide = " · ".join(target_graphemes)
    acoustic_score = acoustic.get("acoustic_score") if acoustic else None

    notes: list[str] = []
    steps: list[str] = []

    if similarity < MODERATE_CONFIDENCE:
        notes.append(
            f"Wrong word spoken: Student said '{spoken}', which does not match the correct word '{target}'."
        )
        steps.append(f"Correct word to say: '{target}' ({syllable_guide}).")
        steps.append("Listen to the Teacher's clip and the reference audio below, then repeat clearly.")
        return " ".join(notes), steps

    if wording_match:
        if acoustic_score is not None and acoustic_score < HIGH_ACOUSTIC:
            gap_type = "strong" if acoustic_score < MODERATE_ACOUSTIC else "noticeable"
            notes.append(
                f"Right Tamil word recognized, with a {gap_type} pronunciation and accent gap compared to the Teacher."
            )
            steps.extend(_target_accent_tips(compact_target, target, syllable_guide))
        elif diglossia_accepted:
            notes.append("Natural spoken Tamil (diglossia) pronunciation matched cleanly.")
            steps.append(
                f"Acceptable in spoken Tamil. For formal literary delivery, pronounce every syllable of '{target}': {syllable_guide}."
            )
        else:
            notes.append("Clear Tamil pronunciation and wording closely matching the Teacher.")
            steps.append("No correction needed. Keep maintaining this clarity and pace.")
    else:
        for group in CONFUSION_GROUPS:
            chars = group["chars"]
            target_counts = {c: compact_target.count(c) for c in chars}
            spoken_counts = {c: compact_spoken.count(c) for c in chars}
            if target_counts != spoken_counts and any(target_counts.values()):
                notes.append(group["description"])
                steps.append(group["solution"])

        for sign, description, solution in VOWEL_LENGTH_PAIRS:
            if compact_target.count(sign) != compact_spoken.count(sign):
                notes.append(description)
                steps.append(solution)

        diff = len(spoken_graphemes) - len(target_graphemes)
        if diff < 0:
            notes.append("Shortened or incomplete syllable delivery.")
            steps.append(
                f"Correct word is '{target}' — pronounce all {len(target_graphemes)} syllables completely: {syllable_guide}."
            )
        elif diff > 0:
            notes.append("Extra syllables or trailing sounds detected.")
            steps.append(
                f"Say '{target}' smoothly in one breath without extra trailing sounds: {syllable_guide}."
            )

        if not notes:
            notes.append("Minor syllable articulation difference compared to the Teacher.")
            steps.append(f"Correct word is '{target}' — repeat cleanly syllable by syllable: {syllable_guide}.")

    if fluency and fluency.get("pause_count", 0) >= 2:
        notes.append(f"Choppy pacing ({fluency['pause_count']} mid-utterance pauses detected).")
        steps.append(f"Speak '{target}' smoothly in one breath without pausing between syllables.")

    return "; ".join(notes), steps


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
        step = "Record or confirm a valid Tamil word or phrase from the Teacher first."
        return {
            "transcription": spoken,
            "target": target,
            "mode": eval_mode,
            "wording_match": False,
            "pronunciation_match": False,
            "similarity": 0.0,
            "overall_score": 0.0,
            "pronunciation_status": "Invalid Benchmark",
            "characteristics": "Teacher recording did not produce a valid Tamil benchmark.",
            "solution": step,
            "coaching_steps": [step],
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
        step = f"Re-record clearly in pure Tamil matching '{target}' ({syllable_breakdown})."
        return {
            "transcription": spoken,
            "target": target,
            "mode": eval_mode,
            "wording_match": False,
            "pronunciation_match": False,
            "similarity": 0.0,
            "overall_score": 0.0,
            "pronunciation_status": "Invalid Input",
            "characteristics": spoken_check["message"],
            "solution": step,
            "coaching_steps": [step],
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

    raw_similarity = _tamil_similarity(spoken, target)

    diglossia_accepted = False
    if eval_mode == "spoken":
        spoken_sim = _tamil_similarity(
            _normalize_spoken_tamil(spoken),
            _normalize_spoken_tamil(target),
        )
        similarity = max(raw_similarity, spoken_sim)
        match_threshold = SPOKEN_WORD_MATCH_THRESHOLD
        diglossia_accepted = (spoken_sim > raw_similarity) and (spoken_sim >= match_threshold)
    else:
        similarity = raw_similarity
        match_threshold = WORD_MATCH_THRESHOLD

    wording_match = similarity >= match_threshold
    confidence_score = round(similarity * (spoken_check["purity"] / 100.0), 1)

    if similarity < MODERATE_CONFIDENCE:
        overall_score = (
            round(min(confidence_score, 0.50 * confidence_score + 0.50 * acoustic_score), 1)
            if acoustic_score is not None
            else confidence_score
        )
        pronunciation_match = False
        confidence_label = "Low"
        status = "Word Mismatch"
    elif wording_match:
        if acoustic_score is None:
            overall_score = confidence_score
            pronunciation_match = True
            confidence_label = (
                "High" if (confidence_score >= HIGH_CONFIDENCE or diglossia_accepted) else "Moderate"
            )
            status = (
                "Accurate Pronunciation (Spoken)"
                if diglossia_accepted
                else "Accurate Pronunciation"
            )
        else:
            overall_score = round(0.40 * confidence_score + 0.60 * acoustic_score, 1)
            if acoustic_score >= HIGH_ACOUSTIC:
                pronunciation_match = True
                confidence_label = "High"
                status = (
                    "Accurate Pronunciation (Spoken)"
                    if diglossia_accepted
                    else "Accurate Pronunciation"
                )
            else:
                pronunciation_match = False
                confidence_label = "Moderate" if acoustic_score >= MODERATE_ACOUSTIC else "Low"
                status = "Right Word, Pronunciation Needs Work"
    else:
        overall_score = (
            round(0.55 * confidence_score + 0.45 * acoustic_score, 1)
            if acoustic_score is not None
            else confidence_score
        )
        pronunciation_match = False
        confidence_label = "Moderate" if overall_score >= MODERATE_CONFIDENCE else "Low"
        status = "Phonetic Shift Detected"

    characteristics, coaching_steps = _analyze_pronunciation(
        spoken,
        target,
        similarity,
        wording_match,
        diglossia_accepted,
        acoustic,
        fluency,
    )

    return {
        "transcription": spoken,
        "target": target,
        "mode": eval_mode,
        "wording_match": wording_match,
        "pronunciation_match": pronunciation_match,
        "similarity": similarity,
        "overall_score": overall_score,
        "pronunciation_status": status,
        "characteristics": characteristics,
        "solution": " ".join(coaching_steps),
        "coaching_steps": coaching_steps,
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