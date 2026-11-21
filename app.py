import hashlib
import html
import os

import streamlit as st

from config import (
    ALT_MODEL,
    DEFAULT_MODEL,
    MAX_AUDIO_BYTES,
    MAX_AUDIO_SECONDS,
    MAX_SESSION_CACHE,
    SAMPLE_RATE,
    SAMPLE_WORDS,
)
from evaluator import (
    analyze_fluency,
    compare_acoustics,
    decode_audio_pcm,
    evaluate_pronunciation,
    evaluate_teacher,
    extract_phonetic_embeddings,
    generate_reference_audio,
    load_asr_pipeline,
    transcribe_audio,
    validate_tamil_script,
)

st.set_page_config(
    page_title="Tamil Speech To Text Pronunciation Evaluator",
    layout="wide",
)


def inject_ui_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2.5rem;
            max-width: 1220px;
        }
        .app-hero {
            padding: 1.25rem 1.5rem;
            border-radius: 14px;
            border: 1px solid rgba(128, 128, 128, 0.24);
            background: linear-gradient(135deg, rgba(37, 99, 235, 0.10), rgba(16, 185, 129, 0.08));
            margin-bottom: 1.25rem;
        }
        .app-hero h1 {
            margin: 0 0 0.3rem 0;
            font-size: 1.85rem;
            font-weight: 700;
            letter-spacing: -0.02em;
        }
        .app-hero p {
            margin: 0;
            opacity: 0.85;
            font-size: 0.96rem;
        }
        .syllable-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
            margin: 0.45rem 0 0.8rem 0;
        }
        .syllable-chip {
            padding: 0.22rem 0.62rem;
            border-radius: 999px;
            font-size: 0.95rem;
            font-weight: 600;
            border: 1px solid rgba(59, 130, 246, 0.35);
            background: rgba(59, 130, 246, 0.12);
        }
        .verdict-banner {
            padding: 0.7rem 1rem;
            border-radius: 10px;
            font-weight: 600;
            font-size: 0.98rem;
            margin-bottom: 0.9rem;
        }
        .verdict-pass {
            background: rgba(16, 185, 129, 0.14);
            border-left: 4px solid #10b981;
        }
        .verdict-warn {
            background: rgba(245, 158, 11, 0.15);
            border-left: 4px solid #f59e0b;
        }
        .verdict-fail {
            background: rgba(239, 68, 68, 0.14);
            border-left: 4px solid #ef4444;
        }
        .tamil-text-box {
            padding: 0.75rem 1rem;
            border-radius: 10px;
            border: 1px solid rgba(59, 130, 246, 0.30);
            background: rgba(59, 130, 246, 0.08);
            font-size: 1.12rem;
            font-weight: 600;
            margin-bottom: 0.6rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_syllable_chips(syllables: list[str]) -> None:
    if not syllables:
        return
    chips = "".join(f'<span class="syllable-chip">{html.escape(item)}</span>' for item in syllables)
    st.markdown(f'<div class="syllable-row">{chips}</div>', unsafe_allow_html=True)


@st.cache_resource(max_entries=1, show_spinner="Loading Tamil Whisper model...")
def get_asr_pipeline(model_name: str):
    return load_asr_pipeline(model_name)


@st.cache_data(show_spinner=False)
def get_reference_audio(text: str) -> bytes:
    return generate_reference_audio(text)


def clear_session_attempt() -> None:
    st.session_state["round_id"] += 1
    st.session_state["evaluation"] = None


def render_speaker_panel(
    role: str,
    subtitle: str,
    speaker_key: str,
    round_id: int,
    asr_pipe,
    model_name: str,
    allow_edit: bool = False,
) -> dict | None:
    with st.container(border=True):
        st.subheader(role)
        st.caption(subtitle)

        recorded = st.audio_input(
            f"Record {role} voice",
            sample_rate=SAMPLE_RATE,
            key=f"{speaker_key}_mic_{round_id}",
        )
        uploaded = st.file_uploader(
            f"Or upload {role} audio (.wav / .mp3, up to {MAX_AUDIO_SECONDS}s)",
            type=["wav", "mp3"],
            key=f"{speaker_key}_upload_{round_id}",
        )

        active_file = recorded if recorded is not None else uploaded
        if active_file is None:
            st.caption(f"Record or upload up to {MAX_AUDIO_SECONDS} seconds of Tamil speech.")
            return None

        audio_bytes = active_file.getvalue()
        if len(audio_bytes) > MAX_AUDIO_BYTES:
            st.error("Audio file exceeds the 10 MB limit. Please upload a shorter clip.")
            return None

        suffix = os.path.splitext(active_file.name)[1].lower() or ".wav"
        audio_format = "audio/mp3" if suffix == ".mp3" else "audio/wav"

        st.audio(audio_bytes, format=audio_format)

        audio_hash = hashlib.sha256(audio_bytes).hexdigest()
        cache_key = f"{speaker_key}:{model_name}:{audio_hash}"
        transcriptions = st.session_state["transcriptions"]

        if cache_key not in transcriptions:
            if len(transcriptions) >= MAX_SESSION_CACHE:
                transcriptions.clear()
            with st.spinner(f"Transcribing {role.lower()} audio..."):
                try:
                    transcriptions[cache_key] = transcribe_audio(asr_pipe, audio_bytes, suffix=suffix)
                except (OSError, RuntimeError, ValueError) as err:
                    st.error(f"Could not transcribe {role.lower()} audio: {err}")
                    return None

        transcript = transcriptions[cache_key]
        script_check = validate_tamil_script(transcript)

        st.markdown("**Transcribed Tamil Text**")
        if not transcript:
            st.warning("No speech detected in this recording.")
        elif script_check["has_latin"]:
            st.warning(script_check["message"])
        else:
            st.markdown(
                f'<div class="tamil-text-box">{html.escape(transcript)}</div>',
                unsafe_allow_html=True,
            )

        final_text = transcript
        if allow_edit:
            edited = st.text_input(
                "Confirm or edit Teacher Tamil text (optional)",
                value=transcript,
                key=f"teacher_edit_{round_id}_{audio_hash[:8]}",
                help="Fix any ASR mishearing in pure Tamil script before evaluating the Student.",
            ).strip()
            final_text = edited
            if edited:
                edit_check = validate_tamil_script(edited)
                if not edit_check["is_valid"]:
                    st.warning(edit_check["message"])

        return {
            "transcript": final_text,
            "audio_bytes": audio_bytes,
            "suffix": suffix,
            "cache_key": cache_key,
        }


def render_teacher_card(teacher: dict) -> None:
    with st.container(border=True):
        st.markdown("#### Teacher Benchmark")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="Tamil Script Purity",
                value=f"{teacher['purity']}%",
                delta=teacher["status"],
                delta_color="normal" if teacher["script_valid"] else "inverse",
            )
        with col2:
            st.metric(
                label="Active Speech",
                value=f"{teacher['speech_duration_s']}s",
                delta=teacher["fluency_status"],
                delta_color="off",
            )

        if teacher["script_valid"]:
            st.markdown(f"**Confirmed Tamil Benchmark:** {teacher['transcription']}")
            st.markdown("**Syllable Breakdown**")
            render_syllable_chips(teacher["syllables"])
        else:
            st.warning(teacher["message"])


def render_student_card(student: dict) -> None:
    with st.container(border=True):
        st.markdown("#### Student Pronunciation Evaluation")

        mode_label = "Conversational Spoken" if student["mode"] == "spoken" else "Strict Literary"
        if student["pronunciation_match"]:
            banner_class = "verdict-pass"
        elif student["pronunciation_status"] in {
            "Right Word, Pronunciation Needs Work",
            "Phonetic Shift Detected",
        }:
            banner_class = "verdict-warn"
        else:
            banner_class = "verdict-fail"

        st.markdown(
            f'<div class="verdict-banner {banner_class}">'
            f'{html.escape(student["pronunciation_status"])} &nbsp;·&nbsp; {html.escape(mode_label)} Mode'
            f"</div>",
            unsafe_allow_html=True,
        )

        # Pronunciation & Accent first, Wording Match beside it
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            acoustic_val = (
                f"{student['acoustic_score']}%"
                if student["acoustic_score"] is not None
                else "N/A"
            )
            st.metric(
                label="Pronunciation & Accent",
                value=acoustic_val,
                delta=student["acoustic_label"],
                delta_color="off",
            )
        with m_col2:
            st.metric(
                label="Overall Score",
                value=f"{student['overall_score']}%",
                delta=student["confidence_label"],
                delta_color="normal" if student["pronunciation_match"] else "inverse",
            )
        with m_col3:
            st.metric(
                label="Wording Match",
                value=f"{student['similarity']}%",
                delta="Word Matched" if student["wording_match"] else "Mismatch",
                delta_color="normal" if student["wording_match"] else "inverse",
            )

        st.caption(
            f"Audio Fluency: {student['fluency_status']} ({student['speech_duration_s']}s active speech)"
        )
        st.markdown(f"**Diagnosis:** {student['characteristics']}")

        st.markdown("**How to Pronounce Correctly**")
        for step in student["coaching_steps"]:
            st.markdown(f"- {step}")

        if not student["pronunciation_match"] and student["target"]:
            try:
                guide_audio = get_reference_audio(student["target"])
                if guide_audio:
                    st.caption(f"Listen to clear Tamil pronunciation of '{student['target']}':")
                    st.audio(guide_audio, format="audio/mp3")
            except (OSError, RuntimeError, ValueError):
                pass


def main() -> None:
    inject_ui_styles()

    st.session_state.setdefault("transcriptions", {})
    st.session_state.setdefault("embeddings", {})
    st.session_state.setdefault("round_id", 0)
    st.session_state.setdefault("evaluation", None)

    round_id = st.session_state["round_id"]

    st.markdown(
        """
        <div class="app-hero">
            <h1>Tamil Speech To Text Pronunciation Evaluator</h1>
            <p>Evaluate Student Tamil pronunciation, accent, and wording directly against the Teacher's voice.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Evaluation Settings")
        model_choices = {
            "Tamil Whisper Small (Default)": DEFAULT_MODEL,
            "Tamil Whisper Medium (Higher Accuracy)": ALT_MODEL,
        }
        selected_label = st.selectbox("Select Tamil Whisper model", list(model_choices.keys()))
        model_name = model_choices[selected_label]
        st.caption(f"Active checkpoint: `{model_name}`")

        mode_choices = {
            "Strict Literary Tamil": "literary",
            "Conversational Spoken Tamil": "spoken",
        }
        mode_label = st.radio("Tamil Evaluation Mode", list(mode_choices.keys()), index=0)
        eval_mode = mode_choices[mode_label]

    try:
        asr_pipe = get_asr_pipeline(model_name)
    except (OSError, RuntimeError, ValueError) as err:
        st.error(f"Failed to load ASR model '{model_name}': {err}")
        st.stop()

    st.header("Voice Recording")
    teacher_col, student_col = st.columns(2, gap="medium")
    with teacher_col:
        teacher_data = render_speaker_panel(
            "Teacher",
            "Speak any Tamil word or sentence to set the live benchmark.",
            "teacher",
            round_id,
            asr_pipe,
            model_name,
            allow_edit=True,
        )
    with student_col:
        student_data = render_speaker_panel(
            "Student",
            "Repeat the Teacher's Tamil speech to check pronunciation, accent, and wording.",
            "student",
            round_id,
            asr_pipe,
            model_name,
        )

    st.divider()
    st.header("Pronunciation Evaluation")

    btn_col1, btn_col2 = st.columns([3, 1])
    with btn_col1:
        evaluate_clicked = st.button(
            "Evaluate Pronunciation",
            type="primary",
            use_container_width=True,
        )
    with btn_col2:
        st.button(
            "Clear / Speak Again",
            on_click=clear_session_attempt,
            use_container_width=True,
        )

    if evaluate_clicked:
        if teacher_data is None or student_data is None:
            st.warning("Provide both Teacher and Student recordings before evaluating.")
        else:
            with st.spinner("Analyzing pronunciation, accent, wording, and fluency..."):
                embeddings = st.session_state["embeddings"]
                try:
                    t_samples = decode_audio_pcm(teacher_data["audio_bytes"], teacher_data["suffix"])
                    s_samples = decode_audio_pcm(student_data["audio_bytes"], student_data["suffix"])
                    t_fluency = analyze_fluency(t_samples)
                    s_fluency = analyze_fluency(s_samples)

                    t_key = teacher_data["cache_key"]
                    s_key = student_data["cache_key"]
                    if len(embeddings) >= MAX_SESSION_CACHE:
                        embeddings.clear()
                    if t_key not in embeddings:
                        embeddings[t_key] = extract_phonetic_embeddings(asr_pipe, t_samples)
                    if s_key not in embeddings:
                        embeddings[s_key] = extract_phonetic_embeddings(asr_pipe, s_samples)

                    acoustic = compare_acoustics(
                        t_samples,
                        s_samples,
                        teacher_emb=embeddings[t_key],
                        student_emb=embeddings[s_key],
                    )
                except (OSError, RuntimeError, ValueError):
                    t_fluency = None
                    s_fluency = None
                    acoustic = None

                st.session_state["evaluation"] = {
                    "teacher": evaluate_teacher(teacher_data["transcript"], fluency=t_fluency),
                    "student": evaluate_pronunciation(
                        student_data["transcript"],
                        teacher_data["transcript"],
                        mode=eval_mode,
                        acoustic=acoustic,
                        fluency=s_fluency,
                    ),
                }

    evaluation = st.session_state["evaluation"]
    if evaluation is not None:
        res_col1, res_col2 = st.columns(2, gap="medium")
        with res_col1:
            render_teacher_card(evaluation["teacher"])
        with res_col2:
            render_student_card(evaluation["student"])

    st.divider()
    st.header("Reference Tamil Practice Guide")
    st.caption("Listen to example Tamil words and practice proper articulation before recording.")

    sample_list = list(SAMPLE_WORDS.keys())
    selected_word = st.selectbox("Choose a practice Tamil word", sample_list, index=0)
    custom_word = st.text_input(
        "Or type any Tamil word to hear how it is pronounced (Tamil script only)",
        placeholder="எ.கா. வாழைப்பழம்",
    ).strip()

    practice_word = custom_word if custom_word else selected_word
    practice_check = validate_tamil_script(practice_word)

    if not practice_check["is_valid"]:
        st.error(practice_check["message"])
    else:
        guidance = (
            "Articulate each Tamil syllable clearly and hold long vowels distinctly."
            if custom_word
            else SAMPLE_WORDS[selected_word]
        )
        with st.container(border=True):
            st.markdown(f"### {practice_word}")
            st.write(f"**Pronunciation Guide:** {guidance}")
            try:
                ref_audio = get_reference_audio(practice_word)
                if ref_audio:
                    st.audio(ref_audio, format="audio/mp3")
            except (OSError, RuntimeError, ValueError) as err:
                st.warning(f"Reference audio unavailable: {err}")


if __name__ == "__main__":
    main()