DEFAULT_MODEL = "vasista22/whisper-tamil-small"
ALT_MODEL = "vasista22/whisper-tamil-medium"

SAMPLE_RATE = 16000
MAX_AUDIO_SECONDS = 30
MAX_AUDIO_BYTES = 10 * 1024 * 1024
MAX_DECODE_TOKENS = 128
FFMPEG_TIMEOUT_S = 30
MAX_SESSION_CACHE = 20

MIN_PAUSE_MS = 220
MAX_DTW_FRAMES = 200

SAMPLE_WORDS = {
    "வாழைப்பழம்": "Hold the long vowel in 'வா' and curl the tongue back clearly for both 'ழை' and 'ழ'.",
    "தமிழ்": "Keep 'த' and 'மி' crisp, and finish with a clear retroflex 'ழ்'.",
    "பள்ளி": "Stress the double retroflex 'ள்ளி' distinctly instead of softening it to 'லி'.",
    "வணக்கம்": "Articulate the retroflex 'ண' clearly and pause briefly on the double 'க்க'.",
    "மழை": "Pronounce 'ழை' with a curled tongue so it stays distinct from 'லை' and 'ளை'.",
    "நன்றி": "Keep the nasal 'ன்' light and follow it with a crisp 'றி'.",
}

# Tamil consonant confusion sets paired with diagnostic descriptions and corrective tips
CONFUSION_GROUPS = (
    {
        "chars": ("ல", "ள", "ழ"),
        "description": "Lateral and retroflex liquid shift (ல / ள / ழ)",
        "solution": "Adjust tongue position: touch the teeth ridge for 'ல', curl slightly for 'ள', and curl deep back without touching the palate for 'ழ'.",
    },
    {
        "chars": ("ன", "ண"),
        "description": "Alveolar and retroflex nasal shift (ன / ண)",
        "solution": "Keep the tongue forward behind the upper teeth for 'ன' and curl it back toward the roof of the mouth for 'ண'.",
    },
    {
        "chars": ("ர", "ற"),
        "description": "Soft tap and hard trill rhotic shift (ர / ற)",
        "solution": "Use a single light tongue tap for 'ர' and a firmer, stronger ridge trill for 'ற'.",
    },
)

# Long Tamil vowel signs paired with descriptions and corrective tips
VOWEL_LENGTH_PAIRS = (
    ("ா", "Long vowel duration shift on 'ா'", "Hold the open 'ஆ' sound slightly longer on syllables carrying 'ா'."),
    ("ீ", "Long vowel duration shift on 'ீ'", "Extend the 'ஈ' vowel sound clearly instead of cutting it short."),
    ("ூ", "Long vowel duration shift on 'ூ'", "Round the lips and hold the 'ஊ' vowel sound longer."),
    ("ே", "Long vowel duration shift on 'ே'", "Sustain the long 'ஏ' sound rather than rushing it like 'எ'."),
    ("ோ", "Long vowel duration shift on 'ோ'", "Sustain the long 'ஓ' sound cleanly instead of shortening it to 'ஒ'."),
)

# Common word-final Tamil endings softened or dropped in conversational spoken diglossia
SPOKEN_DROPPED_ENDINGS = ("ை", "ம்", "ன்")

# Scoring cutoffs (%)
HIGH_CONFIDENCE = 85.0
MODERATE_CONFIDENCE = 65.0
WORD_MATCH_THRESHOLD = 85.0
SPOKEN_WORD_MATCH_THRESHOLD = 78.0
HIGH_ACOUSTIC = 72.0
MODERATE_ACOUSTIC = 55.0