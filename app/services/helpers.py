"""
Helper functions for speaker diarization.
Adapted from https://github.com/MahmoudAshraf97/whisper-diarization
"""

import os
import re
import shutil
from typing import List, Dict, Any, Tuple, Optional

import nltk

# NLTK data (punkt, punkt_tab) must be pre-installed during Docker build
# Do not attempt to download at runtime as the filesystem may be read-only

# Languages supported by the punctuation model
punct_model_langs = [
    "en",
    "fr",
    "de",
    "es",
    "it",
    "nl",
    "pt",
    "bg",
    "pl",
    "cs",
    "sk",
    "sl",
]

# Language code to ISO mapping for forced alignment
langs_to_iso = {
    "af": "afr",
    "am": "amh",
    "ar": "ara",
    "as": "asm",
    "az": "aze",
    "ba": "bak",
    "be": "bel",
    "bg": "bul",
    "bn": "ben",
    "bo": "tib",
    "br": "bre",
    "bs": "bos",
    "ca": "cat",
    "cs": "cze",
    "cy": "wel",
    "da": "dan",
    "de": "ger",
    "el": "gre",
    "en": "eng",
    "es": "spa",
    "et": "est",
    "eu": "baq",
    "fa": "per",
    "fi": "fin",
    "fo": "fao",
    "fr": "fre",
    "gl": "glg",
    "gu": "guj",
    "ha": "hau",
    "haw": "haw",
    "he": "heb",
    "hi": "hin",
    "hr": "hrv",
    "ht": "hat",
    "hu": "hun",
    "hy": "arm",
    "id": "ind",
    "is": "ice",
    "it": "ita",
    "ja": "jpn",
    "jw": "jav",
    "ka": "geo",
    "kk": "kaz",
    "km": "khm",
    "kn": "kan",
    "ko": "kor",
    "la": "lat",
    "lb": "ltz",
    "ln": "lin",
    "lo": "lao",
    "lt": "lit",
    "lv": "lav",
    "mg": "mlg",
    "mi": "mao",
    "mk": "mac",
    "ml": "mal",
    "mn": "mon",
    "mr": "mar",
    "ms": "may",
    "mt": "mlt",
    "my": "bur",
    "ne": "nep",
    "nl": "dut",
    "nn": "nno",
    "no": "nor",
    "oc": "oci",
    "pa": "pan",
    "pl": "pol",
    "ps": "pus",
    "pt": "por",
    "ro": "rum",
    "ru": "rus",
    "sa": "san",
    "sd": "snd",
    "si": "sin",
    "sk": "slo",
    "sl": "slv",
    "sn": "sna",
    "so": "som",
    "sq": "alb",
    "sr": "srp",
    "su": "sun",
    "sv": "swe",
    "sw": "swa",
    "ta": "tam",
    "te": "tel",
    "tg": "tgk",
    "th": "tha",
    "tk": "tuk",
    "tl": "tgl",
    "tr": "tur",
    "tt": "tat",
    "uk": "ukr",
    "ur": "urd",
    "uz": "uzb",
    "vi": "vie",
    "yi": "yid",
    "yo": "yor",
    "yue": "yue",
    "zh": "chi",
}

LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
    "yue": "cantonese",
}

TO_LANGUAGE_CODE = {
    **{language: code for code, language in LANGUAGES.items()},
    "burmese": "my",
    "valencian": "ca",
    "flemish": "nl",
    "haitian": "ht",
    "letzeburgesch": "lb",
    "pushto": "ps",
    "panjabi": "pa",
    "moldavian": "ro",
    "moldovan": "ro",
    "sinhalese": "si",
    "castilian": "es",
}

whisper_langs = sorted(LANGUAGES.keys()) + sorted(
    [k.title() for k in TO_LANGUAGE_CODE.keys()]
)


def get_word_ts_anchor(s: float, e: float, option: str = "start") -> float:
    """Get the anchor timestamp for a word."""
    if option == "end":
        return e
    elif option == "mid":
        return (s + e) / 2
    return s


def get_words_speaker_mapping(
    wrd_ts: List[Dict], spk_ts: List[Tuple], word_anchor_option: str = "start"
) -> List[Dict]:
    """Map words to speakers based on timestamps."""
    if not spk_ts:
        # No speaker segments, assign all to Speaker 0
        return [
            {
                "word": wrd_dict.get("word") or wrd_dict.get("text", ""),
                "start_time": int(wrd_dict["start"] * 1000),
                "end_time": int(wrd_dict["end"] * 1000),
                "speaker": "0",
            }
            for wrd_dict in wrd_ts
        ]

    s, e, sp = spk_ts[0]
    turn_idx = 0
    wrd_spk_mapping = []

    for wrd_dict in wrd_ts:
        ws = int(wrd_dict["start"] * 1000)
        we = int(wrd_dict["end"] * 1000)
        # Handle both 'word' and 'text' keys
        wrd = wrd_dict.get("word") or wrd_dict.get("text", "")
        wrd_pos = get_word_ts_anchor(ws, we, word_anchor_option)

        while wrd_pos > float(e):
            turn_idx += 1
            turn_idx = min(turn_idx, len(spk_ts) - 1)
            s, e, sp = spk_ts[turn_idx]
            if turn_idx == len(spk_ts) - 1:
                e = get_word_ts_anchor(ws, we, option="end")

        wrd_spk_mapping.append(
            {
                "word": wrd,
                "start_time": ws,
                "end_time": we,
                "speaker": sp,
            }
        )

    return wrd_spk_mapping


sentence_ending_punctuations = ".?!"


def get_first_word_idx_of_sentence(
    word_idx: int, word_list: List[str], speaker_list: List[str], max_words: int
) -> int:
    """Find the first word index of the current sentence."""
    is_word_sentence_end = (
        lambda x: x >= 0 and word_list[x][-1] in sentence_ending_punctuations
    )
    left_idx = word_idx

    while (
        left_idx > 0
        and word_idx - left_idx < max_words
        and speaker_list[left_idx - 1] == speaker_list[left_idx]
        and not is_word_sentence_end(left_idx - 1)
    ):
        left_idx -= 1

    return left_idx if left_idx == 0 or is_word_sentence_end(left_idx - 1) else -1


def get_last_word_idx_of_sentence(
    word_idx: int, word_list: List[str], max_words: int
) -> int:
    """Find the last word index of the current sentence."""
    is_word_sentence_end = (
        lambda x: x >= 0 and word_list[x][-1] in sentence_ending_punctuations
    )
    right_idx = word_idx

    while (
        right_idx < len(word_list) - 1
        and right_idx - word_idx < max_words
        and not is_word_sentence_end(right_idx)
    ):
        right_idx += 1

    return (
        right_idx
        if right_idx == len(word_list) - 1 or is_word_sentence_end(right_idx)
        else -1
    )


def get_realigned_ws_mapping_with_punctuation(
    word_speaker_mapping: List[Dict], max_words_in_sentence: int = 50
) -> List[Dict]:
    """Realign word-speaker mapping based on punctuation boundaries."""
    is_word_sentence_end = (
        lambda x: x >= 0
        and word_speaker_mapping[x]["word"][-1] in sentence_ending_punctuations
    )
    wsp_len = len(word_speaker_mapping)

    words_list, speaker_list = [], []
    for line_dict in word_speaker_mapping:
        word, speaker = line_dict["word"], line_dict["speaker"]
        words_list.append(word)
        speaker_list.append(speaker)

    k = 0
    while k < len(word_speaker_mapping):
        if (
            k < wsp_len - 1
            and speaker_list[k] != speaker_list[k + 1]
            and not is_word_sentence_end(k)
        ):
            left_idx = get_first_word_idx_of_sentence(
                k, words_list, speaker_list, max_words_in_sentence
            )
            right_idx = (
                get_last_word_idx_of_sentence(
                    k, words_list, max_words_in_sentence - k + left_idx - 1
                )
                if left_idx > -1
                else -1
            )

            if min(left_idx, right_idx) == -1:
                k += 1
                continue

            spk_labels = speaker_list[left_idx : right_idx + 1]
            mod_speaker = max(set(spk_labels), key=spk_labels.count)

            if spk_labels.count(mod_speaker) < len(spk_labels) // 2:
                k += 1
                continue

            speaker_list[left_idx : right_idx + 1] = [mod_speaker] * (
                right_idx - left_idx + 1
            )
            k = right_idx

        k += 1

    realigned_list = []
    for k, line_dict in enumerate(word_speaker_mapping):
        new_dict = line_dict.copy()
        new_dict["speaker"] = speaker_list[k]
        realigned_list.append(new_dict)

    return realigned_list


def get_sentences_speaker_mapping(
    word_speaker_mapping: List[Dict], spk_ts: List[Tuple]
) -> List[Dict]:
    """Convert word-level speaker mapping to sentence-level mapping."""
    sentence_checker = nltk.tokenize.PunktSentenceTokenizer().text_contains_sentbreak

    if not word_speaker_mapping:
        return []

    first_word = word_speaker_mapping[0]
    prev_spk = first_word["speaker"]

    snts = []
    snt = {
        "speaker": prev_spk,
        "start_time": first_word["start_time"] / 1000.0,
        "end_time": first_word["end_time"] / 1000.0,
        "text": "",
    }

    for wrd_dict in word_speaker_mapping:
        wrd, spk = wrd_dict["word"], wrd_dict["speaker"]
        s, e = wrd_dict["start_time"] / 1000.0, wrd_dict["end_time"] / 1000.0

        if spk != prev_spk or sentence_checker(snt["text"] + " " + wrd):
            snts.append(snt)
            snt = {
                "speaker": spk,
                "start_time": s,
                "end_time": e,
                "text": "",
            }
        else:
            snt["end_time"] = e

        snt["text"] += wrd + " "
        prev_spk = spk

    snts.append(snt)
    return snts


def get_speaker_aware_transcript(sentences_speaker_mapping: List[Dict]) -> str:
    """Generate speaker-aware transcript text."""
    if not sentences_speaker_mapping:
        return ""

    lines = []
    previous_speaker = sentences_speaker_mapping[0]["speaker"]
    current_line = f"{previous_speaker}: "

    for sentence_dict in sentences_speaker_mapping:
        speaker = sentence_dict["speaker"]
        sentence = sentence_dict["text"].strip()

        if speaker != previous_speaker:
            lines.append(current_line.strip())
            current_line = f"\n{speaker}: "
            previous_speaker = speaker

        current_line += sentence + " "

    lines.append(current_line.strip())
    return "\n".join(lines)


def format_timestamp(
    milliseconds: float, always_include_hours: bool = True, decimal_marker: str = ","
) -> str:
    """Format milliseconds to SRT/VTT timestamp format."""
    assert milliseconds >= 0, "non-negative timestamp expected"

    hours = int(milliseconds // 3_600_000)
    milliseconds -= hours * 3_600_000

    minutes = int(milliseconds // 60_000)
    milliseconds -= minutes * 60_000

    seconds = int(milliseconds // 1_000)
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{int(milliseconds):03d}"


def write_srt(transcript: List[Dict]) -> str:
    """Write transcript to SRT format string."""
    lines = []
    for i, segment in enumerate(transcript, start=1):
        start = format_timestamp(segment["start_time"] * 1000)
        end = format_timestamp(segment["end_time"] * 1000)
        text = segment["text"].strip().replace("-->", "->")
        speaker = segment["speaker"]
        lines.append(f"{i}\n{start} --> {end}\n{speaker}: {text}\n")
    return "\n".join(lines)


def write_vtt(transcript: List[Dict]) -> str:
    """Write transcript to VTT format string."""
    lines = ["WEBVTT\n"]
    for segment in transcript:
        start = format_timestamp(segment["start_time"] * 1000, decimal_marker=".")
        end = format_timestamp(segment["end_time"] * 1000, decimal_marker=".")
        text = segment["text"].strip()
        speaker = segment["speaker"]
        lines.append(f"{start} --> {end}\n{speaker}: {text}\n")
    return "\n".join(lines)


def find_numeral_symbol_tokens(tokenizer) -> List[int]:
    """Find token IDs that contain numerals or currency symbols."""
    numeral_symbol_tokens = [-1]
    for token, token_id in tokenizer.get_vocab().items():
        has_numeral_symbol = any(c in "0123456789%$£" for c in token)
        if has_numeral_symbol:
            numeral_symbol_tokens.append(token_id)
    return numeral_symbol_tokens


def cleanup(path: str) -> None:
    """Clean up a file or directory."""
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)


def process_language_arg(language: Optional[str], model_name: str) -> Optional[str]:
    """Process and validate the language argument."""
    if language is not None:
        language = language.lower()
        if language not in LANGUAGES:
            if language in TO_LANGUAGE_CODE:
                language = TO_LANGUAGE_CODE[language]
            else:
                raise ValueError(f"Unsupported language: {language}")

        if model_name.endswith(".en") and language != "en":
            raise ValueError(
                f"{model_name} is an English-only model but chosen language is '{language}'"
            )

    return language
