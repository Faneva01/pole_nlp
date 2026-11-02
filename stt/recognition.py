"""Reconnaissance d'un énoncé : une seule passe encodeur, repli en température.

Chemin court (énoncé <= 30 s, cas du micro) :
    mel -> encodeur (1 fois) -> détection de langue -> décodage
La détection de langue et le décodage réutilisent les mêmes features, au lieu
de recalculer mel + encodeur deux fois (la première en fp32) comme avant.

Chemin long (fichier > 30 s) : ``model.transcribe`` d'origine.
"""

import re
import time

import torch
import whisper
from whisper.audio import N_SAMPLES
from whisper.decoding import DecodingOptions

from .config import (
    BEST_OF,
    COMPRESSION_RATIO_THRESHOLD,
    HALLUCINATION_PHRASES,
    LOGPROB_THRESHOLD,
    NO_SPEECH_THRESHOLD,
    SAMPLE_RATE,
    TEMPERATURES,
)
from .language import choose_language
from .optimization import build_decode_options, use_fp16
from .prompts import PROMPT_TOKEN_LIMIT, PROMPT_TOKEN_TARGET, PROMPTS, build_initial_prompt


# ---------------------------------------------------------------
# Filtres sur le texte
# ---------------------------------------------------------------
def is_hallucination(text):
    """True si le texte contient une phrase typique d'hallucination de Whisper."""
    normalized = re.sub(r"\s+", " ", text.lower().replace("’", "'"))
    return any(phrase in normalized for phrase in HALLUCINATION_PHRASES)


# ---------------------------------------------------------------
# Vérification des prompts
# ---------------------------------------------------------------
def prompt_token_count(model, text):
    tokenizer = whisper.tokenizer.get_tokenizer(
        model.is_multilingual, num_languages=model.num_languages
    )
    return len(tokenizer.encode(" " + text.strip()))


def validate_prompts(model):
    """Retourne la liste des prompts trop longs (Whisper tronque au-delà de 223 tokens)."""
    problems = []
    for language, profiles in PROMPTS.items():
        for profile, text in profiles.items():
            count = prompt_token_count(model, text)
            if count > PROMPT_TOKEN_LIMIT:
                problems.append((language, profile, count, "TRONQUÉ"))
            elif count > PROMPT_TOKEN_TARGET:
                problems.append((language, profile, count, "proche de la limite"))
    return problems


# ---------------------------------------------------------------
# Chemin court
# ---------------------------------------------------------------
def encode_audio(model, audio, fp16):
    """mel + encodeur, une seule fois. ``audio`` : float32 mono 16 kHz, <= 30 s."""
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio, model.dims.n_mels, device=model.device)
    mel = mel.to(torch.float16 if fp16 else torch.float32)
    with torch.no_grad():
        return model.encoder(mel.unsqueeze(0))


def _needs_fallback(result):
    """Même règle que ``whisper.transcribe`` : résultat répétitif ou peu probable,
    sauf s'il s'agit simplement de silence."""
    needs = (
        result.compression_ratio > COMPRESSION_RATIO_THRESHOLD
        or result.avg_logprob < LOGPROB_THRESHOLD
    )
    if result.no_speech_prob > NO_SPEECH_THRESHOLD and result.avg_logprob < LOGPROB_THRESHOLD:
        needs = False  # silence : inutile de réessayer
    return needs


def recognize_short(audio, model, args, device, last_language=None):
    start_time = time.time()
    fp16 = use_fp16(device)
    audio_seconds = len(audio) / SAMPLE_RATE

    features = encode_audio(model, audio, fp16)
    choice = choose_language(
        model, features, args.language, audio_seconds, last_language
    )
    language = choice["language"]
    prompt = build_initial_prompt(getattr(args, "prompt_profile", "stade"), language)

    result = None
    for temperature in TEMPERATURES:
        options = {
            "task": args.task,
            "language": language,
            "temperature": temperature,
            "fp16": fp16,
            "without_timestamps": True,
        }
        if prompt:
            options["prompt"] = prompt
        if temperature > 0:
            options["best_of"] = BEST_OF
        result = model.decode(features, DecodingOptions(**options))[0]
        if not _needs_fallback(result):
            break

    text = result.text.strip()
    filtered = None
    if result.no_speech_prob > NO_SPEECH_THRESHOLD and result.avg_logprob < LOGPROB_THRESHOLD:
        text, filtered = "", "silence"
    elif text and is_hallucination(text):
        text, filtered = "", "hallucination"

    return {
        "texte": text,
        "language": language,
        "confidence": choice["confidence"],
        "language_mass": choice["mass"],
        "language_supported": choice["supported"],
        "language_uncertain": choice["uncertain"],
        "avg_logprob": float(result.avg_logprob),
        "no_speech_prob": float(result.no_speech_prob),
        "temperature": float(result.temperature),
        "filtered": filtered,
        "duration": time.time() - start_time,
        "audio_duration": audio_seconds,
    }


# ---------------------------------------------------------------
# Chemin long (fichiers)
# ---------------------------------------------------------------
def recognize_long(audio, model, args, device, last_language=None):
    start_time = time.time()
    fp16 = use_fp16(device)
    features = encode_audio(model, audio[:N_SAMPLES], fp16)
    choice = choose_language(model, features, args.language, None, last_language)

    options = build_decode_options(
        args.task, device, choice["language"], getattr(args, "prompt_profile", "stade")
    )
    result = model.transcribe(audio, **options)
    text = result["text"].strip()
    filtered = None
    if text and is_hallucination(text):
        text, filtered = "", "hallucination"

    return {
        "texte": text,
        "language": choice["language"],
        "confidence": choice["confidence"],
        "language_mass": choice["mass"],
        "language_supported": choice["supported"],
        "language_uncertain": choice["uncertain"],
        "avg_logprob": float("nan"),
        "no_speech_prob": float("nan"),
        "temperature": float("nan"),
        "filtered": filtered,
        "duration": time.time() - start_time,
        "audio_duration": len(audio) / SAMPLE_RATE,
    }


def recognize(audio, model, args, device, last_language=None):
    """Choisit le chemin court (<= 30 s) ou long selon la durée."""
    if len(audio) <= N_SAMPLES:
        return recognize_short(audio, model, args, device, last_language)
    return recognize_long(audio, model, args, device, last_language)
