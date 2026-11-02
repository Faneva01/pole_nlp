"""Détection de la langue, limitée au français et à l'anglais.

La détection travaille sur les *features de l'encodeur* déjà calculées (une
seule passe encodeur par énoncé, partagée avec le décodage), et la confiance
est réellement utilisée : détection douteuse → on garde la langue de la
conversation en cours plutôt que de forcer une erreur.
"""

from .config import (
    LANGUAGE_FALLBACK,
    LANGUAGE_MIN_CONFIDENCE,
    LANGUAGE_MIN_CONFIDENCE_SHORT,
    LANGUAGE_MIN_MASS,
    LANGUAGE_SHORT_SECONDS,
)


def detect_fr_en(model, features):
    """Retourne (langue, confiance fr/en, masse fr+en) à partir des features.

    - confiance : part du gagnant dans fr+en (comme avant) ;
    - masse : probabilité totale fr+en. Faible = ce n'est probablement ni du
      français ni de l'anglais (la confiance seule masquait ce cas).
    """
    _, probabilities = model.detect_language(features)
    if isinstance(probabilities, list):  # features avec dimension de lot
        probabilities = probabilities[0]

    prob_fr = probabilities["fr"]
    prob_en = probabilities["en"]
    mass = prob_fr + prob_en
    if mass <= 0:
        return LANGUAGE_FALLBACK, 0.0, 0.0
    if prob_en > prob_fr:
        return "en", prob_en / mass, mass
    return "fr", prob_fr / mass, mass


def choose_language(
    model, features, requested, audio_seconds=None, last_language=None
):
    """Choisit la langue de transcription.

    Retourne un dict : language, confidence, mass, supported, uncertain.
    - ``supported`` False : la parole ne ressemble ni à fr ni à en ;
    - ``uncertain`` True : détection peu fiable (confiance basse, énoncé court,
      langue non supportée) ; la langue de la conversation (``last_language``)
      est alors préférée, sinon LANGUAGE_FALLBACK pour le cas non supporté.
    """
    if requested != "auto":
        return {
            "language": requested,
            "confidence": 1.0,
            "mass": 1.0,
            "supported": True,
            "uncertain": False,
        }

    language, confidence, mass = detect_fr_en(model, features)
    supported = mass >= LANGUAGE_MIN_MASS
    is_short = audio_seconds is not None and audio_seconds < LANGUAGE_SHORT_SECONDS
    required = LANGUAGE_MIN_CONFIDENCE_SHORT if is_short else LANGUAGE_MIN_CONFIDENCE
    uncertain = (not supported) or confidence < required

    if uncertain:
        language = last_language or (LANGUAGE_FALLBACK if not supported else language)

    return {
        "language": language,
        "confidence": confidence,
        "mass": mass,
        "supported": supported,
        "uncertain": uncertain,
    }
