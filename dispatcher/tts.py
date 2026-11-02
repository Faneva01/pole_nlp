"""Text to Speech avec Piper."""

from __future__ import annotations

import os
import sys
from pathlib import Path

DOSSIER = Path(__file__).resolve().parent
RACINE = DOSSIER.parent
PHASE2 = RACINE.parents[1]

_CHEMINS_PIPER = [
    DOSSIER / "venv" / "lib" / "python3.12" / "site-packages",
    PHASE2 / "assistant_vocal" / "venv" / "lib" / "python3.12" / "site-packages",
]

for _chemin in reversed(_CHEMINS_PIPER):
    _voice = _chemin / "piper" / "voice.py"
    if _voice.exists() and _voice.stat().st_size > 0 and str(_chemin) not in sys.path:
        sys.path.insert(0, str(_chemin))

try:
    import numpy as np
    import sounddevice as sd
    from piper import PiperVoice
except ImportError as exc:  # pragma: no cover - dépend de l'installation
    raise ImportError(
        "Piper est introuvable. Installe piper-tts dans un environnement "
        "contenant aussi numpy et sounddevice."
    ) from exc


def _trouver_voix(chemin: str | Path | None = None) -> Path:
    if chemin is not None:
        chemin = Path(chemin)
        if chemin.is_file() and chemin.stat().st_size > 0:
            return chemin

    variable = os.environ.get("PIPER_VOICE")
    if variable:
        chemin = Path(variable)
        if chemin.is_file() and chemin.stat().st_size > 0:
            return chemin

    candidats = [
        *sorted((DOSSIER / "voix").glob("*.onnx")),
        PHASE2 / "assistant_vocal" / "fr_FR-siwis-medium.onnx",
        DOSSIER / "fr_FR-siwis-medium.onnx",
        PHASE2 / "assistant_vocal" / "fr_FR-siwis-low.onnx",
    ]
    for candidat in candidats:
        if candidat.is_file() and candidat.stat().st_size > 0:
            return candidat

    raise FileNotFoundError(
        "Aucun modèle Piper trouvé. Place un fichier .onnx dans "
        f"{DOSSIER / 'voix'} ou définis PIPER_VOICE."
    )


def charger_voix(chemin: str | Path | None = None):
    """Charge la voix Piper une seule fois au démarrage."""
    fichier = _trouver_voix(chemin)
    print("Chargement du TTS (Piper) :", fichier.name, "...")
    return PiperVoice.load(str(fichier))


def synthetiser(voix, texte: str):
    """Retourne l'audio int16 du texte, ou None si le texte est vide."""
    if not texte or not texte.strip():
        return None

    morceaux = [
        morceau.audio_int16_array for morceau in voix.synthesize(texte.strip())
    ]
    if not morceaux:
        return None
    return np.concatenate(morceaux)


def jouer(audio, samplerate: int):
    sd.play(audio, samplerate=samplerate)
    sd.wait()


def parler(voix, texte: str) -> bool:
    """Synthétise et joue le texte. Retourne True si quelque chose a été dit."""
    audio = synthetiser(voix, texte)
    if audio is None:
        return False
    jouer(audio, voix.config.sample_rate)
    return True
