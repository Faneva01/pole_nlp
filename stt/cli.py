"""Lecture des arguments de la ligne de commande."""

import argparse

from .config import (
    DEFAULT_DEVICE,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROMPT_PROFILE,
    DEFAULT_SILENCE_MS,
    DEFAULT_TASK,
    DEFAULT_VAD_THRESHOLD,
)


def read_arguments():
    parser = argparse.ArgumentParser(
        description="Transcription automatique (VAD) avec Whisper"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Modèle Whisper (voir --list-models). Défaut : %s" % DEFAULT_MODEL,
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Afficher la liste des modèles Whisper puis quitter",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default=DEFAULT_DEVICE,
        help="Matériel de calcul (défaut : auto)",
    )
    parser.add_argument(
        "--language",
        choices=["auto", "fr", "en"],
        default=DEFAULT_LANGUAGE,
        help="auto = détection entre français et anglais (défaut), fr, en",
    )
    parser.add_argument(
        "--task",
        choices=["transcribe", "translate"],
        default=DEFAULT_TASK,
        help="transcribe = texte dans la langue parlée, translate = traduction en anglais",
    )
    parser.add_argument(
        "--prompt-profile",
        choices=["stade", "general", "aucun"],
        default=DEFAULT_PROMPT_PROFILE,
        help="Contexte lexical Whisper (défaut : stade)",
    )
    parser.add_argument(
        "--audio",
        help="Fichier audio à transcrire au lieu d'utiliser le micro",
    )
    parser.add_argument(
        "--mic",
        type=int,
        default=None,
        help="Numéro du micro (voir --list-devices). Si vide : micro par défaut",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Afficher la liste des micros disponibles puis quitter",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Sauvegarder chaque transcription dans le dossier output",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Dossier où écrire les transcriptions (défaut : %s)" % DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=DEFAULT_VAD_THRESHOLD,
        help="Seuil de probabilité de parole, entre 0 et 1 (défaut : %s)"
        % DEFAULT_VAD_THRESHOLD,
    )
    parser.add_argument(
        "--silence-ms",
        type=int,
        default=DEFAULT_SILENCE_MS,
        help="Silence (en ms) qui marque la fin d'une phrase (défaut : %s)"
        % DEFAULT_SILENCE_MS,
    )
    return parser.parse_args()
