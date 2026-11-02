"""
Chargement des modèles (Whisper et Silero VAD).

Pour tester un autre modèle Whisper, il suffit de changer son nom :
    - dans config.py (DEFAULT_MODEL),
    - ou en ligne de commande : python transcrit.py --model small
    - ou dans le code : change_model(stt, "medium")
On peut aussi donner le chemin d'un fichier .pt (modèle personnalisé).
"""

import os

import torch
import whisper
from silero_vad import load_silero_vad

# Description des modèles les plus utilisés (pour --list-models)
MODEL_DESCRIPTIONS = {
    "tiny": "Très rapide, précision faible",
    "base": "Rapide, précision correcte",
    "small": "Bon compromis vitesse / précision",
    "medium": "Précis, plus lent",
    "large-v3": "Le plus précis, le plus lent",
    "large-v3-turbo": "Presque aussi précis que large-v3, bien plus rapide",
    "turbo": "Alias de large-v3-turbo",
    "tiny.en": "Anglais uniquement",
    "base.en": "Anglais uniquement",
    "small.en": "Anglais uniquement",
    "medium.en": "Anglais uniquement",
}

# Modèles Whisper déjà chargés : (nom, device) -> modèle
_loaded_models = {}


def list_models():
    """Affiche les modèles disponibles."""
    print("Modèles Whisper disponibles :")
    for name in whisper.available_models():
        description = MODEL_DESCRIPTIONS.get(name, "")
        print("  %-16s %s" % (name, description))
    print("Un chemin vers un fichier .pt est aussi accepté.")


def check_model_name(model_name):
    if os.path.isfile(model_name):
        return
    if model_name not in whisper.available_models():
        raise ValueError(
            "Modèle inconnu : %s (voir --list-models)" % model_name
        )


def load_whisper(model_name, device, use_cache=True):
    """Charge un modèle Whisper. Un modèle déjà chargé est réutilisé."""
    check_model_name(model_name)
    key = (model_name, device)

    if use_cache and key in _loaded_models:
        return _loaded_models[key]

    print("Chargement de Whisper", model_name, "sur", device, "...")
    model = whisper.load_model(model_name, device=device)

    if use_cache:
        _loaded_models[key] = model
    return model


def load_vad():
    print("Chargement de Silero VAD (CPU) ...")
    return load_silero_vad()


def free_memory():
    """Oublie les modèles Whisper chargés et libère la mémoire GPU."""
    _loaded_models.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
