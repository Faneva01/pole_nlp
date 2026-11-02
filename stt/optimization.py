"""
Optimisation : choix du matériel, réglages PyTorch, options de décodage,
échauffement du modèle et calcul rapide de la probabilité de parole.
"""

import numpy as np
import torch

from .config import (
    CPU_THREADS,
    DECODE_OPTIONS,
    SAMPLE_RATE,
    USE_TF32,
)
from .prompts import build_initial_prompt


def choose_device(requested="auto"):
    """Retourne "cuda" ou "cpu" selon la demande et le matériel disponible."""
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA n'est pas disponible, utilisation du CPU.")
        return "cpu"

    return requested


def apply_torch_settings(device):
    """Applique les réglages PyTorch (à appeler une fois au démarrage)."""
    if CPU_THREADS is not None:
        torch.set_num_threads(CPU_THREADS)

    if device == "cuda" and USE_TF32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def use_fp16(device):
    """Le calcul en demi-précision n'est utile que sur GPU."""
    return device == "cuda"


def build_decode_options(task, device, language, prompt_profile="stade"):
    """Rassemble les options passées à model.transcribe."""
    options = dict(DECODE_OPTIONS)
    options["language"] = language
    options["task"] = task
    options["fp16"] = use_fp16(device)
    initial_prompt = build_initial_prompt(prompt_profile, language)
    if initial_prompt:
        options["initial_prompt"] = initial_prompt
    return options


def warm_up(model, device):
    """Petit calcul à blanc : le premier vrai message n'est plus ralenti.

    Passe par le même chemin que les vrais énoncés (encodeur, détection de
    langue, décodage), donc tous les noyaux CUDA utiles sont chauffés.
    """
    from .config import build_args
    from .recognition import recognize_short

    print("Échauffement du modèle ...")
    silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
    recognize_short(silence, model, build_args(language="auto"), device)


def speech_probability(vad_model, block):
    """Probabilité de parole d'un bloc audio (sans calcul de gradient)."""
    with torch.no_grad():
        return vad_model(torch.from_numpy(block), SAMPLE_RATE).item()
