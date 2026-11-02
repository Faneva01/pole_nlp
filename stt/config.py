"""
Paramètres d'ajustement du module speech-to-text.

Tout ce qu'on peut vouloir régler se trouve ici :
modèle par défaut, micro / VAD, décodage Whisper, matériel.
Les valeurs servent de défaut à la ligne de commande (voir cli.py).

Les seuils marqués « à calibrer » n'ont PAS été validés sur le matériel
cible : ils sont à régler avec benchmark_models.py sur des enregistrements
faits avec le micro réel du robot.
"""

import argparse

# ---------------------------------------------------------------
# Modèle
# ---------------------------------------------------------------
DEFAULT_MODEL = "large-v3-turbo"  # voir models.py pour la liste des choix

# Modèles comparés par défaut dans benchmark_models.py
BENCHMARK_MODELS = ["base", "small", "large-v3-turbo"]

# ---------------------------------------------------------------
# Audio
# ---------------------------------------------------------------
SAMPLE_RATE = 16000  # Whisper et Silero attendent du 16 kHz
BLOCK_SIZE = 512  # Silero exige 512 échantillons à 16 kHz (32 ms)
BLOCK_SECONDS = BLOCK_SIZE / SAMPLE_RATE
MAX_QUEUE_SECONDS = 60.0  # file audio bornée : au-delà, les blocs sont jetés

# ---------------------------------------------------------------
# Détection de parole (VAD) et découpage en segments
# ---------------------------------------------------------------
DEFAULT_VAD_THRESHOLD = 0.5  # seuil de probabilité de parole (0 à 1)
DEFAULT_SILENCE_MS = 700  # silence qui marque la fin d'une phrase

START_BLOCKS = 3  # blocs consécutifs de parole pour déclencher le début (~100 ms)
PRE_ROLL_SECONDS = 0.3  # audio gardé avant le déclenchement (évite de couper les mots)
# Durée MINIMALE DE PAROLE (blocs voisés uniquement, pré-roll et silence
# final exclus). Sous ce seuil le segment est ignoré (toux, claquement...).
# Un « oui » / « non » dure ~0,2 à 0,4 s : à calibrer.
MIN_SPEECH_SECONDS = 0.25
TAIL_SECONDS = 0.2  # silence final conservé (le reste est retiré du segment)
MAX_SEGMENT_SECONDS = 30.0  # limite de Whisper : le segment est coupé de force
NEGATIVE_MARGIN = 0.15  # hystérésis : seuil de silence = seuil de parole - marge

# Après une transcription, on récupère la suite de la phrase si elle a déjà
# été captée (pause en milieu de phrase). Cette valeur ajoute une attente
# supplémentaire (en s) pour une suite qui n'a pas encore commencé.
# 0.0 = aucune latence ajoutée. Valeur à essayer : 0.3 à 0.5.
MERGE_GRACE_SECONDS = 0.0

# ---------------------------------------------------------------
# Langues et tâche
# ---------------------------------------------------------------
LANGUAGES = {"fr": "Français", "en": "Anglais"}
DEFAULT_LANGUAGE = "auto"  # auto = détection entre français et anglais
DEFAULT_TASK = "transcribe"  # transcribe ou translate (traduction en anglais)
DEFAULT_PROMPT_PROFILE = "stade"  # stade, general ou aucun

# Détection automatique (à calibrer)
LANGUAGE_FALLBACK = "fr"  # langue utilisée si la détection n'est pas fiable
LANGUAGE_MIN_CONFIDENCE = 0.75  # part fr/en minimale du gagnant
LANGUAGE_SHORT_SECONDS = 1.0  # en dessous, la détection est moins fiable...
LANGUAGE_MIN_CONFIDENCE_SHORT = 0.90  # ...donc on exige davantage
LANGUAGE_MIN_MASS = 0.5  # masse de probabilité fr+en minimale (sinon autre langue)

# ---------------------------------------------------------------
# Décodage Whisper
# ---------------------------------------------------------------
# Repli en température : si le résultat est répétitif ou peu probable, on
# réessaie avec une température plus haute (comportement par défaut de
# Whisper). Un seul essai (0.0) supprime ce filet de sécurité.
TEMPERATURES = (0.0, 0.2, 0.4)
BEST_OF = 3  # utilisé seulement quand temperature > 0
COMPRESSION_RATIO_THRESHOLD = 2.4  # au-delà : texte répétitif
LOGPROB_THRESHOLD = -1.0  # en dessous : décodage peu fiable
NO_SPEECH_THRESHOLD = 0.6  # au-delà (et logprob bas) : on considère du silence

# Décodage glouton : pas de beam_size (en openai-whisper, beam_size=1 active
# quand même BeamSearchDecoder). Pour un beam search, mettre par ex. 5.
DECODE_OPTIONS = {
    "temperature": TEMPERATURES,
    "best_of": BEST_OF,
    "compression_ratio_threshold": COMPRESSION_RATIO_THRESHOLD,
    "logprob_threshold": LOGPROB_THRESHOLD,
    "no_speech_threshold": NO_SPEECH_THRESHOLD,
    "condition_on_previous_text": False,
    "verbose": False,
}

# Phrases que Whisper « invente » sur du bruit ou du silence (artefacts de ses
# données d'entraînement). Un texte qui en contient une est rejeté.
# Liste volontairement courte et spécifique : ajouter avec prudence.
HALLUCINATION_PHRASES = (
    "sous-titrage",
    "sous-titres réalisés par",
    "sous-titres par",
    "amara.org",
    "merci d'avoir regardé",
    "n'oubliez pas de vous abonner",
    "thanks for watching",
    "thank you for watching",
    "subtitles by",
    "please subscribe",
)

# ---------------------------------------------------------------
# Matériel et optimisation (voir optimization.py)
# ---------------------------------------------------------------
DEFAULT_DEVICE = "auto"  # auto, cuda ou cpu
CPU_THREADS = None  # None = valeur par défaut de PyTorch
USE_TF32 = True  # sans effet en fp16 ; ne joue que sur les calculs fp32
WARM_UP = True  # un petit calcul à blanc pour éviter la lenteur du premier message

# ---------------------------------------------------------------
# Sortie
# ---------------------------------------------------------------
DEFAULT_OUTPUT_DIR = "output"


def build_args(
    language=DEFAULT_LANGUAGE,
    task=DEFAULT_TASK,
    model=DEFAULT_MODEL,
    device=DEFAULT_DEVICE,
    mic=None,
    save=False,
    output_dir=DEFAULT_OUTPUT_DIR,
    vad_threshold=DEFAULT_VAD_THRESHOLD,
    silence_ms=DEFAULT_SILENCE_MS,
    prompt_profile=DEFAULT_PROMPT_PROFILE,
):
    """Construit les mêmes options que la ligne de commande, sans argparse."""
    return argparse.Namespace(
        language=language,
        task=task,
        model=model,
        device=device,
        mic=mic,
        save=save,
        output_dir=output_dir,
        vad_threshold=vad_threshold,
        silence_ms=silence_ms,
        prompt_profile=prompt_profile,
    )
