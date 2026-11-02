"""
Transcription : chargement complet du système (load_stt), transcription
d'un segment, thread de transcription et fonction transcrire_message.
"""

import time

from .audio import SpeechListener
from .config import (
    LANGUAGES,
    MERGE_GRACE_SECONDS,
    WARM_UP,
    build_args,
)
from .models import free_memory, load_vad, load_whisper
from .optimization import (
    apply_torch_settings,
    choose_device,
    warm_up,
)
from .recognition import recognize, validate_prompts
from .storage import save_text


# ---------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------
def load_stt(args=None, with_vad=True, **kwargs):
    """Charge Whisper et, si nécessaire, Silero VAD une seule fois.

    Retourne un dictionnaire à passer à ``transcrire_message``.
    ``args`` peut être le résultat de ``read_arguments`` ou de ``build_args``.
    """
    if args is None:
        args = build_args(**kwargs)

    device = choose_device(args.device)
    apply_torch_settings(device)

    whisper_model = load_whisper(args.model, device)
    for language, profile, count, state in validate_prompts(whisper_model):
        print("ATTENTION prompt %s/%s : %d tokens (%s)" % (language, profile, count, state))
    vad_model = load_vad() if with_vad else None

    if WARM_UP:
        warm_up(whisper_model, device)

    return {
        "model": whisper_model,
        "vad": vad_model,
        "device": device,
        "args": args,
        "listener": None,  # créé au premier message (flux micro persistant)
        "last_language": None,  # langue fiable la plus récente de la conversation
    }


def change_model(stt, model_name, keep_old=False):
    """Remplace le modèle Whisper sans recharger le VAD.

    keep_old=True garde l'ancien modèle en mémoire (retour instantané,
    mais plus de mémoire utilisée).
    """
    stt["model"] = None
    if not keep_old:
        free_memory()

    stt["model"] = load_whisper(model_name, stt["device"])
    stt["args"].model = model_name

    if WARM_UP:
        warm_up(stt["model"], stt["device"])
    return stt


def close_stt(stt):
    """Ferme le flux micro ouvert par ``transcrire_message`` (à appeler à l'arrêt)."""
    listener = stt.get("listener")
    if listener is not None:
        listener.close()
        stt["listener"] = None


def _get_listener(stt):
    if stt.get("listener") is None:
        listener = SpeechListener(stt["vad"], stt["args"])
        listener.start()
        stt["listener"] = listener
    return stt["listener"]


# ---------------------------------------------------------------
# Transcription d'un segment
# ---------------------------------------------------------------
def transcrire_segment(audio, model, args, device, last_language=None):
    """Transcrit un segment audio déjà capturé et retourne les informations utiles.

    Clés : texte, language, confidence, language_supported, language_uncertain,
    avg_logprob, no_speech_prob, filtered ("silence" / "hallucination" / None),
    duration (calcul), audio_duration.
    """
    return recognize(audio, model, args, device, last_language)


def print_info(info):
    flags = ""
    if not info.get("language_supported", True):
        flags += " langue non supportée ?"
    elif info.get("language_uncertain"):
        flags += " langue incertaine"
    if info.get("segments", 1) > 1:
        flags += " | %d segments fusionnés" % info["segments"]
    print(
        "[%s %.0f %% | %.1f s audio | %.1f s calcul%s]"
        % (
            LANGUAGES[info["language"]],
            info["confidence"] * 100,
            info["audio_duration"],
            info["duration"],
            flags,
        )
    )


# ---------------------------------------------------------------
# Thread de transcription
#    Il reçoit les segments de parole détectés par le VAD, ce qui
#    permet de continuer à écouter pendant que Whisper travaille.
# ---------------------------------------------------------------
def transcription_worker(segments, model, args, device):
    last_language = None
    while True:
        audio = segments.get()
        if audio is None:  # signal d'arrêt
            break

        info = transcrire_segment(audio, model, args, device, last_language)
        if info["language_supported"] and not info["language_uncertain"]:
            last_language = info["language"]
        text = info["texte"]
        if not text:
            continue

        print()
        print_info(info)
        print(text)
        print()

        if args.save:
            file_path = save_text(text, args.output_dir)
            print("Sauvegardé dans :", file_path)


# ---------------------------------------------------------------
# Une phrase à la fois (pour envoyer le texte à un LLM)
# ---------------------------------------------------------------
def _empty_info():
    return {
        "texte": "",
        "language": None,
        "confidence": 0.0,
        "language_supported": True,
        "language_uncertain": False,
        "filtered": None,
        "segments": 0,
        "duration": 0.0,
        "audio_duration": 0.0,
    }


def _merge_infos(infos):
    """Fusionne plusieurs segments d'une même phrase (pause en milieu de phrase)."""
    merged = dict(infos[0])
    merged["texte"] = " ".join(info["texte"] for info in infos)
    merged["segments"] = len(infos)
    merged["duration"] = sum(info["duration"] for info in infos)
    merged["audio_duration"] = sum(info["audio_duration"] for info in infos)
    return merged


def transcrire_message_info(stt=None, max_wait_seconds=None):
    """Capture une phrase au micro et retourne un dict (texte + métadonnées).

    Comme ``transcrire_message`` mais avec la langue, la confiance, les drapeaux
    ``language_supported`` / ``language_uncertain`` et ``segments`` : de quoi
    laisser le NLP demander une confirmation ou répondre « je parle français
    et anglais ». ``texte`` vaut "" si aucune parole n'a été détectée.

    - ``max_wait_seconds`` borne l'attente du début de la parole ; une phrase
      commencée n'est plus abandonnée quand le délai expire.
    - Le micro n'écoute que pendant cet appel : rien n'est capté pendant que le
      robot parle (pas d'auto-écoute). La suite d'une phrase déjà prononcée
      pendant la transcription est récupérée et fusionnée au lieu d'être jetée.
    """
    owns_stt = stt is None
    if owns_stt:
        stt = load_stt()

    args = stt["args"]
    listener = _get_listener(stt)
    deadline = None if max_wait_seconds is None else time.monotonic() + max_wait_seconds

    listener.begin()
    try:
        while True:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            segment = listener.next_segment(remaining)
            if segment is None:
                return _empty_info()

            infos = []
            pending = [segment]
            while pending:
                info = transcrire_segment(
                    pending.pop(0),
                    stt["model"],
                    args,
                    stt["device"],
                    stt.get("last_language"),
                )
                if info["language_supported"] and not info["language_uncertain"]:
                    stt["last_language"] = info["language"]
                if info["texte"]:
                    infos.append(info)
                pending.extend(listener.drain(MERGE_GRACE_SECONDS))

            if not infos:
                continue  # bruit / silence / hallucination filtrée : on réécoute

            merged = _merge_infos(infos)
            print_info(merged)
            if args.save:
                file_path = save_text(merged["texte"], args.output_dir)
                print("Sauvegardé dans :", file_path)
            return merged
    finally:
        listener.end()
        if owns_stt:
            close_stt(stt)


def transcrire_message(stt=None, max_wait_seconds=None):
    """Capture une phrase au micro et retourne le texte transcrit.

    C'est la fonction à appeler pour obtenir le message à envoyer au LLM.
    Si ``stt`` est omis, les modèles sont chargés automatiquement.
    Retourne une chaîne vide si aucune parole n'est détectée.
    Pour les métadonnées (langue, confiance...), voir ``transcrire_message_info``.
    """
    return transcrire_message_info(stt, max_wait_seconds)["texte"]


# Alias explicites pour un usage depuis un autre module.
transcribe_message = transcrire_message
transcribe_message_info = transcrire_message_info
