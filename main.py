"""Assistant vocal complet : STT -> LLM -> dispatcher -> TTS.

Ordre de chargement imposé : STT (Whisper + VAD), puis LLM (Ollama),
puis TTS (Piper).

Usage :
    python main.py
    python main.py --language fr --mic 2 --save
"""

from __future__ import annotations

import argparse
import json
import unicodedata

from stt.transcriber import load_stt, transcrire_message
from llm.main import charger_modele, traiter as traiter_llm
from dispatcher.dispatcher import Dispatcher

MOTS_SORTIE = {"quitter", "exit", "q", "stop", "au revoir", "fin"}


def _normaliser(texte: str) -> str:
    texte = unicodedata.normalize("NFKD", texte or "")
    texte = "".join(c for c in texte if not unicodedata.combining(c))
    return " ".join(texte.lower().strip(" .,!?:;").split())


def lire_arguments():
    parser = argparse.ArgumentParser(description="Assistant vocal STT + LLM + TTS")
    parser.add_argument(
        "--language",
        choices=["auto", "fr", "en"],
        default="auto",
        help="Langue du STT (défaut : auto entre fr et en)",
    )
    parser.add_argument("--task", choices=["transcribe", "translate"], default="transcribe")
    parser.add_argument("--mic", type=int, default=None, help="Numéro du micro")
    parser.add_argument(
        "--save", action="store_true", help="Sauvegarder les transcriptions"
    )
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--vad-threshold", type=float, default=0.5)
    parser.add_argument("--silence-ms", type=int, default=700)
    return parser.parse_args()


def executer_action(action: dict) -> None:
    """Point d'intégration robot : à remplacer par l'appel Nav2 / ROS réel."""
    if action.get("type") in {"naviguer", "localiser"}:
        print("[ROBOT]", action.get("type"), action.get("lieux"))


def main():
    args = lire_arguments()

    print("=== 1/3 Chargement du STT ===")
    stt = load_stt(
        language=args.language,
        task=args.task,
        mic=args.mic,
        save=args.save,
        output_dir=args.output_dir,
        vad_threshold=args.vad_threshold,
        silence_ms=args.silence_ms,
    )

    print("=== 2/3 Chargement du LLM ===")
    charger_modele()

    print("=== 3/3 Chargement du TTS ===")
    voix = None
    tts = None
    try:
        from dispatcher import tts

        voix = tts.charger_voix()
    except Exception as exc:
        print("TTS indisponible, la conversation continue sans voix :", exc)

    def dire(texte: str) -> None:
        if voix is None or tts is None or not texte:
            return
        try:
            tts.parler(voix, texte)
        except Exception as exc:
            print("Erreur TTS :", exc)

    dispatcher = Dispatcher()
    print("\nAssistant prêt. Ctrl+C pour quitter.\n")

    try:
        while True:
            print("Parlez...")
            message = transcrire_message(stt)
            if not message:
                continue

            print("Vous :", message)
            if _normaliser(message) in MOTS_SORTIE:
                dire("Au revoir.")
                break

            reponse_llm = traiter_llm(message)
            print("LLM :", json.dumps(reponse_llm, ensure_ascii=False))

            resultat = dispatcher.traiter(reponse_llm)
            print(
                "Action :",
                json.dumps(resultat["action"], ensure_ascii=False, indent=2),
            )
            executer_action(resultat["action"])

            texte_a_dire = resultat.get("message") or resultat.get("reponse") or ""
            if texte_a_dire:
                print("Assistant :", texte_a_dire)
                dire(texte_a_dire)
            else:
                print("Assistant : (pas de réponse vocale)")

            if resultat["intention"] == "fin":
                break
    except KeyboardInterrupt:
        print("\nArrêt demandé.")
        dire("Au revoir.")
    finally:
        print("Au revoir !")


if __name__ == "__main__":
    main()
