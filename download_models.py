"""
Téléchargement et vérification de tous les modèles du projet.

À placer à la racine du projet, à côté de main.py.

Usage :
    python download_models.py                 (télécharge tout : STT + LLM + TTS)
    python download_models.py --list          (affiche ce qui est déjà installé)
    python download_models.py --stt           (modèles Whisper de la liste ci-dessous)
    python download_models.py --stt small medium
    python download_models.py --llm qwen3:0.6b llama3.1:8b
    python download_models.py --tts fr_FR-siwis-low
    python download_models.py --stt --llm     (plusieurs catégories)

Silero VAD n'a rien à télécharger : ses poids sont dans le paquet pip silero-vad.
"""

import argparse
import os
import urllib.request
from pathlib import Path

RACINE = Path(__file__).resolve().parent

# ---------------------------------------------------------------
# Modèles par défaut (à modifier selon vos besoins)
# ---------------------------------------------------------------
# STT : doit contenir le modèle de stt/config.py (DEFAULT_MODEL)
WHISPER_MODELS = ["large-v3-turbo"]

# LLM : doit contenir le modèle de llm/config.py (MODEL)
# Autres modèles de llm/model.txt : "qwen3:0.6b", "llama3.1:8b"
OLLAMA_MODELS = ["qwen3:1.7b"]

# TTS : voix Piper, dans le dossier dispatcher/voix
PIPER_VOICES = ["fr_FR-siwis-medium"]
DOSSIER_VOIX = RACINE / "dispatcher" / "voix"
URL_VOIX = "https://huggingface.co/rhasspy/piper-voices/resolve/main"


# ---------------------------------------------------------------
# 1. STT : Whisper
# ---------------------------------------------------------------
def dossier_cache_whisper():
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "whisper"


def telecharger_whisper(noms):
    import whisper

    for nom in noms:
        print("\n[STT] Whisper", nom)
        # load_model télécharge le fichier s'il est absent du cache.
        # On charge sur le CPU pour ne pas dépendre du GPU.
        modele = whisper.load_model(nom, device="cpu")
        del modele
        print("[STT] OK :", nom)


# ---------------------------------------------------------------
# 2. LLM : Ollama (le serveur Ollama doit être lancé)
# ---------------------------------------------------------------
def telecharger_ollama(noms):
    import ollama

    for nom in noms:
        print("\n[LLM] Ollama", nom)
        try:
            dernier_statut = ""
            for etat in ollama.pull(nom, stream=True):
                statut = etat.get("status", "")
                total = etat.get("total")
                fait = etat.get("completed")

                if total and fait:
                    print("\r  %s : %.0f %%" % (statut[:30], fait * 100 / total), end="", flush=True)
                elif statut != dernier_statut:
                    print("\n  " + statut, end="", flush=True)
                dernier_statut = statut
            print()
            print("[LLM] OK :", nom)
        except Exception as erreur:
            print("\n[LLM] Échec pour", nom, ":", erreur)
            print("      Vérifiez que le serveur Ollama est lancé.")


# ---------------------------------------------------------------
# 3. TTS : voix Piper (fichiers .onnx + .onnx.json)
# ---------------------------------------------------------------
def adresse_voix(nom):
    """fr_FR-siwis-medium -> .../fr/fr_FR/siwis/medium/fr_FR-siwis-medium"""
    langue, locuteur, qualite = nom.split("-")
    famille = langue.split("_")[0]
    return "%s/%s/%s/%s/%s/%s" % (URL_VOIX, famille, langue, locuteur, qualite, nom)


def afficher_progression(nb_blocs, taille_bloc, taille_totale):
    if taille_totale <= 0:
        return
    pourcent = min(100, nb_blocs * taille_bloc * 100 / taille_totale)
    print("\r  %.0f %%" % pourcent, end="", flush=True)


def telecharger_fichier(adresse, destination):
    if destination.is_file() and destination.stat().st_size > 0:
        print("  Déjà présent :", destination.name)
        return

    print("  Téléchargement :", destination.name)
    temporaire = destination.with_name(destination.name + ".part")
    urllib.request.urlretrieve(adresse, temporaire, afficher_progression)
    temporaire.replace(destination)
    print()


def telecharger_voix(noms):
    DOSSIER_VOIX.mkdir(parents=True, exist_ok=True)

    for nom in noms:
        print("\n[TTS] Piper", nom)
        try:
            base = adresse_voix(nom)
            telecharger_fichier(base + ".onnx", DOSSIER_VOIX / (nom + ".onnx"))
            telecharger_fichier(base + ".onnx.json", DOSSIER_VOIX / (nom + ".onnx.json"))
            print("[TTS] OK :", nom)
        except Exception as erreur:
            print("\n[TTS] Échec pour", nom, ":", erreur)


# ---------------------------------------------------------------
# 4. État des modèles installés
# ---------------------------------------------------------------
def afficher_etat():
    print("=== STT : Whisper (%s) ===" % dossier_cache_whisper())
    fichiers = sorted(dossier_cache_whisper().glob("*.pt"))
    if not fichiers:
        print("  (aucun)")
    for fichier in fichiers:
        print("  %-30s %6.0f Mo" % (fichier.name, fichier.stat().st_size / 1e6))

    print("\n=== LLM : Ollama ===")
    try:
        import ollama

        reponse = ollama.list()
        modeles = reponse["models"]
        if not modeles:
            print("  (aucun)")
        for modele in modeles:
            nom = modele.get("model") or modele.get("name")
            taille = modele.get("size") or 0
            print("  %-30s %6.0f Mo" % (nom, taille / 1e6))
    except Exception as erreur:
        print("  Ollama injoignable :", erreur)

    print("\n=== TTS : Piper (%s) ===" % DOSSIER_VOIX)
    voix = sorted(DOSSIER_VOIX.glob("*.onnx")) if DOSSIER_VOIX.exists() else []
    if not voix:
        print("  (aucune)")
    for fichier in voix:
        print("  %-30s %6.0f Mo" % (fichier.name, fichier.stat().st_size / 1e6))


# ---------------------------------------------------------------
# Programme principal
# ---------------------------------------------------------------
def choisir(valeur, par_defaut):
    """None = option absente, [] = option sans nom (liste par défaut)."""
    if valeur is None:
        return []
    if len(valeur) == 0:
        return par_defaut
    return valeur


def main():
    parser = argparse.ArgumentParser(description="Télécharger les modèles du projet")
    parser.add_argument("--list", action="store_true", help="Afficher les modèles installés")
    parser.add_argument("--stt", nargs="*", metavar="MODELE", help="Modèles Whisper")
    parser.add_argument("--llm", nargs="*", metavar="MODELE", help="Modèles Ollama")
    parser.add_argument("--tts", nargs="*", metavar="VOIX", help="Voix Piper")
    args = parser.parse_args()

    if args.list:
        afficher_etat()
        return

    rien_choisi = args.stt is None and args.llm is None and args.tts is None
    if rien_choisi:
        args.stt, args.llm, args.tts = [], [], []  # tout, avec les listes par défaut

    telecharger_whisper(choisir(args.stt, WHISPER_MODELS))
    telecharger_ollama(choisir(args.llm, OLLAMA_MODELS))
    telecharger_voix(choisir(args.tts, PIPER_VOICES))

    print("\nTerminé. Vérification : python download_models.py --list")


if __name__ == "__main__":
    main()