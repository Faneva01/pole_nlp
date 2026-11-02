"""Dispatcher : reçoit le JSON du LLM et produit une action exécutable."""

from __future__ import annotations

import json
import logging
from pathlib import Path

try:
    from .action import construire_action
    from .registre import REGISTRE, RegistreLieux
    from .validateur import FormatReponseInvalide, reponse_canonique
except ImportError:  # exécution directe depuis le dossier dispatcher
    from action import construire_action
    from registre import REGISTRE, RegistreLieux
    from validateur import FormatReponseInvalide, reponse_canonique

DOSSIER = Path(__file__).resolve().parent
JOURNAL = DOSSIER / "dispatcher.log"

logger = logging.getLogger("dispatcher")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    try:
        handler = logging.FileHandler(JOURNAL, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        )
        logger.addHandler(handler)
    except OSError:
        logger.addHandler(logging.NullHandler())


def traiter(reponse, registre: RegistreLieux | None = None) -> dict:
    """Valide la réponse du LLM puis retourne le JSON enrichi de l'action.

    Format d'entrée accepté (schéma du LLM) :
        {"intention": ..., "lieux": [...], "reponse": ...}

    Format de sortie :
        {
            "intention": ...,
            "lieux": [...],
            "reponse": "...",       # texte brut du LLM
            "action": {...},        # action pour le robot
            "message": "..."        # texte à prononcer par le TTS
        }
    """
    registre = registre or REGISTRE
    try:
        canonique = reponse_canonique(reponse)
    except (FormatReponseInvalide, TypeError, ValueError) as exc:
        logger.warning("Réponse LLM invalide : %s | %r", exc, reponse)
        canonique = {
            "intention": "autre",
            "lieux": [],
            "reponse": "Je n'ai pas compris. Pouvez-vous reformuler ?",
        }

    canonique = {
        **canonique,
        "lieux": registre.resoudre_liste(canonique["lieux"]),
    }
    action = construire_action(canonique, registre)
    resultat = {
        **canonique,
        "action": action.to_dict(),
        "message": action.message,
    }
    logger.info(
        "LLM=%s | action=%s",
        json.dumps(canonique, ensure_ascii=False),
        json.dumps(resultat["action"], ensure_ascii=False),
    )
    return resultat


def traiter_json(texte: str, registre: RegistreLieux | None = None) -> dict:
    """Variante qui accepte directement la chaîne JSON renvoyée par le LLM."""
    return traiter(texte, registre=registre)


class Dispatcher:
    """Petit objet pratique pour garder un registre et un état de session."""

    def __init__(self, registre: RegistreLieux | None = None):
        self.registre = registre or REGISTRE
        self.en_cours = False

    def traiter(self, reponse) -> dict:
        resultat = traiter(reponse, registre=self.registre)
        self.en_cours = resultat["intention"] != "fin"
        return resultat

    def dispatch(self, reponse) -> dict:
        return self.traiter(reponse)

    def executer(self, reponse) -> dict:
        return self.traiter(reponse)


dispatch = traiter
