"""Validation et normalisation du JSON renvoyé par le LLM.

Le schéma de référence est celui de llm/models.py :

    {
        "intentions": ["navigation", ...],   # liste, au moins un élément
        "lieux": ["nom du lieu", ...],
        "confirmation": "oui" | "non" | "",
        "reponse": "phrase à dire"
    }

Ce module accepte aussi des variantes fréquentes (intent, locations, response,
confirm, etc.) ainsi que l'ancien format {"intention": "navigation"} en entrée.
Il renvoie toujours les quatre clés canoniques, avec ``intentions`` sous forme
de liste non vide et ``confirmation`` normalisée en oui / non / chaîne vide.
"""

from __future__ import annotations

import json
import unicodedata

INTENTIONS = {
    "navigation",
    "localiser",
    "information",
    "confirmation",
    "salutation",
    "fin",
    "autre",
}

ALIAS_INTENTION = {
    "navigation": "navigation",
    "naviguer": "navigation",
    "navigate": "navigation",
    "aller": "navigation",
    "emmener": "navigation",
    "deplacer": "navigation",
    "move": "navigation",
    "go": "navigation",
    "localiser": "localiser",
    "localize": "localiser",
    "localisation": "localiser",
    "locate": "localiser",
    "ou est": "localiser",
    "where": "localiser",
    "information": "information",
    "info": "information",
    "question": "information",
    "renseignement": "information",
    "confirmation": "confirmation",
    "confirmer": "confirmation",
    "confirm": "confirmation",
    "oui": "confirmation",
    "non": "confirmation",
    "yes": "confirmation",
    "no": "confirmation",
    "accept": "confirmation",
    "salutation": "salutation",
    "greeting": "salutation",
    "bonjour": "salutation",
    "hello": "salutation",
    "salut": "salutation",
    "fin": "fin",
    "end": "fin",
    "quit": "fin",
    "au revoir": "fin",
    "goodbye": "fin",
    "stop": "fin",
    "autre": "autre",
    "other": "autre",
    "unknown": "autre",
    "inconnu": "autre",
    "none": "autre",
}

ALIAS_CHAMPS = {
    "intentions": "intentions",
    "intention": "intentions",
    "intent": "intentions",
    "action": "intentions",
    "type": "intentions",
    "lieux": "lieux",
    "lieu": "lieux",
    "locations": "lieux",
    "location": "lieux",
    "places": "lieux",
    "place": "lieux",
    "destinations": "lieux",
    "destination": "lieux",
    "confirmation": "confirmation",
    "confirm": "confirmation",
    "reponse": "reponse",
    "response": "reponse",
    "answer": "reponse",
    "text": "reponse",
    "texte": "reponse",
    "message": "reponse",
}

VALEURS_OUI = {"oui", "yes", "ok", "true"}
VALEURS_NON = {"non", "no", "false"}


class FormatReponseInvalide(ValueError):
    """Le JSON du LLM ne peut pas être transformé en réponse canonique."""


def _sans_accents(texte: str) -> str:
    texte = unicodedata.normalize("NFKD", str(texte))
    return "".join(char for char in texte if not unicodedata.combining(char))


def _normaliser_cle(cle: str) -> str:
    return _sans_accents(cle).lower().strip().replace("-", "_")


def _extraire_dict(brut) -> dict:
    if isinstance(brut, dict):
        return brut

    if not isinstance(brut, str):
        raise FormatReponseInvalide(
            "La réponse du LLM doit être un dictionnaire ou une chaîne JSON."
        )

    contenu = brut.strip()
    if contenu.startswith("```"):
        contenu = contenu.strip("`").strip()
        if contenu.lower().startswith("json"):
            contenu = contenu[4:].strip()

    try:
        valeur = json.loads(contenu)
    except json.JSONDecodeError:
        debut = contenu.find("{")
        fin = contenu.rfind("}")
        if debut == -1 or fin <= debut:
            raise FormatReponseInvalide("Aucun objet JSON trouvé.")
        try:
            valeur = json.loads(contenu[debut : fin + 1])
        except json.JSONDecodeError as exc:
            raise FormatReponseInvalide(f"JSON invalide : {exc}") from exc

    if not isinstance(valeur, dict):
        raise FormatReponseInvalide("Le JSON du LLM n'est pas un objet.")
    return valeur


def _champs_canoniques(donnees: dict) -> dict:
    resultat = {}
    for cle, valeur in donnees.items():
        cle_canonique = ALIAS_CHAMPS.get(_normaliser_cle(cle))
        if cle_canonique and cle_canonique not in resultat:
            resultat[cle_canonique] = valeur
    return resultat


def normaliser_intention(valeur) -> str:
    """Ramène une intention isolée à une valeur connue, sinon ``autre``."""
    if valeur is None:
        return "autre"
    cle = _sans_accents(valeur).lower().strip().replace("_", " ")
    cle = " ".join(cle.split()).strip(" .,!?:;")
    return ALIAS_INTENTION.get(cle, cle if cle in INTENTIONS else "autre")


def normaliser_intentions(valeur) -> list[str]:
    """Normalise une liste d'intentions (ou une chaîne seule) sans doublon."""
    if valeur is None:
        return ["autre"]

    if isinstance(valeur, str):
        elements = [valeur]
    elif isinstance(valeur, dict):
        elements = list(valeur.values())
    else:
        try:
            elements = list(valeur)
        except TypeError:
            elements = [valeur]

    intentions = []
    for element in elements:
        intention = normaliser_intention(element)
        if intention not in intentions:
            intentions.append(intention)
    return intentions or ["autre"]


def normaliser_confirmation(valeur) -> str:
    """Normalise une confirmation en ``oui``, ``non`` ou chaîne vide."""
    if isinstance(valeur, bool):
        return "oui" if valeur else "non"
    if valeur is None:
        return ""
    cle = _sans_accents(valeur).lower().strip().strip(" .,!?:;")
    if cle in VALEURS_OUI:
        return "oui"
    if cle in VALEURS_NON:
        return "non"
    return ""


def normaliser_lieux(valeur) -> list[str]:
    if valeur is None:
        return []

    if isinstance(valeur, str):
        valeur = valeur.strip()
        if not valeur:
            return []
        for separateur in (" puis ", " et ", ";", ","):
            valeur = valeur.replace(separateur, "\n")
        elements = valeur.splitlines()
    elif isinstance(valeur, dict):
        elements = list(valeur.values())
    else:
        try:
            elements = list(valeur)
        except TypeError:
            elements = [valeur]

    lieux = []
    for element in elements:
        if isinstance(element, dict):
            element = element.get("nom") or element.get("name") or element.get("lieu")
        if element is None:
            continue
        nom = str(element).strip()
        if nom and nom not in lieux:
            lieux.append(nom)
    return lieux


def normaliser_reponse(valeur) -> str:
    if valeur is None:
        return ""
    if isinstance(valeur, (list, tuple)):
        return " ".join(str(element).strip() for element in valeur if element).strip()
    return str(valeur).strip()


def reponse_canonique(brut) -> dict:
    """Transforme n'importe quelle variante en schéma canonique du LLM."""
    donnees = _champs_canoniques(_extraire_dict(brut))
    return {
        "intentions": normaliser_intentions(donnees.get("intentions")),
        "lieux": normaliser_lieux(donnees.get("lieux")),
        "confirmation": normaliser_confirmation(donnees.get("confirmation")),
        "reponse": normaliser_reponse(donnees.get("reponse")),
    }


def valider(brut) -> dict:
    """Alias explicite de ``reponse_canonique`` pour les appelants."""
    return reponse_canonique(brut)


def est_valide(brut) -> bool:
    """Indique si ``brut`` peut être converti au schéma canonique."""
    try:
        reponse_canonique(brut)
        return True
    except (FormatReponseInvalide, TypeError, ValueError):
        return False
