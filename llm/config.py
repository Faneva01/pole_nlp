import json
from pathlib import Path

import yaml

MODEL = "qwen3:1.7b"

DOSSIER = Path(__file__).parent


def _charger_yaml(nom_fichier: str) -> dict:
    with open(DOSSIER / nom_fichier, encoding="utf-8") as f:
        return yaml.safe_load(f)


LIEUX = _charger_yaml("lieux.yaml")["lieux"]
PROMPT = _charger_yaml("prompt.yaml")
SANS_EMPLACEMENT = [l["nom"] for l in LIEUX if not l.get("emplacement")]


def _article(designation: str) -> str:
    """'à l'accueil' -> "l'accueil" ; 'aux toilettes' -> 'les toilettes'."""
    for debut, rempl in (("à l'", "l'"), ("à la ", "la "), ("aux ", "les "), ("au ", "le ")):
        if designation.startswith(debut):
            return rempl + designation[len(debut):]
    return designation


def _localisation(lieu: dict) -> str:
    """Première phrase de la réponse : où se trouve le lieu."""
    art = _article(lieu["designation"])
    verbe = "se trouvent" if art.startswith("les ") else "se trouve"
    if lieu.get("emplacement"):
        phrase = f"{art} {verbe} {lieu['emplacement']}."
    else:
        phrase = f"Je connais le chemin vers {art}."
    return phrase[0].upper() + phrase[1:]


def _exemples_lieux() -> list[dict]:
    """Exemples 'localiser' générés depuis lieux.yaml : toujours cohérents avec les emplacements."""
    ex = []
    for lieu in LIEUX:
        art = _article(lieu["designation"])
        ex.append({
            "phrase": f"C'est où {art} ?",
            "sortie": {"intentions": ["localiser"], "lieux": [lieu["nom"]], "confirmation": "",
                       "reponse": f"{_localisation(lieu)} Voulez-vous que je vous y accompagne ?"},
        })
    second = LIEUX[1 % len(LIEUX)]
    ex.append({
        "phrase": f"[Contexte : accompagnement proposé vers {second['nom']}] Pouvez-vous me préciser le chemin ?",
        "sortie": {"intentions": ["localiser"], "lieux": [second["nom"]], "confirmation": "",
                   "reponse": f"{_localisation(second)} Voulez-vous que je vous y accompagne ?"},
    })
    premier = LIEUX[0]
    ex.append({
        "phrase": f"Bonjour, c'est où {_article(premier['designation'])} ?",
        "sortie": {"intentions": ["salutation", "localiser"], "lieux": [premier["nom"]], "confirmation": "",
                   "reponse": f"Bonjour ! {_localisation(premier)} Voulez-vous que je vous y accompagne ?"},
    })
    return ex


def _construire_system() -> str:
    noms = ", ".join(lieu["nom"] for lieu in LIEUX)
    intentions = "\n".join(f"- {k} : {v}" for k, v in PROMPT["intentions"].items())
    regles = "\n".join(f"- {r}" for r in PROMPT["regles"])
    empl = "\n".join(
        f"- {l['nom']} : {l['emplacement']}" for l in LIEUX if l.get("emplacement")
    )
    bloc_empl = f"Emplacements :\n{empl}\n\n" if empl else ""
    programme = "\n".join(f"- {p}" for p in PROMPT.get("programme") or [])
    bloc_programme = f"Programme :\n{programme}\n\n" if programme else ""
    tous = _exemples_lieux() + PROMPT["exemples"]
    exemples = "\n".join(
        f'"{e["phrase"]}" -> {json.dumps(e["sortie"], ensure_ascii=False)}' for e in tous
    )
    return (
        f"{PROMPT['role'].strip()}\n\n"
        f"Intentions possibles :\n{intentions}\n\n"
        f"Lieux possibles : {noms}\n\n"
        f"{bloc_empl}"
        f"{bloc_programme}"
        f"Règles :\n{regles}\n\n"
        f"Exemples :\n{exemples}\n"
    )


SYSTEM = _construire_system()