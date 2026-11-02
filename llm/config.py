
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


def _construire_system() -> str:
    noms = ", ".join(lieu["nom"] for lieu in LIEUX)
    intentions = "\n".join(f"- {k} : {v}" for k, v in PROMPT["intentions"].items())
    regles = "\n".join(f"- {r}" for r in PROMPT["regles"])
    exemples = "\n".join(
        f'"{e["phrase"]}" -> {json.dumps(e["sortie"], ensure_ascii=False)}'
        for e in PROMPT["exemples"]
    )
    return (
        f"{PROMPT['role'].strip()}\n\n"
        f"Intentions possibles :\n{intentions}\n\n"
        f"Lieux possibles : {noms}\n\n"
        f"Règles :\n{regles}\n\n"
        f"Exemples :\n{exemples}\n"
    )


SYSTEM = _construire_system()
