import json

import ollama
from pydantic import ValidationError

try:
    from .config import MODEL, SANS_EMPLACEMENT, SYSTEM
    from .models import Resultat
except ImportError:  # exécution directe : python llm/main.py
    from config import MODEL, SANS_EMPLACEMENT, SYSTEM
    from models import Resultat


def charger_modele() -> str:
    """Vérifie que le modèle Ollama existe et le garde chargé en mémoire."""
    print("Chargement du LLM", MODEL, "...")
    ollama.show(MODEL)
    # Un appel minimal force le serveur Ollama à charger les poids.
    ollama.generate(
        model=MODEL,
        prompt=".",
        keep_alive="30m",
        options={"num_predict": 1},
    )
    return MODEL


def traiter(message: str, lieux_proposes: list[str] | None = None) -> dict:
    """lieux_proposes : lieux pour lesquels le robot vient de proposer d'accompagner le visiteur."""
    if lieux_proposes:
        message = f"[Contexte : accompagnement proposé vers {', '.join(lieux_proposes)}] {message}"
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": message},
        ],
        think=False,
        format=Resultat.model_json_schema(),
        options={"temperature": 0.2, "num_predict": 300},
        keep_alive="30m",
    )
    try:
        d = Resultat.model_validate_json(response["message"]["content"]).model_dump()
        if d["intentions"] == ["navigation"]:
            d["reponse"] = ""          # garde-fou : navigation seule = pas de phrase
        return d
    except ValidationError:
        print("DEBUG :", response["done_reason"], repr(response["message"]["content"]))
        return {"intentions": ["autre"], "lieux": [], "confirmation": "",
                "reponse": "Pardon, pouvez-vous répéter ?"}


if __name__ == "__main__":
    if SANS_EMPLACEMENT:
        print("ATTENTION : emplacement manquant dans lieux.yaml pour :", ", ".join(SANS_EMPLACEMENT))
    print("Tapez votre message (ou 'quit' pour arrêter).")
    lieux_proposes: list[str] = []   # mémoire de la dernière proposition d'accompagnement

    while True:
        message = input("Vous : ").strip()
        if message.lower() in ("quit", "exit", "q"):
            break
        if not message:
            continue

        r = traiter(message, lieux_proposes)
        print(json.dumps(r, ensure_ascii=False, indent=2))
        intentions = r["intentions"]

        lieux_proposes = r["lieux"] if "localiser" in intentions else []