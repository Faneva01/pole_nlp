import json
import ollama
from pydantic import ValidationError

try:
    from .config import MODEL, SYSTEM
    from .models import Resultat
except ImportError:  # exécution directe : python llm/main.py
    from config import MODEL, SYSTEM
    from models import Resultat

REPONSE_PAR_DEFAUT = {"intention": "autre", "lieux": [], "reponse": ""}


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


def _extraire_json(contenu: str) -> dict:
    """Relit le JSON même si le modèle l'entoure de texte ou de balises Markdown."""
    contenu = (contenu or "").strip()
    if contenu.startswith("```"):
        contenu = contenu.strip("`")
        if contenu.lstrip().lower().startswith("json"):
            contenu = contenu.lstrip()[4:]
    try:
        return json.loads(contenu)
    except json.JSONDecodeError:
        debut = contenu.find("{")
        fin = contenu.rfind("}")
        if debut != -1 and fin > debut:
            return json.loads(contenu[debut : fin + 1])
        raise


def traiter(message: str) -> dict:
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": message},
        ],
        think=False,   # à la place de #think=True
        format=Resultat.model_json_schema(),
        options={"temperature": 0.2, "num_predict": 300},
        keep_alive="30m",
    )
    try:
        return Resultat.model_validate_json(
            response["message"]["content"]
        ).model_dump()
    except (ValidationError, json.JSONDecodeError, TypeError, KeyError):
        try:
            return Resultat.model_validate(
                _extraire_json(response["message"]["content"])
            ).model_dump()
        except (ValidationError, json.JSONDecodeError, TypeError, KeyError):
            print(
                "DEBUG :",
                response.get("done_reason"),
                repr(response.get("message", {}).get("content")),
            )
            return dict(REPONSE_PAR_DEFAUT)


if __name__ == "__main__":
    print("Tapez votre message (ou 'quit' pour arrêter).")
    resultats = []

    while True:
        message = input("Vous : ").strip()
        if message.lower() in ("quit", "exit", "q"):
            break
        if not message:
            continue

        r = traiter(message)
        resultats.append({"message": message, **r})
        print(json.dumps(r, ensure_ascii=False, indent=2))
