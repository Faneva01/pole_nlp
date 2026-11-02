"""Transformation d'une réponse LLM validée en action concrète."""

from __future__ import annotations

from dataclasses import dataclass, field

from .registre import RegistreLieux

PRIORITE_INTENTIONS = (
    "navigation",
    "localiser",
    "fin",
    "information",
    "confirmation",
    "salutation",
    "autre",
)


@dataclass
class Action:
    intention: str
    type: str
    intentions: list[str] = field(default_factory=list)
    lieux: list[str] = field(default_factory=list)
    ids: list[int | None] = field(default_factory=list)
    confirmation: str = ""
    message: str = ""
    parametres: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "intentions": self.intentions,
            "intention": self.intention,
            "type": self.type,
            "lieux": self.lieux,
            "ids": self.ids,
            "confirmation": self.confirmation,
            "message": self.message,
            "parametres": self.parametres,
        }


def intention_principale(intentions: list[str]) -> str:
    """Première intention présente selon l'ordre de priorité métier."""
    for intention in PRIORITE_INTENTIONS:
        if intention in intentions:
            return intention
    return "autre"


def _article(designation: str) -> str:
    """'à l'accueil' -> "l'accueil" ; 'aux toilettes' -> 'les toilettes'."""
    for debut, remplacement in (
        ("à l'", "l'"),
        ("à la ", "la "),
        ("aux ", "les "),
        ("au ", "le "),
    ):
        if designation.startswith(debut):
            return remplacement + designation[len(debut):]
    return designation


def _message_navigation(lieux: list[str], registre: RegistreLieux) -> str:
    if not lieux:
        return "Je n'ai pas compris la destination."
    return f"Je vous emmène {registre.designation(lieux[-1])}."


def _message_localisation(lieux: list[str], registre: RegistreLieux) -> str:
    if not lieux:
        return "Voulez-vous que je vous y accompagne ?"

    article = _article(registre.designation(lieux[0]))
    emplacement = registre.emplacement(lieux[0])
    if not emplacement:
        return (
            f"Je connais le chemin vers {article}. "
            "Voulez-vous que je vous y accompagne ?"
        )

    verbe = "se trouvent" if article.startswith("les ") else "se trouve"
    phrase = f"{article} {verbe} {emplacement}."
    return f"{phrase[0].upper() + phrase[1:]} Voulez-vous que je vous y accompagne ?"


def _parametres_navigation(lieux: list[str], ids: list[int | None]) -> dict:
    parametres = {}
    if len(lieux) >= 2:
        parametres["depart"] = lieux[0]
    if lieux:
        parametres["destination"] = lieux[-1]
    parametres["destinations"] = lieux
    parametres["ids"] = ids
    return parametres


def _intentions(reponse: dict) -> list[str]:
    intentions = list(reponse.get("intentions") or [])
    if not intentions:
        intentions = [reponse.get("intention") or "autre"]
    return intentions


def construire_action(
    reponse: dict,
    registre: RegistreLieux,
    lieux_proposes: list[str] | None = None,
) -> Action:
    """Construit l'action à partir d'une réponse déjà au format canonique.

    ``lieux_proposes`` est la mémoire de la dernière proposition d'accompagnement.
    Elle sert uniquement si le visiteur confirme (oui) sans redonner de lieu.
    """
    intentions = _intentions(reponse)
    intention = intention_principale(intentions)
    confirmation = str(reponse.get("confirmation") or "").strip().lower()
    lieux_bruts = list(reponse.get("lieux") or [])

    lieux = registre.resoudre_liste(lieux_bruts)
    if intention == "navigation" and not lieux and confirmation == "oui":
        lieux = registre.resoudre_liste(lieux_proposes or [])

    ids = [registre.identifiant(lieu) for lieu in lieux]
    reponse_llm = (reponse.get("reponse") or "").strip()

    def action(type_action: str, message: str = "", parametres: dict | None = None):
        return Action(
            intention=intention,
            type=type_action,
            intentions=intentions,
            lieux=lieux,
            ids=ids,
            confirmation=confirmation,
            message=message,
            parametres=parametres or {},
        )

    if intention == "navigation":
        if not lieux:
            return action("parler", "Je n'ai pas compris la destination.")
        if any(identifiant is None for identifiant in ids):
            return action(
                "parler",
                "Je ne connais pas ce lieu, l'accueil pourra vous renseigner.",
            )
        return action(
            "naviguer",
            reponse_llm or _message_navigation(lieux, registre),
            _parametres_navigation(lieux, ids),
        )

    if intention == "localiser":
        return action(
            "localiser",
            reponse_llm or _message_localisation(lieux, registre),
            {"destinations": lieux, "ids": ids},
        )

    if intention == "fin":
        return action("terminer", reponse_llm or "Au revoir.")

    if intention == "information":
        return action("parler", reponse_llm)

    if intention == "confirmation":
        return action(
            "parler",
            reponse_llm or "Je n'ai pas compris. Pouvez-vous reformuler ?",
        )

    if intention == "salutation":
        return action(
            "parler",
            reponse_llm or "Bonjour, comment puis-je vous aider ?",
        )

    return action(
        "parler",
        reponse_llm or "Je n'ai pas compris. Pouvez-vous reformuler ?",
    )
