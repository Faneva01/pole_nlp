"""Transformation d'une réponse LLM validée en action concrète."""

from __future__ import annotations

from dataclasses import dataclass, field

from .registre import RegistreLieux


@dataclass
class Action:
    intention: str
    type: str
    lieux: list[str] = field(default_factory=list)
    ids: list[int | None] = field(default_factory=list)
    message: str = ""
    parametres: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "intention": self.intention,
            "type": self.type,
            "lieux": self.lieux,
            "ids": self.ids,
            "message": self.message,
            "parametres": self.parametres,
        }


def _message_navigation(lieux: list[str], registre: RegistreLieux) -> str:
    if not lieux:
        return "Je n'ai pas compris la destination."
    if len(lieux) == 1:
        return f"Je vous emmène {registre.designation(lieux[0])}."
    destinations = ", puis ".join(
        registre.designation(lieu) for lieu in lieux
    )
    return f"Je vous emmène {destinations}."


def construire_action(reponse: dict, registre: RegistreLieux) -> Action:
    """Construit l'action à partir d'une réponse déjà au format canonique."""
    intention = reponse["intention"]
    lieux = registre.resoudre_liste(reponse.get("lieux", []))
    ids = [registre.identifiant(lieu) for lieu in lieux]
    reponse_llm = (reponse.get("reponse") or "").strip()

    if intention == "navigation":
        return Action(
            intention=intention,
            type="naviguer",
            lieux=lieux,
            ids=ids,
            message=reponse_llm or _message_navigation(lieux, registre),
            parametres={"destinations": lieux, "ids": ids},
        )

    if intention == "localiser":
        return Action(
            intention=intention,
            type="localiser",
            lieux=lieux,
            ids=ids,
            message=reponse_llm or "Voulez-vous que je vous y accompagne ?",
            parametres={"destinations": lieux, "ids": ids},
        )

    if intention == "fin":
        return Action(
            intention=intention,
            type="terminer",
            lieux=lieux,
            ids=ids,
            message=reponse_llm or "Au revoir.",
            parametres={},
        )

    if intention == "salutation":
        return Action(
            intention=intention,
            type="parler",
            lieux=lieux,
            ids=ids,
            message=reponse_llm or "Bonjour, comment puis-je vous aider ?",
            parametres={},
        )

    if intention == "information":
        return Action(
            intention=intention,
            type="parler",
            lieux=lieux,
            ids=ids,
            message=reponse_llm,
            parametres={},
        )

    return Action(
        intention=intention,
        type="parler",
        lieux=lieux,
        ids=ids,
        message=reponse_llm or "Je n'ai pas compris. Pouvez-vous reformuler ?",
        parametres={},
    )
