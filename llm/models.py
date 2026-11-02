from typing import Literal
from pydantic import BaseModel, Field

class Resultat(BaseModel):
    intentions: list[
        Literal[
            "navigation",
            "localiser",
            "information",
            "confirmation",
            "salutation",
            "fin",
            "autre",
        ]
    ] = Field(
        min_length=1,
        description="Intentions du visiteur (une ou plusieurs, ex. confirmation + navigation).",
    )
    lieux: list[str] = Field(
        default_factory=list,
        description="Lieux concernés, dans l'ordre. Liste vide si aucun.",
    )
    confirmation: Literal["oui", "non", ""] = Field(
        default="",
        description="oui ou non si le visiteur confirme ou refuse, sinon chaîne vide.",
    )
    reponse: str = Field(
        default="",
        description="Phrase à dire par le robot. Vide uniquement si intentions = [navigation].",
    )
