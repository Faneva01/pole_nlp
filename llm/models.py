
from typing import Literal

from pydantic import BaseModel, Field


class Resultat(BaseModel):
    intention: Literal[
        "navigation", "localiser", "information", "salutation", "fin", "autre"
    ] = Field(description="Intention principale du visiteur")
    lieux: list[str] = Field(
        default_factory=list,
        description="Lieux demandés, dans l'ordre. Liste vide si aucun.",
    )
    reponse: str = Field(
        default="",
        description="Phrase courte à dire, seulement pour information, salutation, autre.",
    )
