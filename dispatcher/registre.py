"""Registre des lieux connus du robot guide."""

from __future__ import annotations

import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import yaml

DOSSIER = Path(__file__).resolve().parent
FICHIER_LIEUX = DOSSIER / "lieux.yaml"
FICHIER_LIEUX_SECOURS = DOSSIER.parent / "llm" / "lieux.yaml"

ALIAS = {
    "wc": "toilettes",
    "w c": "toilettes",
    "toilette": "toilettes",
    "sanitaire": "toilettes",
    "sanitaires": "toilettes",
    "reception": "accueil",
    "bar": "buvette",
    "cafeteria": "buvette",
    "sortie": "sortie principale",
    "medical": "infirmerie",
    "poste de secours": "infirmerie",
}


def normaliser(texte: str) -> str:
    """Minuscules, sans accents, sans ponctuation superflue."""
    texte = unicodedata.normalize("NFKD", str(texte or ""))
    texte = "".join(char for char in texte if not unicodedata.combining(char))
    texte = texte.lower().replace("-", " ").replace("_", " ")
    return " ".join(texte.split()).strip(" .,!?:;")


class RegistreLieux:
    """Charge lieux.yaml et ramène les noms du LLM aux noms officiels."""

    def __init__(self, chemin: str | Path | None = None):
        self.chemin = Path(chemin) if chemin else FICHIER_LIEUX
        self.lieux = self._charger()
        self._par_nom = {normaliser(lieu["nom"]): lieu for lieu in self.lieux}
        self._par_designation = {
            normaliser(lieu.get("designation", "")): lieu for lieu in self.lieux
        }

    def _charger(self) -> list[dict]:
        chemin = self.chemin
        if not chemin.exists() or chemin.stat().st_size == 0:
            chemin = FICHIER_LIEUX_SECOURS
        with open(chemin, encoding="utf-8") as fichier:
            contenu = yaml.safe_load(fichier) or {}
        return contenu.get("lieux", [])

    @property
    def noms(self) -> list[str]:
        return [lieu["nom"] for lieu in self.lieux]

    def trouver(self, nom: str) -> dict | None:
        """Retourne le lieu officiel correspondant, ou None."""
        cle = normaliser(nom)
        if not cle:
            return None

        if cle in self._par_nom:
            return self._par_nom[cle]
        if cle in self._par_designation:
            return self._par_designation[cle]

        cle_alias = ALIAS.get(cle)
        if cle_alias and cle_alias in self._par_nom:
            return self._par_nom[cle_alias]

        meilleur = None
        meilleur_score = 0.0
        for candidat, lieu in self._par_nom.items():
            score = SequenceMatcher(None, cle, candidat).ratio()
            if score > meilleur_score:
                meilleur = lieu
                meilleur_score = score
        if meilleur_score >= 0.82:
            return meilleur
        return None

    def resoudre(self, nom: str) -> str:
        """Nom officiel si connu, sinon le nom d'origine nettoyé."""
        lieu = self.trouver(nom)
        if lieu is not None:
            return lieu["nom"]
        return normaliser(nom)

    def resoudre_liste(self, lieux: list[str]) -> list[str]:
        resultat = []
        for nom in lieux:
            nom_officiel = self.resoudre(nom)
            if nom_officiel and nom_officiel not in resultat:
                resultat.append(nom_officiel)
        return resultat

    def identifiant(self, nom: str) -> int | None:
        lieu = self.trouver(nom)
        return None if lieu is None else lieu.get("id")

    def designation(self, nom: str) -> str:
        lieu = self.trouver(nom)
        if lieu is None:
            return nom
        return lieu.get("designation", lieu["nom"])

    def emplacement(self, nom: str) -> str:
        """Emplacement décrit dans lieux.yaml, ou chaîne vide si le lieu est inconnu."""
        lieu = self.trouver(nom)
        if lieu is None:
            return ""
        return lieu.get("emplacement") or ""


REGISTRE = RegistreLieux()
