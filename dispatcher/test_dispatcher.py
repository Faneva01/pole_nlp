"""Tests d'alignement entre le JSON du LLM et le dispatcher.

Exécution depuis la racine du projet :
    python -m unittest discover -s dispatcher -t .
"""

import unittest
from pathlib import Path

import yaml

from dispatcher.dispatcher import Dispatcher, traiter
from dispatcher.validateur import reponse_canonique


class TestAlignementJsonLlmDispatcher(unittest.TestCase):
    def test_navigation_seule(self):
        resultat = traiter(
            {
                "intentions": ["navigation"],
                "lieux": ["tribune B"],
                "confirmation": "",
                "reponse": "",
            }
        )
        self.assertEqual(resultat["intentions"], ["navigation"])
        self.assertEqual(resultat["intention"], "navigation")
        self.assertEqual(resultat["confirmation"], "")
        self.assertEqual(resultat["action"]["type"], "naviguer")
        self.assertEqual(resultat["message"], "Je vous emmène à la tribune B.")
        self.assertEqual(
            set(resultat),
            {
                "intentions",
                "intention",
                "lieux",
                "confirmation",
                "reponse",
                "action",
                "message",
            },
        )

    def test_confirmation_puis_navigation(self):
        resultat = traiter(
            {
                "intentions": ["confirmation", "navigation"],
                "lieux": ["buvette"],
                "confirmation": "oui",
                "reponse": "Très bien, suivez-moi !",
            }
        )
        self.assertEqual(resultat["intentions"], ["confirmation", "navigation"])
        self.assertEqual(resultat["intention"], "navigation")
        self.assertEqual(resultat["action"]["type"], "naviguer")
        self.assertEqual(resultat["message"], "Très bien, suivez-moi !")
        self.assertEqual(resultat["action"]["confirmation"], "oui")

    def test_confirmation_non(self):
        resultat = traiter(
            {
                "intentions": ["confirmation"],
                "lieux": [],
                "confirmation": "non",
                "reponse": "D'accord, je reste à votre disposition.",
            }
        )
        self.assertEqual(resultat["intention"], "confirmation")
        self.assertEqual(resultat["confirmation"], "non")
        self.assertEqual(resultat["action"]["type"], "parler")
        self.assertEqual(
            resultat["message"], "D'accord, je reste à votre disposition."
        )

    def test_confirmation_oui_sans_contexte(self):
        resultat = traiter(
            {
                "intentions": ["confirmation"],
                "lieux": [],
                "confirmation": "oui",
                "reponse": "Oui à quoi ? Dites-moi où vous voulez aller.",
            }
        )
        self.assertEqual(resultat["action"]["type"], "parler")
        self.assertEqual(
            resultat["message"], "Oui à quoi ? Dites-moi où vous voulez aller."
        )

    def test_salutation_et_localiser(self):
        resultat = traiter(
            {
                "intentions": ["salutation", "localiser"],
                "lieux": ["accueil"],
                "confirmation": "",
                "reponse": "Bonjour ! L'accueil se trouve au nord. Voulez-vous que je vous y accompagne ?",
            }
        )
        self.assertEqual(resultat["intention"], "localiser")
        self.assertEqual(resultat["action"]["type"], "localiser")
        self.assertIn("Bonjour", resultat["message"])

    def test_fin_arrete_la_boucle(self):
        dispatcher = Dispatcher()
        resultat = dispatcher.traiter(
            {
                "intentions": ["fin"],
                "lieux": [],
                "confirmation": "",
                "reponse": "Au revoir.",
            }
        )
        self.assertEqual(resultat["action"]["type"], "terminer")
        self.assertFalse(dispatcher.en_cours)

    def test_ancien_format_toujours_accepte(self):
        resultat = traiter(
            {"intention": "navigation", "lieux": ["accueil"], "reponse": ""}
        )
        self.assertEqual(resultat["intentions"], ["navigation"])
        self.assertEqual(resultat["confirmation"], "")
        self.assertEqual(resultat["action"]["type"], "naviguer")

    def test_intention_inconnue_donne_autre(self):
        resultat = traiter(
            {
                "intentions": ["danser"],
                "lieux": [],
                "confirmation": "",
                "reponse": "",
            }
        )
        self.assertEqual(resultat["intentions"], ["autre"])
        self.assertEqual(resultat["action"]["type"], "parler")

    def test_json_invalide_ne_casse_pas_le_dispatcher(self):
        resultat = traiter("pas du json")
        self.assertEqual(resultat["intentions"], ["autre"])
        self.assertEqual(resultat["action"]["type"], "parler")
        self.assertEqual(
            resultat["message"], "Je n'ai pas compris. Pouvez-vous reformuler ?"
        )

    def test_chaine_json_avec_balises_markdown(self):
        resultat = traiter(
            '```json\n{"intentions":["information"],"lieux":[],"confirmation":"","reponse":"Oui."}\n```'
        )
        self.assertEqual(resultat["intentions"], ["information"])
        self.assertEqual(resultat["action"]["type"], "parler")
        self.assertEqual(resultat["message"], "Oui.")

    def test_alias_de_champs(self):
        resultat = traiter(
            {
                "intent": "localiser",
                "locations": "WC",
                "response": "Je vous y accompagne.",
            }
        )
        self.assertEqual(resultat["intentions"], ["localiser"])
        self.assertEqual(resultat["lieux"], ["toilettes"])
        self.assertEqual(resultat["reponse"], "Je vous y accompagne.")

    def test_navigation_deux_lieux(self):
        resultat = traiter(
            {
                "intentions": ["navigation"],
                "lieux": ["accueil", "sortie principale"],
                "confirmation": "",
                "reponse": "",
            }
        )
        parametres = resultat["action"]["parametres"]
        self.assertEqual(resultat["action"]["type"], "naviguer")
        self.assertEqual(parametres["depart"], "accueil")
        self.assertEqual(parametres["destination"], "sortie principale")
        self.assertEqual(
            parametres["destinations"], ["accueil", "sortie principale"]
        )
        self.assertEqual(parametres["ids"], [0, 5])

    def test_navigation_sans_lieux_parle(self):
        resultat = traiter(
            {
                "intentions": ["navigation"],
                "lieux": [],
                "confirmation": "",
                "reponse": "",
            }
        )
        self.assertEqual(resultat["action"]["type"], "parler")
        self.assertEqual(
            resultat["message"], "Je n'ai pas compris la destination."
        )

    def test_lieu_inconnu_ne_navigue_pas(self):
        resultat = traiter(
            {
                "intentions": ["navigation"],
                "lieux": ["piscine"],
                "confirmation": "",
                "reponse": "",
            }
        )
        self.assertEqual(resultat["action"]["type"], "parler")
        self.assertEqual(
            resultat["message"],
            "Je ne connais pas ce lieu, l'accueil pourra vous renseigner.",
        )

    def test_memoire_lieux_proposes(self):
        dispatcher = Dispatcher()
        premier = dispatcher.traiter(
            {
                "intentions": ["localiser"],
                "lieux": ["toilettes"],
                "confirmation": "",
                "reponse": "Les toilettes se trouvent à droite. Voulez-vous que je vous y accompagne ?",
            }
        )
        self.assertEqual(premier["action"]["type"], "localiser")
        self.assertEqual(dispatcher.lieux_proposes, ["toilettes"])

        second = dispatcher.traiter(
            {
                "intentions": ["confirmation", "navigation"],
                "lieux": [],
                "confirmation": "oui",
                "reponse": "",
            }
        )
        self.assertEqual(second["action"]["type"], "naviguer")
        self.assertEqual(second["action"]["parametres"]["destination"], "toilettes")
        self.assertEqual(second["action"]["ids"], [1])
        self.assertEqual(dispatcher.lieux_proposes, [])

    def test_memoire_effacee_apres_une_autre_intention(self):
        dispatcher = Dispatcher()
        dispatcher.traiter(
            {
                "intentions": ["localiser"],
                "lieux": ["buvette"],
                "confirmation": "",
                "reponse": "",
            }
        )
        self.assertEqual(dispatcher.lieux_proposes, ["buvette"])
        dispatcher.traiter(
            {
                "intentions": ["salutation"],
                "lieux": [],
                "confirmation": "",
                "reponse": "Bonjour !",
            }
        )
        self.assertEqual(dispatcher.lieux_proposes, [])

    def test_lieux_yaml_identiques(self):
        dossier = Path(__file__).resolve().parent
        with open(dossier / "lieux.yaml", encoding="utf-8") as fichier:
            lieux_dispatcher = yaml.safe_load(fichier)["lieux"]
        with open(
            dossier.parent / "llm" / "lieux.yaml", encoding="utf-8"
        ) as fichier:
            lieux_llm = yaml.safe_load(fichier)["lieux"]

        self.assertEqual(lieux_dispatcher, lieux_llm)
        self.assertEqual([lieu["id"] for lieu in lieux_dispatcher], list(range(7)))
        self.assertTrue(all(lieu.get("emplacement") for lieu in lieux_dispatcher))

    def test_reponse_canonique_ne_perd_pas_les_intentions(self):
        canonique = reponse_canonique(
            {
                "intentions": ["confirmation", "navigation"],
                "lieux": ["buvette"],
                "confirmation": "oui",
                "reponse": "Très bien, suivez-moi !",
            }
        )
        self.assertEqual(canonique["intentions"], ["confirmation", "navigation"])


if __name__ == "__main__":
    unittest.main()
