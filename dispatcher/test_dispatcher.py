"""Tests d'alignement entre le JSON du LLM et le dispatcher.

Exécution depuis la racine du projet :
    python -m unittest dispatcher.test_dispatcher
"""

import unittest

from dispatcher.dispatcher import Dispatcher, traiter
from dispatcher.validateur import reponse_canonique


class TestAlignementJsonLlmDispatcher(unittest.TestCase):
    def test_schema_canonique_du_llm(self):
        reponse = {
            "intention": "navigation",
            "lieux": ["toilettes", "buvette"],
            "reponse": "",
        }
        resultat = traiter(reponse)
        self.assertEqual(resultat["intention"], "navigation")
        self.assertEqual(resultat["lieux"], ["toilettes", "buvette"])
        self.assertEqual(resultat["action"]["type"], "naviguer")
        self.assertEqual(resultat["action"]["ids"], [1, 2])

    def test_alias_de_champs(self):
        reponse = {
            "intent": "localiser",
            "locations": "WC",
            "response": "Je vous y accompagne.",
        }
        resultat = traiter(reponse)
        self.assertEqual(resultat["intention"], "localiser")
        self.assertEqual(resultat["lieux"], ["toilettes"])
        self.assertEqual(resultat["reponse"], "Je vous y accompagne.")

    def test_chaine_json_avec_balises_markdown(self):
        reponse = '```json\n{"intention":"info","lieux":[],"reponse":"Oui."}\n```'
        resultat = traiter(reponse)
        self.assertEqual(resultat["intention"], "information")
        self.assertEqual(resultat["action"]["type"], "parler")

    def test_intention_inconnue_est_ramenee_a_autre(self):
        self.assertEqual(
            reponse_canonique({"intention": "danser", "lieux": []})["intention"],
            "autre",
        )

    def test_champs_manquants_ont_des_valeurs_par_defaut(self):
        self.assertEqual(
            reponse_canonique({"intention": "salutation"}),
            {"intention": "salutation", "lieux": [], "reponse": ""},
        )

    def test_reponse_invalide_ne_casse_pas_le_dispatcher(self):
        resultat = traiter("pas du json")
        self.assertEqual(resultat["intention"], "autre")
        self.assertEqual(resultat["action"]["type"], "parler")
        self.assertTrue(resultat["message"])

    def test_classe_dispatcher_conserve_l_etat(self):
        dispatcher = Dispatcher()
        self.assertTrue(dispatcher.traiter({"intention": "salutation"}).get("action"))
        self.assertTrue(dispatcher.en_cours)
        dispatcher.traiter({"intention": "fin", "lieux": [], "reponse": ""})
        self.assertFalse(dispatcher.en_cours)


if __name__ == "__main__":
    unittest.main()
