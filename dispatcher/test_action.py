"""Tests de construction d'action à partir d'une réponse canonique."""

import unittest

from dispatcher.action import Action, construire_action, intention_principale
from dispatcher.registre import REGISTRE, RegistreLieux


def canonique(intentions, lieux=None, confirmation="", reponse=""):
    return {
        "intentions": intentions,
        "lieux": lieux or [],
        "confirmation": confirmation,
        "reponse": reponse,
    }


class TestConstructionAction(unittest.TestCase):
    def test_navigation_simple(self):
        action = construire_action(
            canonique(["navigation"], ["tribune B"]),
            REGISTRE,
        )
        self.assertEqual(action.type, "naviguer")
        self.assertEqual(action.intention, "navigation")
        self.assertEqual(action.message, "Je vous emmène à la tribune B.")
        self.assertEqual(action.parametres["destination"], "tribune B")
        self.assertEqual(action.parametres["destinations"], ["tribune B"])
        self.assertEqual(action.parametres["ids"], [4])

    def test_navigation_deux_lieux_depart_et_destination(self):
        action = construire_action(
            canonique(["navigation"], ["accueil", "sortie principale"]),
            REGISTRE,
        )
        self.assertEqual(action.type, "naviguer")
        self.assertEqual(action.message, "Je vous emmène à la sortie principale.")
        self.assertNotIn("puis", action.message)
        self.assertEqual(action.parametres["depart"], "accueil")
        self.assertEqual(action.parametres["destination"], "sortie principale")
        self.assertEqual(
            action.parametres["destinations"], ["accueil", "sortie principale"]
        )
        self.assertEqual(action.parametres["ids"], [0, 5])

    def test_navigation_sans_lieux_parle(self):
        action = construire_action(canonique(["navigation"]), REGISTRE)
        self.assertEqual(action.type, "parler")
        self.assertEqual(action.message, "Je n'ai pas compris la destination.")

    def test_navigation_utilise_la_memoire_sur_confirmation(self):
        action = construire_action(
            canonique(["confirmation", "navigation"], confirmation="oui"),
            REGISTRE,
            lieux_proposes=["buvette"],
        )
        self.assertEqual(action.type, "naviguer")
        self.assertEqual(action.parametres["destination"], "buvette")
        self.assertEqual(action.ids, [2])

    def test_navigation_lieu_inconnu_ne_navigue_pas(self):
        action = construire_action(
            canonique(["navigation"], ["piscine"]),
            REGISTRE,
        )
        self.assertEqual(action.type, "parler")
        self.assertEqual(
            action.message,
            "Je ne connais pas ce lieu, l'accueil pourra vous renseigner.",
        )

    def test_localiser_garde_la_reponse_du_llm(self):
        message = "Bonjour ! L'accueil se trouve au nord. Voulez-vous que je vous y accompagne ?"
        action = construire_action(
            canonique(["salutation", "localiser"], ["accueil"], reponse=message),
            REGISTRE,
        )
        self.assertEqual(action.type, "localiser")
        self.assertEqual(action.intention, "localiser")
        self.assertEqual(action.message, message)

    def test_localiser_secours_utilise_l_emplacement(self):
        action = construire_action(
            canonique(["localiser"], ["accueil"]),
            REGISTRE,
        )
        self.assertEqual(action.type, "localiser")
        self.assertIn("nord du stade", action.message)
        self.assertIn("Voulez-vous que je vous y accompagne ?", action.message)

    def test_fin(self):
        action = construire_action(canonique(["fin"]), REGISTRE)
        self.assertEqual(action.type, "terminer")
        self.assertEqual(action.message, "Au revoir.")

    def test_confirmation_seule_parle(self):
        action = construire_action(
            canonique(["confirmation"], confirmation="non"),
            REGISTRE,
        )
        self.assertEqual(action.type, "parler")
        self.assertEqual(action.confirmation, "non")

    def test_priorite_des_intentions(self):
        self.assertEqual(
            intention_principale(["salutation", "localiser"]),
            "localiser",
        )
        self.assertEqual(
            intention_principale(["confirmation", "navigation"]),
            "navigation",
        )
        self.assertEqual(intention_principale([]), "autre")

    def test_registre_expose_emplacement_et_conserve_le_secours(self):
        self.assertIn("nord du stade", REGISTRE.emplacement("accueil"))
        self.assertEqual(REGISTRE.emplacement("piscine"), "")

        secours = RegistreLieux(chemin="/chemin/inexistant/lieux.yaml")
        self.assertIn("nord du stade", secours.emplacement("accueil"))

    def test_to_dict_expose_les_nouveaux_champs(self):
        action = Action(
            intention="navigation",
            type="naviguer",
            intentions=["confirmation", "navigation"],
            lieux=["buvette"],
            ids=[2],
            confirmation="oui",
            message="Très bien, suivez-moi !",
        )
        donnees = action.to_dict()
        self.assertEqual(donnees["intentions"], ["confirmation", "navigation"])
        self.assertEqual(donnees["confirmation"], "oui")
        self.assertEqual(donnees["intention"], "navigation")


if __name__ == "__main__":
    unittest.main()
