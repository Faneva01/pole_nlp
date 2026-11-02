"""Tests du validateur : schéma canonique du LLM (nouveau et ancien format)."""

import unittest

from dispatcher.validateur import (
    INTENTIONS,
    FormatReponseInvalide,
    est_valide,
    normaliser_confirmation,
    normaliser_intentions,
    reponse_canonique,
)


class TestValidateurNouveauSchema(unittest.TestCase):
    def test_schema_canonique_complet(self):
        reponse = {
            "intentions": ["confirmation", "navigation"],
            "lieux": ["buvette"],
            "confirmation": "oui",
            "reponse": "Très bien, suivez-moi !",
        }
        self.assertEqual(
            reponse_canonique(reponse),
            {
                "intentions": ["confirmation", "navigation"],
                "lieux": ["buvette"],
                "confirmation": "oui",
                "reponse": "Très bien, suivez-moi !",
            },
        )

    def test_ancien_format_intention_unique_est_accepte(self):
        canonique = reponse_canonique({"intention": "navigation", "lieux": ["accueil"]})
        self.assertEqual(canonique["intentions"], ["navigation"])
        self.assertEqual(canonique["confirmation"], "")

    def test_intentions_en_chaine_seule(self):
        self.assertEqual(
            reponse_canonique({"intentions": "localiser"})["intentions"],
            ["localiser"],
        )

    def test_intentions_dedoublonnees_dans_l_ordre(self):
        self.assertEqual(
            normaliser_intentions(["salutation", "navigation", "salutation"]),
            ["salutation", "navigation"],
        )

    def test_intentions_vides_ou_inconnues_donnent_autre(self):
        self.assertEqual(normaliser_intentions([]), ["autre"])
        self.assertEqual(normaliser_intentions(None), ["autre"])
        self.assertEqual(normaliser_intentions("danser"), ["autre"])
        self.assertEqual(
            reponse_canonique({"intentions": ["danser"], "lieux": []})["intentions"],
            ["autre"],
        )

    def test_toutes_les_intentions_connues_sont_conservees(self):
        for intention in INTENTIONS:
            with self.subTest(intention=intention):
                self.assertEqual(
                    reponse_canonique({"intentions": [intention]})["intentions"],
                    [intention],
                )

    def test_confirmations_normalisees(self):
        self.assertEqual(normaliser_confirmation("oui"), "oui")
        self.assertEqual(normaliser_confirmation("YES"), "oui")
        self.assertEqual(normaliser_confirmation("Ok"), "oui")
        self.assertEqual(normaliser_confirmation(True), "oui")
        self.assertEqual(normaliser_confirmation("non"), "non")
        self.assertEqual(normaliser_confirmation("False"), "non")
        self.assertEqual(normaliser_confirmation("peut-être"), "")
        self.assertEqual(normaliser_confirmation(None), "")

    def test_alias_du_champ_confirmation(self):
        self.assertEqual(
            reponse_canonique({"confirm": "yes"})["confirmation"],
            "oui",
        )
        self.assertEqual(
            reponse_canonique({"confirmation": "NO"})["confirmation"],
            "non",
        )

    def test_alias_de_champs(self):
        canonique = reponse_canonique(
            {
                "intent": "localiser",
                "locations": "WC",
                "response": "Je vous y accompagne.",
            }
        )
        self.assertEqual(canonique["intentions"], ["localiser"])
        self.assertEqual(canonique["lieux"], ["WC"])
        self.assertEqual(canonique["reponse"], "Je vous y accompagne.")

    def test_champs_manquants_ont_des_valeurs_par_defaut(self):
        self.assertEqual(
            reponse_canonique({"intention": "salutation"}),
            {
                "intentions": ["salutation"],
                "lieux": [],
                "confirmation": "",
                "reponse": "",
            },
        )
        self.assertEqual(
            reponse_canonique({}),
            {
                "intentions": ["autre"],
                "lieux": [],
                "confirmation": "",
                "reponse": "",
            },
        )

    def test_chaine_json_avec_balises_markdown(self):
        canonique = reponse_canonique(
            '```json\n{"intentions":["information"],"lieux":[],"confirmation":"","reponse":"Oui."}\n```'
        )
        self.assertEqual(canonique["intentions"], ["information"])
        self.assertEqual(canonique["reponse"], "Oui.")

    def test_json_invalide_leve_une_erreur_claire(self):
        with self.assertRaises(FormatReponseInvalide):
            reponse_canonique("pas du json")
        self.assertFalse(est_valide("pas du json"))


if __name__ == "__main__":
    unittest.main()
