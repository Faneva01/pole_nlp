"""Prompts de contexte pour améliorer la reconnaissance du vocabulaire métier.

Whisper utilise ``initial_prompt`` comme *contexte précédent* : ce n'est pas un
modèle à instructions. Un texte comme « transcris fidèlement, sans répondre »
n'est donc pas une consigne, c'est du faux contexte. Les prompts ci-dessous
sont écrits comme du texte naturel (ce qui oriente aussi la ponctuation) et ne
contiennent que du vocabulaire attendu.

Contrainte dure : Whisper ne conserve que les PROMPT_TOKEN_LIMIT derniers tokens
du prompt ; tout ce qui précède est silencieusement supprimé. On vise donc
nettement moins (~100-150 tokens). La longueur est vérifiée au chargement
(stt.recognition.validate_prompts) et par tests/test_prompts.py.

Ils aident à reconnaître les termes mais ne garantissent rien si l'audio est
ambigu : à valider par un A/B avec benchmark_models.py (--prompt-profiles).
La voix et l'intonation des réponses relèvent du TTS, pas du STT.
"""

PROMPT_TOKEN_LIMIT = 223  # n_text_ctx // 2 - 1 pour tous les modèles Whisper
PROMPT_TOKEN_TARGET = 150  # marge de sécurité visée

PROMPTS = {
    "fr": {
        "stade": (
            "Accueil des visiteurs dans un stade. Où se trouvent la billetterie, "
            "la tribune, les gradins, le virage, la porte, l'entrée, la sortie, "
            "le contrôle d'accès, la consigne, les toilettes, l'accès PMR, "
            "l'ascenseur, le parking, la navette, la buvette, la boutique, "
            "l'infirmerie, les objets trouvés, l'espace VIP ? Mon billet, ma "
            "place, mon rang, mon siège. Penalty, corner, hors-jeu, VAR."
        ),
        "general": "Conversation courante, avec ponctuation, noms propres et nombres.",
    },
    "en": {
        "stade": (
            "Visitor assistance in a stadium. Where are the ticket office, the "
            "stand, the gate, the entrance, the exit, the security check, the "
            "cloakroom, the restrooms, the wheelchair access, the elevator, the "
            "parking, the shuttle, the snack bar, the shop, first aid, lost and "
            "found, the VIP area? My ticket, my seat, my row. Penalty, corner, "
            "offside, VAR."
        ),
        "general": "Everyday conversation, with punctuation, proper names and numbers.",
    },
}


def build_initial_prompt(profile, language):
    """Retourne le contexte adapté à la langue, ou ``None`` si désactivé."""
    if profile == "aucun":
        return None
    if profile not in {"stade", "general"}:
        raise ValueError("Profil de prompt inconnu : %s" % profile)
    language_prompts = PROMPTS.get(language, PROMPTS["fr"])
    return language_prompts[profile]
