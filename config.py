DEBATE_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_MAX_TURNS = 2
DEFAULT_TOPIC = "Is artificial intelligence a threat to human flourishing and dignity?"

PHILOSOPHERS = {
    "Plato": {
        "text_file": "texts/plato.txt",
        "description": (
            "Ancient Greek philosopher (428–348 BC), student of Socrates, founder of the Academy. "
            "Author of the Republic, Phaedo, Symposium, and many other dialogues. "
            "He argued that reality consists of eternal, perfect Forms, that knowledge is recollection, "
            "and that the just soul is one in which reason governs."
        ),
        "era": "428–348 BC",
    },
    "Kant": {
        "text_file": "texts/kant.txt",
        "description": (
            "German Enlightenment philosopher (1724–1804), professor at Königsberg. "
            "Author of the Critique of Pure Reason, Groundwork of the Metaphysics of Morals, and Critique of Practical Reason. "
            "He argued that morality is grounded in pure practical reason, and famously formulated the Categorical Imperative."
        ),
        "era": "1724–1804",
    },
    "Nietzsche": {
        "text_file": "texts/nietzsche.txt",
        "description": (
            "German philosopher and cultural critic (1844–1900). "
            "Author of Thus Spoke Zarathustra, Beyond Good and Evil, and On the Genealogy of Morality. "
            "He proclaimed the death of God, critiqued slave morality, championed the will to power, "
            "and called for a revaluation of all values."
        ),
        "era": "1844–1900",
    },
}
