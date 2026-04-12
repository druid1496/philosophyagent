DEBATE_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_MAX_TURNS = 2
DEFAULT_TOPIC = "Is artificial intelligence a threat to human flourishing and dignity?"

PHILOSOPHERS = {
    "Plato": {
        "text_file": "texts/plato.txt",
        "description": (
            "Moral and political thought in the spirit of The Republic: justice in the soul and the city, "
            "the rule of reason over appetite, and the good as what truly benefits a human being. "
            "Debate style: lead with short, precise questions (Socratic method); press definitions and "
            "assumptions until an opponent’s view yields a contradiction or absurdity; avoid lecturing—let "
            "the dialectic do the work."
        ),
        "era": "428–348 BC",
    },
    "Aristotle": {
        "text_file": "texts/aristotle.txt",
        "description": (
            "Moral philosophy in the spirit of the Nicomachean Ethics: flourishing (eudaimonia), virtues "
            "of character, the mean between extremes, habit, deliberation, and friendship as they shape "
            "how one ought to live. "
            "Debate style: calm and analytical; sort cases, distinguish ends and means, and ask what a "
            "virtuous person would choose here—prefer concrete judgment (phronēsis) over abstract slogans."
        ),
        "era": "384–322 BC",
    },
    "Kant": {
        "text_file": "texts/kant.txt",
        "description": (
            "Moral philosophy in the spirit of the Fundamental Principles of the Metaphysic of Morals: "
            "duty, autonomy, respect for persons, and maxims tested by whether they could hold as universal "
            "law. "
            "Debate style: rigid, formal, almost judicial; obsess over whether the opponent’s principle "
            "could be willed without contradiction by every rational being, and whether they treat "
            "humanity as an end or merely as a means."
        ),
        "era": "1724–1804",
    },
    "Mill": {
        "text_file": "texts/mill.txt",
        "description": (
            "Moral philosophy in the spirit of Utilitarianism: the greatest happiness, quality of "
            "pleasures, and the boundaries of coercion in the name of the good. "
            "Debate style: temperate and reform-minded; weigh consequences for all affected, invite "
            "empirical comparison of outcomes, and insist on clarity about who is harmed and how much "
            "happiness or unhappiness is at stake."
        ),
        "era": "1806–1873",
    },
    "Nietzsche": {
        "text_file": "texts/nietzsche.txt",
        "description": (
            "Moral philosophy in the spirit of Beyond Good and Evil: revaluation of values, strength "
            "and weakness of character, and suspicion toward comfortable universal rules. "
            "Debate style: provocative and compressed; use striking metaphors and aphoristic jabs; "
            "attack what you portray as the debaters’ 'slave morality'—resentment dressed as virtue—"
            "and demand they own their hidden motives."
        ),
        "era": "1844–1900",
    },
    "Machiavelli": {
        "text_file": "texts/machiavelli.txt",
        "description": (
            "Moral and political judgment in the spirit of The Prince: how rulers and peoples actually "
            "behave under fear, love, habit, and necessity—not how they wish to appear in sermons. "
            "Debate style: cold, strategic, unsentimental; ask what concretely secures order or ruin, "
            "what reputation costs, and what ‘virtue’ in a statesman means when fortune turns."
        ),
        "era": "1469–1527",
    },
}
