"""
Team-name matching shared by the odds fetcher, the grader and the live tracker.

Three different modules used to compare club names with three slightly different
rules, so a fixture could be found by one and missed by another. This is the one
implementation: strip accents, apply the alias table, then compare.
"""
import re
import unicodedata
from difflib import SequenceMatcher

TEAM_ALIASES = {
    "Nott'm Forest": ["Nottingham Forest", "Nottingham"],
    "Man City": ["Manchester City"],
    "Man United": ["Manchester Utd", "Manchester United"],
    "M'gladbach": ["B. Monchengladbach", "Monchengladbach", "Gladbach"],
    "FC Koln": ["Koln", "Cologne"],
    "Schalke 04": ["Schalke"],
    "Ein Frankfurt": ["Eintracht Frankfurt", "Frankfurt"],
    "Bayern Munich": ["Bayern Munich", "Bayern"],
    "Ath Bilbao": ["Athletic Bilbao"],
    "Vallecano": ["Rayo Vallecano"],
    "Sociedad": ["Real Sociedad", "Sociedad"],
    "Paris SG": ["Paris Saint-Germain", "PSG"],
    "Paris FC": ["Paris FC"],
    "Inter": ["Inter Milan", "Inter"],
    "Milan": ["AC Milan", "Milan"],
    "Ath Madrid": ["Atletico Madrid", "Atletico"],
    "Betis": ["Real Betis"],
    "Espanol": ["Espanyol"],
    "St Etienne": ["Saint-Etienne", "St Etienne"],
    "Celta": ["Celta Vigo"],
    # European nights
    "Leverkusen": ["Bayer Leverkusen"],
    "OFI": ["OFI Crete"],
    "Olympiakos": ["Olympiacos", "Olympiacos Piraeus"],
    "St. Gilloise": ["Union Saint-Gilloise", "Union SG", "St Gilloise", "Royale Union SG"],
    "Hapoel Be'er Sheva": ["Hapoel Beer Sheva", "H. Beer Sheva", "Beer Sheva"],
    "Viktoria Plzeň": ["Viktoria Plzen", "Plzen"],
    "Red Bull Salzburg": ["Salzburg", "RB Salzburg"],
    "Dinamo Zagreb": ["Din. Zagreb"],
    "Ferencváros": ["Ferencvaros"],
    "Lech Poznań": ["Lech Poznan"],
    "Sparta Prague": ["Sparta Praha"],
    "Levski Sofia": ["Levski"],
    "Sturm Graz": ["Sturm"],
    "Lillestrøm": ["Lillestrom"],
    "Jagiellonia Białystok": ["Jagiellonia Bialystok", "Jagiellonia"],
    "Besiktas": ["Beşiktaş"],
    "AZ Alkmaar": ["AZ"],
    "Nijmegen": ["NEC", "NEC Nijmegen"],
    "Torreense": ["Torreense"],
    "Crystal Palace": ["Crystal Palace"],
}


def strip_accents(text: str) -> str:
    """'Plzeň' -> 'Plzen'. Dropping non-ASCII instead would give 'Plze'."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", strip_accents(str(name)).lower())


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def side_similarity(a: str, b: str) -> float:
    """Best similarity across the alias variants of one club name."""
    best = similarity(a, b)
    for variant in TEAM_ALIASES.get(a, []):
        best = max(best, similarity(variant, b))
    return best
