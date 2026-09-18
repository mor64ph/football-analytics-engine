"""
Canonical team identity across data sources.

THE PROBLEM
-----------
Every football data provider spells clubs differently:

    football-data.co.uk   Man City      Ath Bilbao   Espanol    Nott'm Forest
    football-data.org     Manchester    Athletic     RCD        Nottingham
                          City FC       Club         Espanyol   Forest FC
                                                     de Barcelona

Historical results come from one source and live fixtures from the other, so
without a mapping you cannot score an upcoming match at all - the team has no
Elo rating under the name the fixture arrives with.

THE APPROACH
------------
Normalisation alone is not enough. No amount of suffix-stripping turns
"Ath Bilbao" into "Athletic Club", because the difference is abbreviation, not
punctuation. So we use two layers:

  1. normalise()  strips accents, case, punctuation and corporate noise
                  (FC / CF / AC / UD / Calcio / founding years ...).
                  This alone resolves the easy majority.

  2. an explicit alias table for everything else.

Explicit beats clever here. A fuzzy matcher would silently pair "Real Madrid"
with "Real Sociedad" or "Real Oviedo" on some inputs, and a wrong join is far
worse than a loud failure - it corrupts ratings without ever raising an error.
Every mapping below is stated outright and `audit()` proves the coverage.
"""

import re
import unicodedata

# --------------------------------------------------------------------- #
# Canonical registry: slug -> display name
# --------------------------------------------------------------------- #

TEAMS: dict[str, str] = {
    # --- England ---
    "arsenal":        "Arsenal",
    "aston_villa":    "Aston Villa",
    "bournemouth":    "Bournemouth",
    "brentford":      "Brentford",
    "brighton":       "Brighton & Hove Albion",
    "burnley":        "Burnley",
    "cardiff":        "Cardiff City",
    "chelsea":        "Chelsea",
    "crystal_palace": "Crystal Palace",
    "everton":        "Everton",
    "fulham":         "Fulham",
    "huddersfield":   "Huddersfield Town",
    "hull":           "Hull City",
    "ipswich":        "Ipswich Town",
    "leeds":          "Leeds United",
    "leicester":      "Leicester City",
    "liverpool":      "Liverpool",
    "luton":          "Luton Town",
    "man_city":       "Manchester City",
    "man_united":     "Manchester United",
    "middlesbrough":  "Middlesbrough",
    "newcastle":      "Newcastle United",
    "norwich":        "Norwich City",
    "nottm_forest":   "Nottingham Forest",
    "sheffield_utd":  "Sheffield United",
    "southampton":    "Southampton",
    "stoke":          "Stoke City",
    "sunderland":     "Sunderland",
    "swansea":        "Swansea City",
    "tottenham":      "Tottenham Hotspur",
    "watford":        "Watford",
    "west_brom":      "West Bromwich Albion",
    "west_ham":       "West Ham United",
    "wolves":         "Wolverhampton Wanderers",

    # --- Spain ---
    "alaves":         "Deportivo Alavés",
    "almeria":        "UD Almería",
    "ath_bilbao":     "Athletic Club",
    "ath_madrid":     "Atlético Madrid",
    "barcelona":      "FC Barcelona",
    "betis":          "Real Betis",
    "cadiz":          "Cádiz CF",
    "celta":          "Celta Vigo",
    "eibar":          "SD Eibar",
    "elche":          "Elche CF",
    "espanyol":       "RCD Espanyol",
    "getafe":         "Getafe CF",
    "girona":         "Girona FC",
    "granada":        "Granada CF",
    "huesca":         "SD Huesca",
    "la_coruna":      "Deportivo La Coruña",
    "las_palmas":     "UD Las Palmas",
    "leganes":        "CD Leganés",
    "levante":        "Levante UD",
    "malaga":         "Málaga CF",
    "mallorca":       "RCD Mallorca",
    "osasuna":        "CA Osasuna",
    "oviedo":         "Real Oviedo",
    "real_madrid":    "Real Madrid",
    "sevilla":        "Sevilla FC",
    "sociedad":       "Real Sociedad",
    "sp_gijon":       "Sporting Gijón",
    "valencia":       "Valencia CF",
    "valladolid":     "Real Valladolid",
    "vallecano":      "Rayo Vallecano",
    "villarreal":     "Villarreal CF",

    # --- Italy ---
    "atalanta":       "Atalanta",
    "benevento":      "Benevento",
    "bologna":        "Bologna",
    "brescia":        "Brescia",
    "cagliari":       "Cagliari",
    "chievo":         "Chievo Verona",
    "como":           "Como",
    "cremonese":      "Cremonese",
    "crotone":        "Crotone",
    "empoli":         "Empoli",
    "fiorentina":     "Fiorentina",
    "frosinone":      "Frosinone",
    "genoa":          "Genoa",
    "inter":          "Inter Milan",
    "juventus":       "Juventus",
    "lazio":          "Lazio",
    "lecce":          "Lecce",
    "milan":          "AC Milan",
    "monza":          "Monza",
    "napoli":         "Napoli",
    "palermo":        "Palermo",
    "parma":          "Parma",
    "pescara":        "Pescara",
    "pisa":           "Pisa",
    "roma":           "AS Roma",
    "salernitana":    "Salernitana",
    "sampdoria":      "Sampdoria",
    "sassuolo":       "Sassuolo",
    "spal":           "SPAL",
    "spezia":         "Spezia",
    "torino":         "Torino",
    "udinese":        "Udinese",
    "venezia":        "Venezia",
    "verona":         "Hellas Verona",
}


# --------------------------------------------------------------------- #
# Aliases that normalisation cannot resolve on its own
# --------------------------------------------------------------------- #
#
# Only genuinely ambiguous spellings live here - abbreviations, local-language
# variants, and rebrandings. Names that normalise cleanly to their slug (e.g.
# "Arsenal FC" -> "arsenal") need no entry.

ALIASES: dict[str, list[str]] = {
    # England
    "man_city":       ["Man City", "Manchester City FC", "Man. City"],
    "man_united":     ["Man United", "Man Utd", "Manchester United FC", "Man. United"],
    "nottm_forest":   ["Nott'm Forest", "Nottingham Forest FC", "Nottm Forest", "Forest"],
    "wolves":         ["Wolverhampton Wanderers FC", "Wolverhampton"],
    "brighton":       ["Brighton & Hove Albion FC", "Brighton and Hove Albion"],
    "west_brom":      ["West Bromwich Albion FC", "West Bromwich Albion", "West Brom"],
    "west_ham":       ["West Ham United FC"],
    "newcastle":      ["Newcastle United FC"],
    "leeds":          ["Leeds United FC"],
    "leicester":      ["Leicester City FC"],
    "luton":          ["Luton Town FC"],
    "ipswich":        ["Ipswich Town FC"],
    "sheffield_utd":  ["Sheffield United FC", "Sheffield Utd"],
    "tottenham":      ["Tottenham Hotspur FC", "Spurs"],
    "bournemouth":    ["AFC Bournemouth"],
    "sunderland":     ["Sunderland AFC"],
    "cardiff":        ["Cardiff City FC"],
    "hull":           ["Hull City AFC", "Hull City"],
    "stoke":          ["Stoke City FC"],
    "swansea":        ["Swansea City AFC"],
    "norwich":        ["Norwich City FC"],
    "huddersfield":   ["Huddersfield Town AFC"],
    "middlesbrough":  ["Middlesbrough FC"],

    # Spain
    "ath_bilbao":     ["Ath Bilbao", "Athletic Club", "Athletic Bilbao"],
    "ath_madrid":     ["Ath Madrid", "Club Atlético de Madrid", "Atletico Madrid",
                       "Atlético de Madrid", "Atletico de Madrid"],
    "espanyol":       ["Espanol", "RCD Espanyol de Barcelona", "Espanyol",
                       "RCD Espanyol"],
    "sociedad":       ["Sociedad", "Real Sociedad de Fútbol", "Real Sociedad"],
    "betis":          ["Betis", "Real Betis Balompié", "Real Betis"],
    "celta":          ["Celta", "RC Celta de Vigo", "Celta de Vigo", "Celta Vigo"],
    "vallecano":      ["Vallecano", "Rayo Vallecano de Madrid", "Rayo Vallecano",
                       "Rayo"],
    "alaves":         ["Alaves", "Deportivo Alavés", "Alavés", "Deportivo Alaves"],
    "almeria":        ["Almeria", "UD Almería", "Almería"],
    "las_palmas":     ["UD Las Palmas"],
    "leganes":        ["Leganes", "CD Leganés", "Leganés"],
    "osasuna":        ["CA Osasuna"],
    "cadiz":          ["Cadiz", "Cádiz CF", "Cádiz"],
    "valladolid":     ["Real Valladolid CF", "Real Valladolid"],
    "oviedo":         ["Real Oviedo"],
    "mallorca":       ["RCD Mallorca"],
    "la_coruna":      ["La Coruna", "Deportivo La Coruña", "Deportivo",
                       "RC Deportivo de La Coruña"],
    "sp_gijon":       ["Sp Gijon", "Sporting Gijón", "Sporting de Gijón",
                       "Real Sporting de Gijón"],
    "malaga":         ["Malaga", "Málaga CF", "Málaga"],
    "real_madrid":    ["Real Madrid CF"],
    "barcelona":      ["FC Barcelona", "Barca", "Barça"],
    "levante":        ["Levante UD"],
    "eibar":          ["SD Eibar"],
    "huesca":         ["SD Huesca"],

    # Italy
    "inter":          ["Inter", "FC Internazionale Milano", "Internazionale",
                       "Inter Milan"],
    "milan":          ["Milan", "AC Milan"],
    "roma":           ["Roma", "AS Roma"],
    "lazio":          ["Lazio", "SS Lazio"],
    "napoli":         ["Napoli", "SSC Napoli"],
    "verona":         ["Verona", "Hellas Verona FC", "Hellas Verona"],
    "juventus":       ["Juventus FC"],
    "atalanta":       ["Atalanta BC"],
    "fiorentina":     ["ACF Fiorentina"],
    "bologna":        ["Bologna FC 1909"],
    "genoa":          ["Genoa CFC"],
    "torino":         ["Torino FC"],
    "udinese":        ["Udinese Calcio"],
    "cagliari":       ["Cagliari Calcio"],
    "frosinone":      ["Frosinone Calcio"],
    "empoli":         ["Empoli FC"],
    "venezia":        ["Venezia FC"],
    "monza":          ["AC Monza"],
    "como":           ["Como 1907", "Como"],
    "pisa":           ["AC Pisa 1909", "Pisa"],
    "parma":          ["Parma Calcio 1913"],
    "lecce":          ["US Lecce"],
    "sassuolo":       ["US Sassuolo Calcio", "Sassuolo Calcio"],
    "salernitana":    ["US Salernitana 1919", "Salernitana"],
    "cremonese":      ["US Cremonese"],
    "spezia":         ["Spezia Calcio"],
    "spal":           ["SPAL", "Spal"],
    "chievo":         ["Chievo", "Chievo Verona", "AC ChievoVerona"],
    "sampdoria":      ["UC Sampdoria"],
    "benevento":      ["Benevento Calcio"],
    "brescia":        ["Brescia Calcio"],
    "crotone":        ["FC Crotone"],
    "palermo":        ["US Palermo", "Palermo FC"],
    "pescara":        ["Delfino Pescara 1936"],
}


# --------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------- #

# Corporate/legal noise carried by long-form names. Stripped only as whole
# tokens, so "Milan" survives while the "AC" in "AC Milan" does not.
_NOISE = {
    "fc", "cf", "ac", "afc", "ss", "ssc", "sc", "us", "ud", "cd", "ca", "rc",
    "rcd", "sd", "bc", "cfc", "acf", "uc", "calcio", "club", "de", "the",
    "balompie", "futbol", "deportivo",
}

_YEAR = re.compile(r"^(1[89]\d{2}|20\d{2})$")


def normalise(name: str) -> str:
    """
    Reduce a club name to a comparable core.

        "Real Betis Balompié"  -> "real betis"
        "Bologna FC 1909"      -> "bologna"
        "Brighton & Hove ..."  -> "brighton hove albion"
    """
    if not name:
        return ""

    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c))   # drop accents
    s = s.lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)                          # punctuation
    tokens = [t for t in s.split() if t and t not in _NOISE and not _YEAR.match(t)]
    return " ".join(tokens)


# Lookup built once at import: normalised variant -> canonical slug.
_LOOKUP: dict[str, str] = {}

for _slug, _display in TEAMS.items():
    _LOOKUP[normalise(_display)] = _slug
    _LOOKUP[normalise(_slug.replace("_", " "))] = _slug

for _slug, _variants in ALIASES.items():
    if _slug not in TEAMS:
        raise ValueError(f"ALIASES references unknown slug: {_slug}")
    for _v in _variants:
        _LOOKUP[normalise(_v)] = _slug


# --------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------- #

def canonical(name: str) -> str | None:
    """Resolve any spelling to its canonical slug, or None if unknown."""
    if not name:
        return None
    key = normalise(name)
    if key in _LOOKUP:
        return _LOOKUP[key]

    # Last resort: a unique containment match. Requires exactly one candidate,
    # so an ambiguous input fails loudly instead of silently picking wrong.
    hits = {s for k, s in _LOOKUP.items() if k and (k in key or key in k)}
    return hits.pop() if len(hits) == 1 else None


def display(name: str) -> str:
    """Canonical display name, falling back to the input if unrecognised."""
    slug = canonical(name)
    return TEAMS[slug] if slug else str(name)


def map_series(series):
    """Vectorised canonical() for a pandas Series."""
    return series.map(canonical)


def audit(names, label: str = "source") -> dict:
    """
    Report resolution coverage for a list of names.

    Run this after adding any new data source. Unmatched teams are a silent
    correctness bug: rows quietly drop out of joins or a club splits into two
    identities, each with half its match history and a meaningless rating.
    """
    unique = sorted({str(n) for n in names if n and str(n) != "nan"})
    matched, unmatched = {}, []
    for n in unique:
        slug = canonical(n)
        if slug:
            matched[n] = slug
        else:
            unmatched.append(n)

    collisions: dict[str, list[str]] = {}
    for raw, slug in matched.items():
        collisions.setdefault(slug, []).append(raw)

    return {
        "label":      label,
        "total":      len(unique),
        "matched":    len(matched),
        "unmatched":  unmatched,
        "coverage":   round(len(matched) / len(unique), 4) if unique else 0.0,
        "merged":     {k: v for k, v in collisions.items() if len(v) > 1},
    }
