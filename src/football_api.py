"""
football-data.org API client.
Free tier (Tier 10) covers PL (PL), La Liga (PD), Serie A (SA).
Rate limit: 10 calls/min on free tier — calls are throttled here automatically.
"""

import time
import warnings
import requests
import urllib3
from typing import Optional
import os
from datetime import date
from dotenv import load_dotenv

# SSL verification is disabled for local development environments with a
# self-signed proxy cert. Only affects local data-fetch scripts, not the API.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()

BASE_URL = "https://api.football-data.org/v4"

# All five divisions the match model covers. The free tier includes each of
# these; leaving Bundesliga and Ligue 1 out was what left their players with a
# fixture prediction but no valuation.
COMPETITIONS = {
    "PL":  "Premier League",
    "PD":  "La Liga",
    "SA":  "Serie A",
    "BL1": "Bundesliga",
    "FL1": "Ligue 1",
}

# ── Seasons ───────────────────────────────────────────────────────────────────
# football-data.org labels a season by the calendar year it starts in, so
# season=2026 means the 2026-27 campaign.
#
# This was a hardcoded list ending at 2024. Because the scoring pipeline takes
# SEASONS[-1], the whole "current season" half of the product was pinned to
# 2024-25 and could never advance on its own - the dashboard was still listing
# Wirtz at Leverkusen more than a year after he joined Liverpool. Derive it from
# the date so it rolls over by itself.
FIRST_SEASON = 2021

# Months in which the new campaign is too young to value players from. A few
# matchdays in, per-90 rates rest on ~300 minutes and the valuations they
# produce swing wildly week to week, so keep scoring the completed season until
# roughly a quarter of the new one has been played. Output always carries the
# season it actually used; nothing downstream should assume.
EARLY_SEASON_MONTHS = (7, 8, 9, 10)


def current_season(today: Optional[date] = None) -> int:
    """Season label for the campaign in progress. July is the changeover month."""
    d = today or date.today()
    return d.year if d.month >= 7 else d.year - 1


def scoring_season(today: Optional[date] = None) -> int:
    """Season the value model should score. See EARLY_SEASON_MONTHS."""
    d = today or date.today()
    season = current_season(d)
    return season - 1 if d.month in EARLY_SEASON_MONTHS else season


SEASONS = list(range(FIRST_SEASON, current_season() + 1))


class FootballDataClient:
    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.getenv("FOOTBALL_DATA_API_KEY")
        if not key:
            raise ValueError("Set FOOTBALL_DATA_API_KEY in .env or pass api_key=")
        self._headers = {"X-Auth-Token": key}
        self._last_call = 0.0

    def _get(self, endpoint: str, params: dict = None) -> dict:
        # Free tier: max 10 req/min → enforce 6s gap
        gap = time.time() - self._last_call
        if gap < 6:
            time.sleep(6 - gap)
        resp = requests.get(
            f"{BASE_URL}{endpoint}",
            headers=self._headers,
            params=params or {},
            timeout=30,
            verify=False,   # local dev only — see note at module top
        )
        self._last_call = time.time()
        resp.raise_for_status()
        return resp.json()

    def get_scorers(self, competition: str, season: int, limit: int = 100) -> list[dict]:
        """Top scorers for a competition/season (includes goals, assists, minutes played)."""
        data = self._get(
            f"/competitions/{competition}/scorers",
            params={"season": season, "limit": limit},
        )
        scorers = []
        for entry in data.get("scorers", []):
            player = entry.get("player", {})
            team   = entry.get("team", {})
            stats  = entry  # goals/assists/penalties are top-level in scorers response
            scorers.append({
                "player_id":       player.get("id"),
                "player_name":     player.get("name"),
                "nationality":     player.get("nationality"),
                "position":        player.get("position"),
                "date_of_birth":   player.get("dateOfBirth"),
                "team_id":         team.get("id"),
                "team_name":       team.get("name"),
                "competition":     competition,
                "season":          season,
                "goals":           stats.get("goals", 0) or 0,
                "assists":         stats.get("assists", 0) or 0,
                "penalties":       stats.get("penalties", 0) or 0,
                "played_matches":  stats.get("playedMatches", 0) or 0,
            })
        return scorers

    def get_squad(self, team_id: int) -> list[dict]:
        """Full squad with positions and date-of-birth (for non-scorers)."""
        data = self._get(f"/teams/{team_id}")
        squad = []
        for p in data.get("squad", []):
            squad.append({
                "player_id":     p.get("id"),
                "player_name":   p.get("name"),
                "position":      p.get("position"),
                "date_of_birth": p.get("dateOfBirth"),
                "nationality":   p.get("nationality"),
                "team_id":       team_id,
                "team_name":     data.get("name"),
            })
        return squad

    def get_matches(self, competition: str, season: int, status: str = "FINISHED") -> list[dict]:
        """
        All matches for a competition/season, flattened to one row per match.

        Free tier exposes scores (full-time + half-time) but no shot/possession
        statistics, so every downstream feature must be derived from results.
        """
        data = self._get(
            f"/competitions/{competition}/matches",
            params={"season": season, "status": status},
        )
        rows = []
        for m in data.get("matches", []):
            score    = m.get("score", {}) or {}
            full     = score.get("fullTime", {}) or {}
            half     = score.get("halfTime", {}) or {}
            home     = m.get("homeTeam", {}) or {}
            away     = m.get("awayTeam", {}) or {}
            rows.append({
                "match_id":      m.get("id"),
                "competition":   competition,
                "season":        season,
                "matchday":      m.get("matchday"),
                "utc_date":      m.get("utcDate"),
                "status":        m.get("status"),
                "home_team_id":  home.get("id"),
                "home_team":     home.get("name"),
                "away_team_id":  away.get("id"),
                "away_team":     away.get("name"),
                "home_goals":    full.get("home"),
                "away_goals":    full.get("away"),
                "ht_home_goals": half.get("home"),
                "ht_away_goals": half.get("away"),
                "winner":        score.get("winner"),
            })
        return rows

    def get_competition_teams(self, competition: str, season: int) -> list[dict]:
        """All teams in a competition for a given season."""
        data = self._get(
            f"/competitions/{competition}/teams",
            params={"season": season},
        )
        return [
            {"team_id": t["id"], "team_name": t["name"], "competition": competition, "season": season}
            for t in data.get("teams", [])
        ]
