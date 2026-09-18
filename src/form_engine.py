"""
FormEngine: per-player scoring form from the football-data.org API.

For each player we track, computes:
  - goals_recent, assists_recent  (season-to-date contributions)
  - form_score                    (0-1)
  - form_multiplier               (0.85-1.15, applied to the XGBoost prediction)
  - contract_modifier             (0.65-1.00, age-based contract-years heuristic)
  - dynamic_value_eur             (base x form_multiplier x contract_modifier)

WHY THIS USES SEASON AGGREGATES RATHER THAN THE LAST FIVE MATCHES
-----------------------------------------------------------------
The original implementation walked recent fixtures and read a `goals` array off
each match to build a recency-weighted last-five-appearances score. That array
does not exist on the free tier - not on /competitions/{id}/matches, and not on
/matches/{id} either. The lookup therefore found nothing on every call and
silently handed every player a neutral 1.00 multiplier, which looked like
"no player is in unusual form" rather than like a failure.

What the free tier does expose is /competitions/{id}/scorers: goals, assists and
matches played for the season so far. That supports a genuine form signal -
contribution RATE per match played - just measured over the season to date
rather than a five-match window. Early in a season the two are nearly the same
thing; by May the rate is smoother than true recent form. That trade is worth
making because the alternative is no signal at all.

Free tier note: only goals and assists are exposed, so the signal is strongest
for attackers. Defenders and goalkeepers sit at 1.00 unless they contribute.
"""

import os, time, math, warnings, requests, urllib3
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Optional
import pandas as pd
import numpy as np
from dotenv import load_dotenv

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv()

BASE_URL = "https://api.football-data.org/v4"

# Must match football_api.COMPETITIONS. A player in a division that is not
# polled here silently gets a neutral 1.00 multiplier, which is indistinguishable
# from "not in form" rather than "not measured".
COMP_CODES = ["PL", "PD", "SA", "BL1", "FL1"]

# Form multiplier range
FORM_MIN = 0.85
FORM_MAX = 1.15

# Contribution points per match played that counts as top of the scale. A goal
# is 3 points and an assist 2, so 3.0 is roughly a goal every game - elite, and
# sustained by only a handful of players in a season. Anchoring on a fixed value
# rather than the best player in the current pool keeps multipliers comparable
# between runs; a percentile anchor would drift as the pool changes.
ELITE_POINTS_PER_MATCH = 3.0

# Below this many appearances a rate is mostly noise - two goals in one
# substitute appearance is not elite form - so early-season returns are damped
# toward neutral in proportion to matches played.
RELIABLE_MATCHES = 5

# Contract modifier based on estimated years remaining (age-based heuristic)
def _estimate_contract_years(age: float) -> float:
    if age < 22: return 4.0
    if age < 25: return 3.0
    if age < 28: return 2.5
    if age < 30: return 2.0
    if age < 32: return 1.5
    if age < 34: return 1.0
    return 0.5

def _contract_modifier(contract_years: float) -> float:
    if contract_years >= 3.0: return 1.00
    if contract_years >= 2.0: return 0.95
    if contract_years >= 1.0: return 0.82
    return 0.65


class FormEngine:
    def __init__(self, api_key: Optional[str] = None):
        self._key = api_key or os.getenv("FOOTBALL_DATA_API_KEY", "")
        self._headers = {"X-Auth-Token": self._key}
        self._last_call = 0.0

    def _get(self, endpoint: str, params: dict = None) -> dict:
        gap = time.time() - self._last_call
        if gap < 6:
            time.sleep(6 - gap)
        r = requests.get(
            f"{BASE_URL}{endpoint}",
            headers=self._headers,
            params=params or {},
            timeout=30,
            verify=False,
        )
        self._last_call = time.time()
        r.raise_for_status()
        return r.json()

    def fetch_contributions(self, season: int | None = None,
                            limit: int = 100) -> dict:
        """
        Returns dict:  player_name_lower -> {
            goals_recent, assists_recent, matches_played,
            points_per_match, form_score, form_multiplier
        }

        `season` is the starting year of a season (2026 for 2026/27). Left as
        None it returns whatever the provider considers current, which is what
        a live product wants.
        """
        results: dict[str, dict] = {}

        for comp in COMP_CODES:
            try:
                params = {"limit": limit}
                if season is not None:
                    params["season"] = season
                data = self._get(f"/competitions/{comp}/scorers", params=params)

                scorers = data.get("scorers", [])
                for s in scorers:
                    name = (s.get("player") or {}).get("name")
                    if not name:
                        continue
                    goals = s.get("goals") or 0
                    assists = s.get("assists") or 0
                    played = s.get("playedMatches") or 0
                    if played <= 0:
                        continue

                    points = 3 * goals + 2 * assists
                    rate = points / played

                    norm = min(rate / ELITE_POINTS_PER_MATCH, 1.0)
                    # Shrink toward neutral until enough matches have been played
                    # for the rate to mean anything.
                    confidence = min(played / RELIABLE_MATCHES, 1.0)
                    norm *= confidence

                    results[name.lower().strip()] = {
                        "goals_recent":     int(goals),
                        "assists_recent":   int(assists),
                        "matches_played":   int(played),
                        "points_per_match": round(rate, 3),
                        "form_score":       round(norm, 4),
                        "form_multiplier":  round(
                            FORM_MIN + norm * (FORM_MAX - FORM_MIN), 4),
                    }

                window = data.get("season", {})
                print(f"  {comp}: {len(scorers)} scorers "
                      f"(season {str(window.get('startDate'))[:10]})")

            except Exception as e:
                print(f"  {comp} form fetch failed: {e}")

        print(f"  Form data computed for {len(results)} players")
        return results

    def apply_dynamic_values(
        self,
        df: pd.DataFrame,
        form_data: dict,
        name_col: str = "player_name",
        base_col: str = "predicted_value_eur",
        age_col:  str = "age",
    ) -> pd.DataFrame:
        """
        Adds to df:
          - form_multiplier        (default 1.0 when the player has no returns)
          - contract_modifier      (age-based)
          - form_score             (0-1)
          - goals_recent, assists_recent, matches_played
          - dynamic_value_eur      (base x form_mult x contract_mod)
          - value_delta_pct        (dynamic vs base %)
        """
        df = df.copy()

        fm_list, fs_list, cm_list = [], [], []
        g_list, a_list, mp_list = [], [], []
        for _, row in df.iterrows():
            key = str(row[name_col]).lower().strip()
            fd  = form_data.get(key, {})
            fm  = fd.get("form_multiplier", 1.0)
            fs  = fd.get("form_score", 0.0)

            age = row.get(age_col, 26) if age_col in df.columns else 26
            try:
                age = float(age)
            except Exception:
                age = 26.0
            cy  = _estimate_contract_years(age)
            cm  = _contract_modifier(cy)

            fm_list.append(fm); fs_list.append(fs); cm_list.append(cm)
            g_list.append(fd.get("goals_recent", 0))
            a_list.append(fd.get("assists_recent", 0))
            mp_list.append(fd.get("matches_played", 0))

        df["form_multiplier"]   = fm_list
        df["form_score"]        = fs_list
        df["contract_modifier"] = cm_list
        df["goals_recent"]      = g_list
        df["assists_recent"]    = a_list
        df["matches_played_recent"] = mp_list

        base = pd.to_numeric(df[base_col], errors="coerce").fillna(0)
        df["dynamic_value_eur"] = (base * df["form_multiplier"] * df["contract_modifier"]).round(0)
        df["value_delta_pct"]   = ((df["dynamic_value_eur"] - base) / base.replace(0, np.nan) * 100).round(1)

        return df
