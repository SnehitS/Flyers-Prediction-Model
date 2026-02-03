"""Data collection utilities for assembling features used in win/loss prediction.

This module centralizes data collection across NHL API endpoints and provides
helpers to build modeling features for upcoming games (team stats, recent
form, rest days, rosters, and optional EDGE metrics).

Functions are written to be defensive (returning empty dict/list on failure)
and easy to compose into a feature-assembly pipeline.
"""

from datetime import datetime, date, timedelta
import time
import requests
import pandas as pd
from typing import List, Dict, Any, Optional
import logging

try:
    from nhl_api import NHLAPI
except Exception:  # pragma: no cover - optional dependency
    NHLAPI = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NHL_API_BASE = "https://statsapi.web.nhl.com/api/v1"


def _format_season(season: int) -> str:
    """Return NHL season string like '20252026' for season year 2025."""
    return f"{season}{season+1}"


def fetch_schedule(start_date: Optional[date] = None,
                   end_date: Optional[date] = None,
                   team_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """Fetch schedule between two dates. Returns list of game dicts.

    Dates should be `datetime.date` objects. If omitted, the API will return
    schedule according to NHL defaults (useful but explicit dates are preferred).
    """
    try:
        params = {}
        if start_date:
            params["startDate"] = start_date.isoformat()
        if end_date:
            params["endDate"] = end_date.isoformat()
        if team_id:
            params["teamId"] = str(team_id)

        url = f"{NHL_API_BASE}/schedule"
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        dates = data.get("dates", [])
        games = []
        for d in dates:
            for g in d.get("games", []):
                games.append(g)
        logger.info(f"Fetched {len(games)} scheduled games from {start_date} to {end_date}")
        return games
    except Exception as e:
        logger.exception("Error fetching schedule")
        return []


def fetch_upcoming_games(days: int = 7) -> List[Dict[str, Any]]:
    """Fetch games from today through `days` days ahead."""
    today = date.today()
    end = today + timedelta(days=days)
    return fetch_schedule(start_date=today, end_date=end)


def fetch_team_stats(team_id: int, season: Optional[int] = None) -> Dict[str, Any]:
    """Fetch team statistics (season-level)."""
    try:
        params = {}
        if season:
            params["season"] = _format_season(season)
        url = f"{NHL_API_BASE}/teams/{team_id}"
        params["expand"] = "team.stats"
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        logger.exception("Error fetching team stats for %s", team_id)
        return {}


def fetch_player_stats(player_id: int, season: Optional[int] = None,
                       stats_type: str = "statsSingleSeason") -> Dict[str, Any]:
    """Fetch player stats. `stats_type` can be 'statsSingleSeason' or 'gameLog'."""
    try:
        params = {"stats": stats_type}
        if season:
            params["season"] = _format_season(season)
        url = f"{NHL_API_BASE}/people/{player_id}/stats"
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        logger.exception("Error fetching player stats for %s", player_id)
        return {}
    
    


        def fetch_roster_and_injuries(team_id: int, season: Optional[int] = None) -> Dict[str, Any]:
            """Fetch roster and basic status information for a team.

            The NHL API provides roster data; injury/status reports are less-centralized
            and may require combining endpoints. We return the roster and any status
            information available on the roster entries.
            """
            try:
                params = {}
                if season:
                    params["season"] = _format_season(season)
                url = f"{NHL_API_BASE}/teams/{team_id}"
                params["expand"] = "team.roster"
                resp = requests.get(url, params=params, timeout=15)
                resp.raise_for_status()
                return resp.json()
            except Exception:
                logger.exception("Error fetching roster for %s", team_id)
                return {}


        def fetch_standings(season: Optional[int] = None) -> Dict[str, Any]:
            """Fetch standings for a season or current standings if season is None."""
            try:
                params = {}
                if season:
                    params["season"] = _format_season(season)
                url = f"{NHL_API_BASE}/standings"
                resp = requests.get(url, params=params, timeout=15)
                resp.raise_for_status()
                return resp.json()
            except Exception:
                logger.exception("Error fetching standings")
                return {}


        def fetch_game_boxscore(game_pk: int) -> Dict[str, Any]:
            """Fetch boxscore for a given game primary key (gamePk)."""
            try:
                url = f"{NHL_API_BASE}/game/{game_pk}/boxscore"
                resp = requests.get(url, timeout=15)
                resp.raise_for_status()
                return resp.json()
            except Exception:
                logger.exception("Error fetching boxscore for %s", game_pk)
                return {}


        def fetch_edge_metrics(player_id: int) -> Dict[str, Any]:
            """Attempt to fetch EDGE metrics using `nhl_api` if available.

            EDGE endpoints are less stable; if `nhl_api` is not installed or doesn't
            support edge calls, this will return an empty dict.
            """
            if NHLAPI is None:
                return {}
            try:
                api = NHLAPI()
                # This is best-effort — `nhl_api` wrappers vary. Try common method names.
                if hasattr(api, "edge"):
                    return api.edge.player(player_id)
                if hasattr(api, "player_edge"):
                    return api.player_edge(player_id)
            except Exception:
                logger.exception("Error fetching EDGE metrics for %s", player_id)
            return {}


        def compute_rest_days(games: List[Dict[str, Any]]) -> Dict[int, int]:
            """Compute days since last game per team from a list of past games.

            Returns a map team_id -> rest_days (int). Assumes `games` includes
            past games with 'gameDate' and team information.
            """
            rows = []
            for g in games:
                gd = g.get("gameDate")
                try:
                    played = datetime.fromisoformat(gd.replace("Z", "+00:00")).date()
                except Exception:
                    continue
                teams = g.get("teams") or g.get("teams", {})
                # modern schedule structure places teams under 'teams'
                for side in ("home", "away"):
                    t = g.get("teams", {}).get(side, {}).get("team")
                    if t:
                        rows.append({"team_id": t.get("id"), "date": played})

            if not rows:
                return {}

            df = pd.DataFrame(rows)
            df = df.dropna().sort_values(["team_id", "date"]) 
            rest_map: Dict[int, int] = {}
            for team_id, group in df.groupby("team_id"):
                dates = group["date"].drop_duplicates().sort_values()
                if len(dates) == 0:
                    rest_map[team_id] = -1
                    continue
                last = dates.max()
                rest_map[int(team_id)] = (date.today() - last).days
            return rest_map


        def recent_form(team_id: int, games: List[Dict[str, Any]], window: int = 10) -> Dict[str, Any]:
            """Compute recent form (wins/losses/overtime) for a team from a games list.

            Returns dict with wins, losses, ot, pct over the `window` most recent games.
            """
            recent = []
            for g in sorted(games, key=lambda x: x.get("gameDate", ""), reverse=True):
                teams = g.get("teams", {})
                home = teams.get("home", {}).get("team", {}).get("id")
                away = teams.get("away", {}).get("team", {}).get("id")
                if team_id not in (home, away):
                    continue
                # determine result for team
                status = g.get("status", {}).get("abstractGameState")
                # boxscore may give winner; try to infer from linescore if present
                outcome = g.get("teams", {})
                # fallback: skip games without necessary score info
                home_score = g.get("teams", {}).get("home", {}).get("score")
                away_score = g.get("teams", {}).get("away", {}).get("score")
                if home_score is None or away_score is None:
                    continue
                if team_id == home:
                    if home_score > away_score:
                        recent.append("W")
                    elif home_score < away_score:
                        recent.append("L")
                    else:
                        recent.append("T")
                else:
                    if away_score > home_score:
                        recent.append("W")
                    elif away_score < home_score:
                        recent.append("L")
                    else:
                        recent.append("T")
                if len(recent) >= window:
                    break
            wins = recent.count("W")
            losses = recent.count("L")
            ties = recent.count("T")
            total = len(recent) if len(recent) > 0 else 1
            return {"wins": wins, "losses": losses, "ties": ties, "pct": wins / total}


        def assemble_features_for_game(game: Dict[str, Any], season: Optional[int] = None) -> Dict[str, Any]:
            """Assemble a feature dict for a single scheduled game.

            This is a light-weight, synchronous assembly that calls the NHL REST
            endpoints for team stats, rosters, and computes rest and recent form
            from available schedule information.
            """
            features: Dict[str, Any] = {}
            try:
                teams = game.get("teams", {})
                home = teams.get("home", {}).get("team", {})
                away = teams.get("away", {}).get("team", {})
                home_id = home.get("id")
                away_id = away.get("id")
                features["gamePk"] = game.get("gamePk") or game.get("gameId")
                features["gameDate"] = game.get("gameDate")

                # fetch season-level stats
                features["home_team_stats"] = fetch_team_stats(home_id, season)
                features["away_team_stats"] = fetch_team_stats(away_id, season)

                # roster & status
                features["home_roster"] = fetch_roster_and_injuries(home_id, season)
                features["away_roster"] = fetch_roster_and_injuries(away_id, season)

                # recent schedule (lookback 30 days) for rest and form
                game_date = None
                try:
                    game_date = datetime.fromisoformat(features["gameDate"].replace("Z", "+00:00")).date()
                except Exception:
                    game_date = date.today()
                lookback_start = game_date - timedelta(days=60)
                past_games = fetch_schedule(start_date=lookback_start, end_date=game_date)
                rest_map = compute_rest_days(past_games)
                features["home_rest_days"] = rest_map.get(home_id, None)
                features["away_rest_days"] = rest_map.get(away_id, None)

                features["home_recent_form"] = recent_form(home_id, past_games, window=10)
                features["away_recent_form"] = recent_form(away_id, past_games, window=10)

                return features
            except Exception:
                logger.exception("Error assembling features for game %s", game.get("gamePk"))
                return features


        def fetch_features_for_upcoming_games(days: int = 7, season: Optional[int] = None) -> List[Dict[str, Any]]:
            """Fetch upcoming games and assemble feature dicts for each game."""
            games = fetch_upcoming_games(days=days)
            features = []
            for g in games:
                try:
                    f = assemble_features_for_game(g, season=season)
                    features.append(f)
                    # polite pause to avoid hitting API too quickly
                    time.sleep(0.2)
                except Exception:
                    logger.exception("Failed to process game")
            return features


        if __name__ == "__main__":
            print("Run a quick smoke fetch for the next 3 days...")
            feats = fetch_features_for_upcoming_games(days=3)
            print(f"Assembled features for {len(feats)} games")
