"""NHL data collection with on-disk API caching.

Schedule walks use the weekly endpoint (7 days per request). Responses are
stored under data/raw/cache/ so rebuilds and later feature work reuse them.
"""

from datetime import datetime, date, timedelta
import time
from typing import List, Dict, Any, Optional, Tuple
import logging

from nhlpy import NHLClient

import api_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nhl_client = NHLClient()

REGULAR_SEASON_GAME_TYPE = 2
FINAL_STATES = {"OFF", "FINAL"}
LOOKBACK_DAYS = 60
FORM_WINDOW = 10
_MISS_PAUSE_SEC = 0.05


def _format_season(season: int) -> str:
    """Return NHL season string like '20252026' for season year 2025."""
    return f"{season}{season + 1}"


def _team_abbr(team: Optional[Dict[str, Any]]) -> Optional[str]:
    """Abbreviation from a homeTeam/awayTeam payload."""
    if not isinstance(team, dict):
        return None
    abbr = team.get("abbrev") or team.get("teamAbbrev") or team.get("abbreviation")
    if isinstance(abbr, dict):
        abbr = abbr.get("default")
    if not abbr:
        return None
    return str(abbr).upper()


def _game_pk(game: Dict[str, Any]) -> Optional[int]:
    pk = game.get("id") or game.get("gamePk") or game.get("gameId")
    try:
        return int(pk) if pk is not None else None
    except (TypeError, ValueError):
        return None


def _game_date(game: Dict[str, Any]) -> Optional[date]:
    stamped = game.get("_scheduleDate") or game.get("gameDate")
    if stamped:
        try:
            return date.fromisoformat(str(stamped)[:10])
        except ValueError:
            pass
    utc = game.get("startTimeUTC") or ""
    if utc:
        try:
            return datetime.fromisoformat(str(utc).replace("Z", "+00:00")).date()
        except ValueError:
            return None
    return None


def _is_final(game: Dict[str, Any]) -> bool:
    return str(game.get("gameState") or "").upper() in FINAL_STATES


def _is_regular_season(game: Dict[str, Any]) -> bool:
    gt = game.get("gameType")
    try:
        return int(gt) == REGULAR_SEASON_GAME_TYPE
    except (TypeError, ValueError):
        pk = _game_pk(game)
        if pk is None:
            return False
        # NHL id: YYYYTTnnnn — TT 02 is regular season
        s = f"{pk:010d}"
        return s[4:6] == "02"


def _pause_on_miss() -> None:
    if api_cache.misses > 0:
        time.sleep(_MISS_PAUSE_SEC)


def _fetch_weekly_schedule(week_date: date) -> Dict[str, Any]:
    key = week_date.isoformat()
    before = api_cache.misses
    payload = api_cache.cached(
        "schedule_week",
        key,
        lambda: nhl_client.schedule.weekly_schedule(date=key) or {},
    )
    if api_cache.misses > before:
        _pause_on_miss()
    return payload or {}


def _games_from_weekly(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    games: List[Dict[str, Any]] = []
    for day in payload.get("gameWeek") or []:
        day_date = day.get("date")
        for raw in day.get("games") or []:
            if not isinstance(raw, dict):
                continue
            game = dict(raw)
            if day_date:
                game["_scheduleDate"] = day_date
            games.append(game)
    return games


def fetch_schedule(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    team_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Fetch schedule between two dates using cached weekly nhlpy calls.

    Each game is stamped with `_scheduleDate` (NHL calendar date).
    `team_id` is unused (kept for call-site compatibility).
    """
    del team_id
    try:
        if start_date is None and end_date is None:
            today = date.today()
            start_date = today
            end_date = today
        elif start_date is None:
            start_date = end_date
        elif end_date is None:
            end_date = start_date

        games: List[Dict[str, Any]] = []
        seen = set()
        cursor = start_date
        safety = 0
        while cursor <= end_date and safety < 400:
            safety += 1
            payload = _fetch_weekly_schedule(cursor)
            for game in _games_from_weekly(payload):
                gd = _game_date(game)
                if gd is None or gd < start_date or gd > end_date:
                    continue
                pk = _game_pk(game)
                if pk is not None:
                    if pk in seen:
                        continue
                    seen.add(pk)
                games.append(game)

            next_start = payload.get("nextStartDate")
            if next_start:
                try:
                    nxt = date.fromisoformat(str(next_start)[:10])
                except ValueError:
                    nxt = cursor + timedelta(days=7)
                cursor = nxt if nxt > cursor else cursor + timedelta(days=7)
            else:
                cursor = cursor + timedelta(days=7)

        logger.info(
            "Fetched %s scheduled games from %s to %s (cache hits=%s misses=%s)",
            len(games),
            start_date,
            end_date,
            api_cache.hits,
            api_cache.misses,
        )
        return games
    except Exception:
        logger.exception("Error fetching schedule")
        return []


def fetch_upcoming_games(days: int = 7) -> List[Dict[str, Any]]:
    """Fetch games from today through `days` days ahead."""
    today = date.today()
    end = today + timedelta(days=days)
    return fetch_schedule(start_date=today, end_date=end)


def fetch_standings(
    season: Optional[int] = None,
    as_of: Optional[date] = None,
) -> Dict[str, Any]:
    """Fetch league standings, cached by date (or season end)."""
    try:
        season_str = _format_season(season) if season is not None else None
        date_key = as_of.isoformat() if as_of else ("season-" + season_str if season_str else "now")

        def _call():
            if season_str:
                return nhl_client.standings.league_standings(season=season_str) or {}
            if as_of:
                return nhl_client.standings.league_standings(date=as_of.isoformat()) or {}
            return nhl_client.standings.league_standings() or {}

        before = api_cache.misses
        payload = api_cache.cached("standings", date_key, _call)
        if api_cache.misses > before:
            _pause_on_miss()
        return payload or {}
    except Exception:
        logger.exception("Error fetching standings")
        return {}


def _standings_list(standings_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not standings_data:
        return []
    if isinstance(standings_data, dict) and "standings" in standings_data:
        raw = standings_data["standings"]
        return raw if isinstance(raw, list) else []
    if isinstance(standings_data, list):
        return standings_data
    return []


def _record_from_standing(rec: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "wins": rec.get("wins"),
        "losses": rec.get("losses"),
        "otLosses": rec.get("otLosses"),
        "home_wins": rec.get("homeWins"),
        "home_losses": rec.get("homeLosses"),
        "away_wins": rec.get("roadWins"),
        "away_losses": rec.get("roadLosses"),
        "points": rec.get("points"),
        "gamesPlayed": rec.get("gamesPlayed"),
        "goalFor": rec.get("goalFor"),
        "goalAgainst": rec.get("goalAgainst"),
    }


def fetch_team_stats(
    team_id: Optional[int] = None,
    season: Optional[int] = None,
    team_abbr: Optional[str] = None,
    as_of: Optional[date] = None,
) -> Dict[str, Any]:
    """Season-to-date stats for one team from standings as of `as_of`."""
    del team_id
    try:
        if not team_abbr:
            logger.warning("fetch_team_stats requires team_abbr, returning empty stats")
            return {}
        abbr = team_abbr.upper()
        standings_data = fetch_standings(season=season, as_of=as_of)
        for rec in _standings_list(standings_data):
            if not isinstance(rec, dict):
                continue
            rec_abbrev = rec.get("teamAbbrev", "")
            if isinstance(rec_abbrev, dict):
                rec_abbrev = rec_abbrev.get("default", "")
            if str(rec_abbrev).upper() == abbr:
                return _record_from_standing(rec)
        return {}
    except Exception:
        logger.exception("Error fetching team stats for %s", team_abbr)
        return {}


def fetch_player_stats(
    player_id: int,
    season: Optional[int] = None,
    stats_type: str = "statsSingleSeason",
) -> Dict[str, Any]:
    """Fetch player stats using nhlpy (cached)."""
    try:
        if season is None:
            season = date.today().year if date.today().month >= 10 else date.today().year - 1
        season_str = _format_season(season)
        player_id_str = str(player_id)
        key = f"{player_id_str}_{season_str}_{stats_type}"

        def _call():
            if stats_type == "gameLog":
                return (
                    nhl_client.stats.player_game_log(
                        player_id=player_id_str, season_id=season_str, game_type=2
                    )
                    or {}
                )
            return nhl_client.stats.skater_stats_summary(
                start_season=season_str, end_season=season_str
            ) or {}

        before = api_cache.misses
        payload = api_cache.cached("player_stats", key, _call)
        if api_cache.misses > before:
            _pause_on_miss()
        return payload or {}
    except Exception:
        logger.exception("Error fetching player stats for %s", player_id)
        return {}


def fetch_roster_and_injuries(
    team_id: Optional[int] = None,
    season: Optional[int] = None,
    team_abbr: Optional[str] = None,
) -> Dict[str, Any]:
    """Fetch roster for a team abbreviation and season (cached)."""
    del team_id
    try:
        if not team_abbr:
            logger.warning("fetch_roster_and_injuries requires team_abbr")
            return {}
        if season is None:
            season = date.today().year if date.today().month >= 10 else date.today().year - 1
        season_str = _format_season(season)
        key = f"{team_abbr.upper()}_{season_str}"

        def _call():
            return nhl_client.teams.team_roster(team_abbr.upper(), season_str) or {}

        before = api_cache.misses
        payload = api_cache.cached("roster", key, _call)
        if api_cache.misses > before:
            _pause_on_miss()
        return payload or {}
    except Exception:
        logger.exception("Error fetching roster for %s", team_abbr)
        return {}


def fetch_game_boxscore(game_pk: int) -> Dict[str, Any]:
    """Fetch boxscore for a game id (cached)."""
    try:
        key = str(game_pk)
        before = api_cache.misses
        payload = api_cache.cached(
            "boxscore",
            key,
            lambda: nhl_client.game_center.boxscore(game_id=str(game_pk)) or {},
        )
        if api_cache.misses > before:
            _pause_on_miss()
        return payload or {}
    except Exception:
        logger.exception("Error fetching boxscore for %s", game_pk)
        return {}


def fetch_edge_metrics(player_id: int) -> Dict[str, Any]:
    """Fetch EDGE metrics (shot speed, skating speed) for players using nhlpy."""
    try:
        player_id_str = str(player_id)
        key = f"skater_{player_id_str}"

        def _call():
            metrics = nhl_client.edge.skater_shot_speed_detail(player_id=player_id_str)
            if not metrics:
                metrics = nhl_client.edge.skater_detail(player_id=player_id_str)
            return metrics or {}

        before = api_cache.misses
        payload = api_cache.cached("edge", key, _call)
        if api_cache.misses > before:
            _pause_on_miss()
        return payload or {}
    except Exception:
        logger.exception("Error fetching EDGE metrics for %s", player_id)
        return {}


def _goalies_from_roster(roster: Dict[str, Any]) -> List[Dict[str, Any]]:
    goalies = []
    if not isinstance(roster, dict):
        return goalies
    if "goalies" in roster and isinstance(roster["goalies"], list):
        for person in roster["goalies"]:
            if not isinstance(person, dict):
                continue
            goalies.append(
                {
                    "player_id": person.get("id"),
                    "name": (person.get("firstName") or {}).get("default", "")
                    + " "
                    + (person.get("lastName") or {}).get("default", "")
                    if isinstance(person.get("firstName"), dict)
                    else person.get("fullName"),
                    "number": person.get("sweaterNumber") or person.get("jerseyNumber"),
                }
            )
        return goalies
    for player_info in roster.get("roster", []):
        position = player_info.get("position", {})
        if position.get("code") == "G":
            person = player_info.get("person", {})
            goalies.append(
                {
                    "player_id": person.get("id"),
                    "name": person.get("fullName"),
                    "number": player_info.get("jerseyNumber"),
                }
            )
    return goalies


def fetch_goalie_stats(
    team_id: Optional[int] = None,
    team_abbr: Optional[str] = None,
    season: Optional[int] = None,
) -> Dict[str, Any]:
    """Fetch goalie list for a team from the cached roster (no per-goalie stats)."""
    try:
        roster = fetch_roster_and_injuries(team_id=team_id, season=season, team_abbr=team_abbr)
        if not roster:
            return {}
        goalies = _goalies_from_roster(roster)
        return {"team_id": team_id, "team_abbr": team_abbr, "goalies": goalies}
    except Exception:
        logger.exception("Error fetching goalie stats for %s", team_abbr or team_id)
        return {}


def fetch_goalie_edge_metrics(goalie_id: int) -> Dict[str, Any]:
    """Fetch advanced EDGE metrics for a goalie (cached)."""
    try:
        goalie_id_str = str(goalie_id)
        key = f"goalie_{goalie_id_str}"

        def _call():
            metrics = nhl_client.edge.goalie_save_percentage_detail(player_id=goalie_id_str)
            if not metrics:
                metrics = nhl_client.edge.goalie_detail(player_id=goalie_id_str)
            return metrics or {}

        before = api_cache.misses
        payload = api_cache.cached("edge", key, _call)
        if api_cache.misses > before:
            _pause_on_miss()
        return payload or {}
    except Exception:
        logger.exception("Error fetching goalie EDGE metrics for %s", goalie_id)
        return {}


def compute_goalie_rest(
    team_id: int,
    games: List[Dict[str, Any]],
    reference_date: Optional[date] = None,
) -> Dict[int, int]:
    """Days since the team's last game (placeholder keyed by 0, not per goalie)."""
    try:
        ref_date = reference_date or date.today()
        last = None
        for g in games:
            home = g.get("homeTeam", {})
            away = g.get("awayTeam", {})
            if home.get("id") != team_id and away.get("id") != team_id:
                continue
            gd = _game_date(g)
            if gd is None or gd >= ref_date:
                continue
            if last is None or gd > last:
                last = gd
        if last is None:
            return {}
        return {0: (ref_date - last).days}
    except Exception:
        logger.exception("Error computing goalie rest for team %s", team_id)
        return {}


def goalie_recent_form(
    goalie_id: int, games: List[Dict[str, Any]], window: int = 10
) -> Dict[str, Any]:
    """Stub — starter identification is not wired yet."""
    del goalie_id, games, window
    return {"wins": 0, "losses": 0, "games_played": 0}


def compute_rest_days(
    games: List[Dict[str, Any]], reference_date: Optional[date] = None
) -> Dict[int, int]:
    """Days since last game per team_id, relative to reference_date."""
    ref_date = reference_date or date.today()
    last_by_team: Dict[int, date] = {}
    for g in games:
        gd = _game_date(g)
        if gd is None or gd >= ref_date:
            continue
        for team in (g.get("homeTeam") or {}, g.get("awayTeam") or {}):
            tid = team.get("id")
            if tid is None:
                continue
            prev = last_by_team.get(int(tid))
            if prev is None or gd > prev:
                last_by_team[int(tid)] = gd
    return {tid: (ref_date - last).days for tid, last in last_by_team.items()}


def recent_form(
    team_id: int,
    games: List[Dict[str, Any]],
    window: int = FORM_WINDOW,
    reference_date: Optional[date] = None,
) -> Dict[str, Any]:
    """Wins/losses over the most recent `window` completed games before reference_date."""
    ref_date = reference_date or date.today()
    recent = []
    dated: List[Tuple[date, Dict[str, Any]]] = []
    for g in games:
        gd = _game_date(g)
        if gd is None or gd >= ref_date:
            continue
        dated.append((gd, g))
    dated.sort(key=lambda x: x[0], reverse=True)

    for _, g in dated:
        home_team = g.get("homeTeam") or {}
        away_team = g.get("awayTeam") or {}
        home = home_team.get("id")
        away = away_team.get("id")
        if team_id not in (home, away):
            continue
        home_score = home_team.get("score")
        away_score = away_team.get("score")
        if home_score is None or away_score is None:
            continue
        if team_id == home:
            recent.append("W" if home_score > away_score else "L" if home_score < away_score else "T")
        else:
            recent.append("W" if away_score > home_score else "L" if away_score < home_score else "T")
        if len(recent) >= window:
            break

    wins = recent.count("W")
    losses = recent.count("L")
    ties = recent.count("T")
    total = len(recent) if recent else 1
    return {"wins": wins, "losses": losses, "ties": ties, "pct": wins / total, "games": len(recent)}


def assemble_features_for_game(
    game: Dict[str, Any],
    season: Optional[int] = None,
    past_games: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Assemble features for one game using cached schedule + as-of standings.

    Roster/goalie/EDGE are skipped here so a full-season rebuild stays cheap.
    Pass `past_games` to avoid a per-game lookback fetch.
    """
    features: Dict[str, Any] = {}
    try:
        home_team = game.get("homeTeam") or {}
        away_team = game.get("awayTeam") or {}
        home_id = home_team.get("id")
        away_id = away_team.get("id")
        home_abbr = _team_abbr(home_team)
        away_abbr = _team_abbr(away_team)

        features["gamePk"] = _game_pk(game)
        game_date = _game_date(game) or date.today()
        features["gameDate"] = game_date.isoformat()
        features["home_abbrev"] = home_abbr
        features["away_abbrev"] = away_abbr

        as_of = game_date - timedelta(days=1)
        features["home_team_stats"] = fetch_team_stats(
            team_abbr=home_abbr, season=season, as_of=as_of
        )
        features["away_team_stats"] = fetch_team_stats(
            team_abbr=away_abbr, season=season, as_of=as_of
        )

        if past_games is None:
            lookback_start = game_date - timedelta(days=LOOKBACK_DAYS)
            past_games = fetch_schedule(start_date=lookback_start, end_date=game_date)

        rest_map = compute_rest_days(past_games, reference_date=game_date)
        features["home_rest_days"] = rest_map.get(home_id)
        features["away_rest_days"] = rest_map.get(away_id)
        features["home_recent_form"] = recent_form(
            home_id, past_games, window=FORM_WINDOW, reference_date=game_date
        )
        features["away_recent_form"] = recent_form(
            away_id, past_games, window=FORM_WINDOW, reference_date=game_date
        )
        return features
    except Exception:
        logger.exception("Error assembling features for game %s", game.get("id"))
        return features


def fetch_features_for_upcoming_games(
    days: int = 7, season: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Fetch upcoming games and assemble feature dicts for each game."""
    games = fetch_upcoming_games(days=days)
    if not games:
        return []
    dates = [_game_date(g) for g in games]
    dates = [d for d in dates if d]
    lookback_start = (min(dates) if dates else date.today()) - timedelta(days=LOOKBACK_DAYS)
    lookback_end = max(dates) if dates else date.today()
    past_games = fetch_schedule(start_date=lookback_start, end_date=lookback_end)
    features = []
    for g in games:
        try:
            features.append(
                assemble_features_for_game(g, season=season, past_games=past_games)
            )
        except Exception:
            logger.exception("Failed to process game")
    return features


if __name__ == "__main__":
    print("Run a quick smoke fetch for the next 3 days...")
    feats = fetch_features_for_upcoming_games(days=3)
    print(f"Assembled features for {len(feats)} games")
    print("cache", api_cache.stats())
