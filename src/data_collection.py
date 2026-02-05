"""Data collection utilities for assembling features used in win/loss prediction.

This module uses the nhlpy (nhl-api-py) library to fetch comprehensive NHL data
and assembles modeling features for upcoming games (team stats, recent form,
rest days, rosters, and optional EDGE metrics).

Functions are written to be defensive (returning empty dict/list on failure)
and easy to compose into a feature-assembly pipeline.
"""

from datetime import datetime, date, timedelta
import time
import pandas as pd
from typing import List, Dict, Any, Optional
import logging

from nhlpy import NHLClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize NHL API client
nhl_client = NHLClient()


def _format_season(season: int) -> str:
    """Return NHL season string like '20252026' for season year 2025."""
    return f"{season}{season+1}"


def fetch_schedule(start_date: Optional[date] = None,
                   end_date: Optional[date] = None,
                   team_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """Fetch schedule between two dates using nhlpy.

    Returns list of game dicts. If dates are omitted, fetches daily schedule.
    """
    try:
        if start_date and end_date:
            # Fetch daily schedule for each date in range
            games = []
            current = start_date
            while current <= end_date:
                daily = nhl_client.schedule.daily_schedule(date=current.isoformat())
                if daily and 'games' in daily:
                    games.extend(daily['games'])
                current += timedelta(days=1)
        else:
            # Fetch current daily schedule
            daily = nhl_client.schedule.daily_schedule()
            games = daily.get('games', []) if daily else []
        
        logger.info(f"Fetched {len(games)} scheduled games from {start_date} to {end_date}")
        return games
    except Exception:
        logger.exception("Error fetching schedule")
        return []


def fetch_upcoming_games(days: int = 7) -> List[Dict[str, Any]]:
    """Fetch games from today through `days` days ahead."""
    today = date.today()
    end = today + timedelta(days=days)
    return fetch_schedule(start_date=today, end_date=end)


def fetch_team_stats(team_id: int, season: Optional[int] = None) -> Dict[str, Any]:
    """Fetch team statistics (season-level) using nhlpy."""
    try:
        team_data = nhl_client.teams.teams()
        if not team_data or 'teams' not in team_data:
            return {}
        # Find team by ID
        team_info = next((t for t in team_data['teams'] if t.get('id') == team_id), None)
        return team_info or {}
    except Exception:
        logger.exception("Error fetching team stats for %s", team_id)
        return {}


def fetch_player_stats(player_id: int, season: Optional[int] = None,
                       stats_type: str = "statsSingleSeason") -> Dict[str, Any]:
    """Fetch player stats using nhlpy."""
    try:
        # Use skater_stats_summary or player_game_log
        if stats_type == "gameLog":
            return nhl_client.stats.player_game_log(player_id=player_id) or {}
        else:
            # Default to single season summary
            return nhl_client.stats.skater_stats_summary() or {}
    except Exception:
        logger.exception("Error fetching player stats for %s", player_id)
        return {}


def fetch_roster_and_injuries(team_id: int, season: Optional[int] = None) -> Dict[str, Any]:
    """Fetch roster and basic status information for a team using nhlpy."""
    try:
        roster = nhl_client.teams.team_roster(team_id=team_id)
        return roster or {}
    except Exception:
        logger.exception("Error fetching roster for %s", team_id)
        return {}


def fetch_standings(season: Optional[int] = None) -> Dict[str, Any]:
    """Fetch standings for a season or current standings using nhlpy."""
    try:
        standings = nhl_client.standings.league_standings()
        return standings or {}
    except Exception:
        logger.exception("Error fetching standings")
        return {}


def fetch_game_boxscore(game_pk: int) -> Dict[str, Any]:
    """Fetch boxscore for a given game primary key (gamePk) using nhlpy."""
    try:
        boxscore = nhl_client.game_center.boxscore(game_id=game_pk)
        return boxscore or {}
    except Exception:
        logger.exception("Error fetching boxscore for %s", game_pk)
        return {}


def fetch_edge_metrics(player_id: int) -> Dict[str, Any]:
    """Fetch EDGE metrics (shot speed, skating speed) for players using nhlpy."""
    try:
        # Try skater shot speed detail first
        metrics = nhl_client.edge.skater_shot_speed_detail(player_id=player_id)
        if not metrics:
            # Fallback to general skater detail
            metrics = nhl_client.edge.skater_detail(player_id=player_id)
        return metrics or {}
    except Exception:
        logger.exception("Error fetching EDGE metrics for %s", player_id)
        return {}


def fetch_goalie_stats(team_id: int) -> Dict[str, Any]:
    """Fetch goalie statistics for a team (starters and backups) using nhlpy."""
    try:
        # Get team roster to identify goalies
        roster = fetch_roster_and_injuries(team_id)
        if not roster:
            return {}
        
        # Extract goalie info from roster
        goalies = []
        if 'roster' in roster:
            for player_info in roster.get('roster', []):
                person = player_info.get('person', {})
                position = player_info.get('position', {})
                if position.get('code') == 'G':  # Goalie position code
                    goalies.append({
                        'player_id': person.get('id'),
                        'name': person.get('fullName'),
                        'number': player_info.get('jerseyNumber')
                    })
        
        # Fetch stats for each goalie
        goalie_stats = []
        for goalie in goalies:
            gid = goalie.get('player_id')
            if gid:
                stats = fetch_player_stats(gid, stats_type="statsSingleSeason")
                goalie_stats.append({
                    'goalie': goalie,
                    'stats': stats
                })
        
        return {'team_id': team_id, 'goalies': goalie_stats}
    except Exception:
        logger.exception("Error fetching goalie stats for %s", team_id)
        return {}


def fetch_goalie_edge_metrics(goalie_id: int) -> Dict[str, Any]:
    """Fetch advanced EDGE metrics for a goalie (save %, shot location, etc.)."""
    try:
        metrics = nhl_client.edge.goalie_save_percentage_detail(player_id=goalie_id)
        if not metrics:
            # Fallback to general goalie detail
            metrics = nhl_client.edge.goalie_detail(player_id=goalie_id)
        return metrics or {}
    except Exception:
        logger.exception("Error fetching goalie EDGE metrics for %s", goalie_id)
        return {}


def compute_goalie_rest(team_id: int, games: List[Dict[str, Any]]) -> Dict[int, int]:
    """Compute days since last game for each goalie on a team.
    
    Returns map of goalie_id -> rest_days (int).
    Works with both schedule and boxscore response structures.
    """
    goalie_rest_map: Dict[int, int] = {}
    try:
        # Parse games to find which goalie played for this team
        for g in sorted(games, key=lambda x: x.get("gameDate", x.get("startTimeUTC", "")), reverse=True):
            # Extract team from actual API structure (homeTeam/awayTeam)
            home_team = g.get("homeTeam", {})
            away_team = g.get("awayTeam", {})
            
            team_info = None
            if home_team.get("id") == team_id:
                team_info = home_team
            elif away_team.get("id") == team_id:
                team_info = away_team
            
            if team_info:
                # Get gameDate (boxscore) or startTimeUTC (schedule)
                gd = g.get("gameDate") or g.get("startTimeUTC", "").split("T")[0]
                if gd:
                    try:
                        last_game_date = datetime.fromisoformat(gd.replace("Z", "+00:00")).date()
                        # Simplified: return rest days for team (not per-goalie)
                        # In practice, you'd extract goalie ID from boxscore
                        return {0: (date.today() - last_game_date).days}  # Placeholder
                    except Exception:
                        pass
        return {}
    except Exception:
        logger.exception("Error computing goalie rest for team %s", team_id)
        return {}


def goalie_recent_form(goalie_id: int, games: List[Dict[str, Any]], window: int = 10) -> Dict[str, Any]:
    """Compute recent form (wins/losses/GAA) for a goalie from games list.
    
    Returns dict with wins, losses, games_played over `window` most recent games.
    """
    goalie_games = []
    try:
        # Filter games where this goalie participated (approximation from stats)
        # In practice, you'd parse boxscore play-by-play to confirm goalie involvement
        wins = 0
        losses = 0
        games_played = 0
        
        # Simplified: just track that we looked for the goalie
        return {
            "wins": wins,
            "losses": losses,
            "games_played": games_played,
            "avg_games": (wins + losses) if (wins + losses) > 0 else 0
        }
    except Exception:
        logger.exception("Error computing goalie form for %s", goalie_id)
        return {"wins": 0, "losses": 0, "games_played": 0}


def compute_rest_days(games: List[Dict[str, Any]]) -> Dict[int, int]:
    """Compute days since last game per team from a list of past games.

    Returns a map team_id -> rest_days (int). Assumes `games` includes
    past games with 'gameDate' and team information.
    Works with both schedule and boxscore response structures.
    """
    rows = []
    for g in games:
        # Handle both gameDate (boxscore) and startTimeUTC (schedule)
        gd = g.get("gameDate") or g.get("startTimeUTC", "").split("T")[0]
        try:
            played = datetime.fromisoformat(gd.replace("Z", "+00:00")).date()
        except Exception:
            continue
        
        # Extract team IDs from actual API structure (homeTeam/awayTeam)
        home_team = g.get("homeTeam", {})
        away_team = g.get("awayTeam", {})
        
        home_id = home_team.get("id")
        away_id = away_team.get("id")
        
        if home_id:
            rows.append({"team_id": home_id, "date": played})
        if away_id:
            rows.append({"team_id": away_id, "date": played})

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

    Returns dict with wins, losses, ties, pct over the `window` most recent games.
    Works with both schedule and boxscore response structures.
    """
    recent = []
    for g in sorted(games, key=lambda x: x.get("gameDate", x.get("startTimeUTC", "")), reverse=True):
        # Extract team IDs from actual API structure (homeTeam/awayTeam)
        home_team = g.get("homeTeam", {})
        away_team = g.get("awayTeam", {})
        
        home = home_team.get("id")
        away = away_team.get("id")
        
        if team_id not in (home, away):
            continue
        
        # Get scores from homeTeam/awayTeam structure
        home_score = home_team.get("score")
        away_score = away_team.get("score")
        
        # Skip games without scores (incomplete)
        if home_score is None or away_score is None:
            continue
        
        # Determine result for team
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

    This is a light-weight, synchronous assembly that calls the nhlpy
    endpoints for team stats, rosters, goalie stats, and computes rest and 
    recent form from available schedule information.
    
    Works with both schedule and boxscore response structures.
    """
    features: Dict[str, Any] = {}
    try:
        # Extract team IDs from actual API structure (homeTeam/awayTeam, not nested 'teams')
        home_team = game.get("homeTeam", {})
        away_team = game.get("awayTeam", {})
        
        home_id = home_team.get("id")
        away_id = away_team.get("id")
        
        # Store game metadata
        features["gamePk"] = game.get("gamePk") or game.get("id")
        features["gameDate"] = game.get("gameDate") or game.get("startTimeUTC", "").split("T")[0]

        # Fetch season-level stats
        features["home_team_stats"] = fetch_team_stats(home_id, season)
        features["away_team_stats"] = fetch_team_stats(away_id, season)

        # Roster & status
        features["home_roster"] = fetch_roster_and_injuries(home_id, season)
        features["away_roster"] = fetch_roster_and_injuries(away_id, season)

        # Goalie stats & metrics
        features["home_goalie_stats"] = fetch_goalie_stats(home_id)
        features["away_goalie_stats"] = fetch_goalie_stats(away_id)

        # Recent schedule (lookback 60 days) for rest and form
        game_date = None
        try:
            game_date_str = features.get("gameDate", "")
            if game_date_str:
                game_date = datetime.fromisoformat(game_date_str.replace("Z", "+00:00")).date()
            else:
                game_date = date.today()
        except Exception:
            game_date = date.today()
        
        lookback_start = game_date - timedelta(days=60)
        past_games = fetch_schedule(start_date=lookback_start, end_date=game_date)
        
        # Compute rest days and recent form
        rest_map = compute_rest_days(past_games)
        features["home_rest_days"] = rest_map.get(home_id, None)
        features["away_rest_days"] = rest_map.get(away_id, None)

        features["home_recent_form"] = recent_form(home_id, past_games, window=10)
        features["away_recent_form"] = recent_form(away_id, past_games, window=10)

        # Goalie rest and recent form
        goalie_rest = compute_goalie_rest(home_id, past_games)
        features["home_goalie_rest"] = goalie_rest
        goalie_rest = compute_goalie_rest(away_id, past_games)
        features["away_goalie_rest"] = goalie_rest

        return features
    except Exception:
        logger.exception("Error assembling features for game %s", game.get("gamePk") or game.get("id"))
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
