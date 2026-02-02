"""
Data Collection Module

This module handles fetching game data, player stats, and goalie info
from NHL and ESPN APIs.
"""

import requests
import pandas as pd
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NHL_API_BASE = "https://statsapi.web.nhl.com/api/v1"
FLYERS_ID = 25  # Philadelphia Flyers team ID


def get_flyers_games(season: int) -> List[Dict[str, Any]]:
    """
    Fetch all Flyers games for a given season.
    
    Args:
        season: Season year (e.g., 2025 for 2025-2026 season)
        
    Returns:
        List of game dictionaries with game info
    """
    try:
        url = f"{NHL_API_BASE}/teams/{FLYERS_ID}/schedule?season={season}{season+1}"
        response = requests.get(url)
        response.raise_for_status()
        games = response.json().get("games", [])
        logger.info(f"Retrieved {len(games)} Flyers games for season {season}")
        return games
    except Exception as e:
        logger.error(f"Error fetching Flyers games: {e}")
        return []


def get_game_details(game_id: int) -> Dict[str, Any]:
    """
    Fetch detailed stats for a specific game.
    
    Args:
        game_id: NHL game ID
        
    Returns:
        Dictionary with game details and box score
    """
    try:
        url = f"{NHL_API_BASE}/game/{game_id}/boxscore"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error fetching game details for {game_id}: {e}")
        return {}


def get_team_stats(team_id: int, season: int) -> Dict[str, Any]:
    """
    Fetch team stats for a season.
    
    Args:
        team_id: NHL team ID
        season: Season year
        
    Returns:
        Team statistics
    """
    try:
        url = f"{NHL_API_BASE}/teams/{team_id}?expand=team.stats&season={season}{season+1}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error fetching team stats: {e}")
        return {}


def get_player_stats(player_id: int) -> Dict[str, Any]:
    """
    Fetch individual player stats.
    
    Args:
        player_id: NHL player ID
        
    Returns:
        Player statistics
    """
    try:
        url = f"{NHL_API_BASE}/people/{player_id}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error fetching player stats: {e}")
        return {}


def save_games_to_csv(games: List[Dict], output_path: str = "data/raw/flyers_games.csv"):
    """Save game data to CSV."""
    df = pd.json_normalize(games)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(games)} games to {output_path}")


if __name__ == "__main__":
    # Example usage
    print("Data collection module loaded. Use functions to fetch NHL data.")
