"""Build a labeled dataset of historical games for training.

This script fetches games from a specified date range, assembles features
for each game using data_collection.py, and adds outcome labels (home_win, away_win).
Output is saved to data/processed/features.csv for model training.
"""

import pandas as pd
from datetime import datetime, date, timedelta
from typing import List, Dict, Any
import logging
import json
import os

from data_collection import (
    fetch_schedule, 
    assemble_features_for_game, 
    fetch_game_boxscore
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_historical_games(start_date: date, end_date: date) -> List[Dict[str, Any]]:
    """Fetch all games in a date range."""
    logger.info(f"Fetching games from {start_date} to {end_date}...")
    games = fetch_schedule(start_date=start_date, end_date=end_date)
    logger.info(f"Retrieved {len(games)} games")
    return games


def extract_game_outcome(game: Dict[str, Any]) -> Dict[str, Any]:
    """Extract win/loss outcome from a completed game (boxscore or schedule).
    
    Returns dict with:
      - home_win: 1 if home won, 0 otherwise
      - away_win: 1 if away won, 0 otherwise
      - home_score, away_score: Final scores
      - result_type: 'W', 'L', 'OT' (for home team)
    """
    outcome = {
        "home_win": None,
        "away_win": None,
        "home_score": None,
        "away_score": None,
        "result_type": None
    }
    
    try:
        # Check if game is complete - both schedule and boxscore use 'gameState' = 'FINAL'
        game_state = game.get("gameState", "")
        
        # Only process if marked as final
        if game_state != "FINAL":
            return outcome
        
        # Get scores from homeTeam/awayTeam (works for both schedule and boxscore)
        home_score = game.get("homeTeam", {}).get("score")
        away_score = game.get("awayTeam", {}).get("score")
        
        if home_score is not None and away_score is not None:
            outcome["home_score"] = int(home_score)
            outcome["away_score"] = int(away_score)
            
            if home_score > away_score:
                outcome["home_win"] = 1
                outcome["away_win"] = 0
                outcome["result_type"] = "W"
            elif home_score < away_score:
                outcome["home_win"] = 0
                outcome["away_win"] = 1
                outcome["result_type"] = "L"
            else:
                # Tie - both get 0
                outcome["home_win"] = 0
                outcome["away_win"] = 0
                outcome["result_type"] = "TIE"
    except Exception as e:
        logger.warning(f"Failed to extract outcome for game {game.get('gamePk') or game.get('id')}: {e}")
    
    return outcome


def flatten_features(features: Dict[str, Any], outcome: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten nested feature dict into a single-level dict for CSV output.
    
    Keeps numeric features, flattens nested dicts with underscores,
    and includes outcome labels.
    """
    flat = {}
    
    # Metadata
    flat["gamePk"] = features.get("gamePk")
    flat["gameDate"] = features.get("gameDate")
    
    # Outcome labels
    flat.update(outcome)
    
    # Flatten team stats
    home_stats = features.get("home_team_stats", {})
    away_stats = features.get("away_team_stats", {})
    
    # Extract simple numeric stats if available
    if isinstance(home_stats, dict):
        flat["home_wins"] = home_stats.get("wins")
        flat["home_losses"] = home_stats.get("losses")
    if isinstance(away_stats, dict):
        flat["away_wins"] = away_stats.get("wins")
        flat["away_losses"] = away_stats.get("losses")
    
    # Recent form
    home_form = features.get("home_recent_form", {})
    away_form = features.get("away_recent_form", {})
    
    if isinstance(home_form, dict):
        flat["home_form_wins"] = home_form.get("wins", 0)
        flat["home_form_losses"] = home_form.get("losses", 0)
        flat["home_form_ties"] = home_form.get("ties", 0)
        flat["home_form_pct"] = home_form.get("pct")
    if isinstance(away_form, dict):
        flat["away_form_wins"] = away_form.get("wins", 0)
        flat["away_form_losses"] = away_form.get("losses", 0)
        flat["away_form_ties"] = away_form.get("ties", 0)
        flat["away_form_pct"] = away_form.get("pct")
    
    # Rest days
    flat["home_rest_days"] = features.get("home_rest_days")
    flat["away_rest_days"] = features.get("away_rest_days")
    
    # Goalie count (simple proxy)
    home_goalies = features.get("home_goalie_stats", {}).get("goalies", [])
    away_goalies = features.get("away_goalie_stats", {}).get("goalies", [])
    flat["home_goalie_count"] = len(home_goalies) if isinstance(home_goalies, list) else 0
    flat["away_goalie_count"] = len(away_goalies) if isinstance(away_goalies, list) else 0
    
    return flat


def build_dataset(start_date: date, end_date: date, output_path: str) -> pd.DataFrame:
    """Fetch games in date range, assemble features, extract outcomes, and save CSV.
    
    Args:
        start_date: First date to fetch (inclusive)
        end_date: Last date to fetch (inclusive)
        output_path: Path to save output CSV
        
    Returns:
        DataFrame with assembled features and labels
    """
    games = fetch_historical_games(start_date, end_date)
    if not games:
        logger.error("No games fetched.")
        return pd.DataFrame()
    
    # Debug: print first game structure
    if games:
        logger.info(f"First game keys: {games[0].keys() if isinstance(games[0], dict) else 'not a dict'}")
    
    rows = []
    completed_count = 0
    for i, game in enumerate(games):
        try:
            # Resolve a game primary key for fetching boxscore (works across schedule shapes)
            game_pk = None
            if isinstance(game, dict):
                game_pk = game.get("gamePk") or game.get("id") or game.get("gameId")
            
            # Prefer boxscore for outcome and detailed structure (more consistent)
            boxscore = None
            if game_pk:
                boxscore = fetch_game_boxscore(game_pk)

            # If boxscore available, use it; otherwise fall back to schedule entry
            source_game = boxscore or game

            # Extract outcome first (only keep completed games)
            outcome = extract_game_outcome(source_game)
            if outcome.get("home_win") is None:
                continue  # Skip incomplete games

            completed_count += 1

            # Assemble features using the boxscore (or schedule fallback)
            features = assemble_features_for_game(source_game)

            # Flatten and combine
            row = flatten_features(features, outcome)
            rows.append(row)

            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1} / {len(games)} games ({completed_count} completed)")
        except Exception as e:
            logger.warning(f"Error processing game {game.get('gamePk') if isinstance(game, dict) else 'unknown'}: {e}")
            continue

    logger.info(f"Found {completed_count} completed games out of {len(games)} total")
    
    # Convert to DataFrame
    df = pd.DataFrame(rows)
    
    if df.empty:
        logger.warning("No valid rows were assembled into dataset.")
        return df
    
    logger.info(f"Assembled {len(df)} labeled games")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    logger.info(f"Saved dataset to {output_path}")
    
    # Log summary
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Home win rate: {df['home_win'].mean():.3f}")
    logger.info(f"Missing values:\n{df.isnull().sum()}")
    
    return df


if __name__ == "__main__":
    # Build dataset for 2024-2025 season (historical games)
    # Adjust dates as needed - starting from earlier in season for more data
    start = date(2024, 10, 1)  # Start of 2024-25 season
    end = date(2026, 2, 4)      # Up to today
    
    output = "data/processed/features.csv"
    
    logger.info(f"Building dataset from {start} to {end}")
    df = build_dataset(start, end, output)
    
    if not df.empty:
        print(f"\n✓ Dataset built successfully!")
        print(f"  Shape: {df.shape}")
        print(f"  Output: {output}")
        print(f"  Home win rate: {df['home_win'].mean():.1%}")
    else:
        print("✗ Failed to build dataset")
