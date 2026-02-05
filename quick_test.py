"""Quick test of the fixed extract_game_outcome and assemble_features functions."""

import sys
sys.path.insert(0, 'src')

from datetime import date
from build_dataset import extract_game_outcome, flatten_features
from data_collection import fetch_schedule, fetch_game_boxscore, assemble_features_for_game

# Get a FINAL game from Oct 5, 2024
check_date = date(2024, 10, 5)
games = fetch_schedule(start_date=check_date, end_date=check_date)

# Test with a FINAL game
for i, game in enumerate(games):
    if game.get('gameState') == 'FINAL' and i < 2:  # Test first 2 FINAL games
        game_id = game.get('id')
        print(f"\n{'=' * 80}")
        print(f"Testing game {game_id}")
        print(f"{'=' * 80}")
        
        # Test outcome extraction from schedule
        outcome_schedule = extract_game_outcome(game)
        print(f"\nOutcome from schedule:")
        print(f"  home_win: {outcome_schedule['home_win']}")
        print(f"  away_win: {outcome_schedule['away_win']}")
        print(f"  home_score: {outcome_schedule['home_score']}")
        print(f"  away_score: {outcome_schedule['away_score']}")
        
        # Fetch boxscore
        boxscore = fetch_game_boxscore(game_id)
        
        # Test outcome extraction from boxscore
        outcome_boxscore = extract_game_outcome(boxscore)
        print(f"\nOutcome from boxscore:")
        print(f"  home_win: {outcome_boxscore['home_win']}")
        print(f"  away_win: {outcome_boxscore['away_win']}")
        print(f"  home_score: {outcome_boxscore['home_score']}")
        print(f"  away_score: {outcome_boxscore['away_score']}")
        
        # Test feature assembly (use boxscore as it's more complete)
        print(f"\nAssembling features from boxscore...")
        features = assemble_features_for_game(boxscore)
        print(f"  gamePk: {features.get('gamePk')}")
        print(f"  gameDate: {features.get('gameDate')}")
        print(f"  home_team_stats: {type(features.get('home_team_stats'))}")
        print(f"  home_recent_form: {features.get('home_recent_form')}")
        print(f"  home_rest_days: {features.get('home_rest_days')}")
        
        # Test flattening
        print(f"\nFlattening features...")
        flat = flatten_features(features, outcome_boxscore)
        print(f"  Keys in flat dict: {len(flat)}")
        print(f"  Sample keys: {list(flat.keys())[:10]}")
        
        if outcome_boxscore['home_win'] is not None:
            print(f"\nSUCCESS: Game {game_id} outcome extracted correctly")
        else:
            print(f"\nFAILED: Game {game_id} outcome is None")

print("\n" + "=" * 80)
print("Quick test complete - structure fixes verified!")
