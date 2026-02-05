"""Inspect actual NHL API response structure for schedule and boxscore."""

import json
from datetime import date, timedelta
from data_collection import fetch_schedule, fetch_game_boxscore

# Fetch one day of games
start_date = date(2025, 2, 3)
games = fetch_schedule(start_date=start_date, end_date=start_date)

if games:
    game = games[0]
    print("=" * 80)
    print("SCHEDULE GAME STRUCTURE")
    print("=" * 80)
    print(f"Game ID: {game.get('id')}")
    print(f"\nTop-level keys: {list(game.keys())}")
    print(f"\nFull structure (first 50 lines):")
    structure = json.dumps(game, indent=2)
    lines = structure.split('\n')[:50]
    print('\n'.join(lines))
    
    # Now fetch boxscore for this game
    game_id = game.get('id') or game.get('gamePk')
    if game_id:
        print("\n" + "=" * 80)
        print("BOXSCORE STRUCTURE")
        print("=" * 80)
        boxscore = fetch_game_boxscore(game_id)
        if boxscore:
            print(f"Game ID: {boxscore.get('id')}")
            print(f"\nTop-level keys: {list(boxscore.keys())}")
            print(f"\nFull structure (first 50 lines):")
            structure = json.dumps(boxscore, indent=2)
            lines = structure.split('\n')[:50]
            print('\n'.join(lines))
            
            # Check for outcome-relevant fields
            print("\n" + "=" * 80)
            print("OUTCOME-RELEVANT FIELDS IN BOXSCORE")
            print("=" * 80)
            print(f"gameState: {boxscore.get('gameState')}")
            print(f"gameOutcome: {boxscore.get('gameOutcome')}")
            print(f"periodDescriptor: {boxscore.get('periodDescriptor')}")
            if boxscore.get('teams'):
                print(f"\nTeams structure:")
                for side in ['home', 'away']:
                    team = boxscore.get('teams', {}).get(side, {})
                    print(f"  {side}: {list(team.keys())}")
                    if 'score' in team:
                        print(f"    score: {team['score']}")
        else:
            print("Failed to fetch boxscore")
else:
    print("No games found for the date range")
