"""Check how teams are structured in the actual API responses."""

from datetime import date
from data_collection import fetch_schedule, fetch_game_boxscore

# Get a FINAL game from Oct 5, 2024
check_date = date(2024, 10, 5)
games = fetch_schedule(start_date=check_date, end_date=check_date)

# Find a FINAL game
for game in games:
    if game.get('gameState') == 'FINAL':
        game_id = game.get('id')
        print("=" * 80)
        print("SCHEDULE RESPONSE - Team Structure")
        print("=" * 80)
        print(f"Keys at game level: {list(game.keys())}")
        print(f"'homeTeam' present? {' homeTeam' in game}")
        print(f"'awayTeam' present? {'awayTeam' in game}")
        print(f"'teams' present? {'teams' in game}")
        
        if 'homeTeam' in game:
            print(f"\nstructure: game['homeTeam']['id'] = {game['homeTeam']['id']}")
        
        # Fetch boxscore
        boxscore = fetch_game_boxscore(game_id)
        print("\n" + "=" * 80)
        print("BOXSCORE RESPONSE - Team Structure")
        print("=" * 80)
        print(f"Keys at game level: {list(boxscore.keys())}")
        print(f"'homeTeam' present? {'homeTeam' in boxscore}")
        print(f"'awayTeam' present? {'awayTeam' in boxscore}")
        print(f"'teams' present? {'teams' in boxscore}")
        
        if 'homeTeam' in boxscore:
            print(f"\nstructure: boxscore['homeTeam']['id'] = {boxscore['homeTeam']['id']}")
        
        break
