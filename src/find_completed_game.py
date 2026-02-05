"""Find a completed game to inspect."""

import json
from datetime import date, timedelta
from data_collection import fetch_schedule, fetch_game_boxscore

# Look back to find a completed game from last week
for days_back in range(1, 14):
    check_date = date(2025, 2, 3) - timedelta(days=days_back)
    games = fetch_schedule(start_date=check_date, end_date=check_date)
    
    for game in games:
        game_state = game.get('gameState')
        if game_state and 'FINAL' in game_state:
            print(f"Found completed game on {check_date}: {game.get('id')}")
            
            # Fetch boxscore
            boxscore = fetch_game_boxscore(game.get('id'))
            
            print("\n" + "=" * 80)
            print("SCHEDULE GAME (Completed)")
            print("=" * 80)
            print(f"gameState: {game.get('gameState')}")
            print(f"gameScheduleState: {game.get('gameScheduleState')}")
            print(f"gameOutcome: {game.get('gameOutcome')}")
            if game.get('homeTeam'):
                print(f"Home team: {game['homeTeam'].get('commonName', {}).get('default')}")
            if game.get('awayTeam'):
                print(f"Away team: {game['awayTeam'].get('commonName', {}).get('default')}")
            
            print("\n" + "=" * 80)
            print("BOXSCORE GAME (Completed)")
            print("=" * 80)
            if boxscore:
                print(f"gameState: {boxscore.get('gameState')}")
                print(f"gameScheduleState: {boxscore.get('gameScheduleState')}")
                print(f"gameOutcome: {boxscore.get('gameOutcome')}")
                
                # Check team scores
                home = boxscore.get('homeTeam', {})
                away = boxscore.get('awayTeam', {})
                print(f"\nHome team: {home.get('commonName', {}).get('default')}")
                print(f"  Keys: {list(home.keys())[:10]}")
                if 'score' in home:
                    print(f"  score: {home['score']}")
                print(f"Away team: {away.get('commonName', {}).get('default')}")
                print(f"  Keys: {list(away.keys())[:10]}")
                if 'score' in away:
                    print(f"  score: {away['score']}")
                
                # Print gameOutcome structure
                print(f"\ngameOutcome full structure:")
                print(json.dumps(boxscore.get('gameOutcome'), indent=2))
            
            exit()

print("No completed games found in the last 2 weeks")
