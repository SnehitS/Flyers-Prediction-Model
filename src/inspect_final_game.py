"""Inspect structure of a FINAL game to see how scores are stored."""

import json
from datetime import date
from data_collection import fetch_schedule, fetch_game_boxscore

# Get games from Oct 5, 2024 (opening day has FINAL games)
check_date = date(2024, 10, 5)
games = fetch_schedule(start_date=check_date, end_date=check_date)

# Find a FINAL game
final_game = None
for game in games:
    if game.get('gameState') == 'FINAL':
        final_game = game
        break

if final_game:
    print("=" * 80)
    print("SCHEDULE RESPONSE - FINAL GAME")
    print("=" * 80)
    game_id = final_game.get('id')
    print(f"Game ID: {game_id}")
    print(f"gameState: {final_game.get('gameState')}")
    print(f"gameOutcome: {final_game.get('gameOutcome')}")
    
    print(f"\nHome Team structure:")
    home = final_game.get('homeTeam', {})
    print(f"  Keys: {list(home.keys())}")
    print(f"  commonName: {home.get('commonName', {}).get('default')}")
    if 'score' in home:
        print(f"  score: {home['score']}")
    else:
        print(f"  (no 'score' key)")
    
    print(f"\nAway Team structure:")
    away = final_game.get('awayTeam', {})
    print(f"  Keys: {list(away.keys())}")
    print(f"  commonName: {away.get('commonName', {}).get('default')}")
    if 'score' in away:
        print(f"  score: {away['score']}")
    else:
        print(f"  (no 'score' key)")
    
    # Now fetch boxscore for the same game
    print("\n" + "=" * 80)
    print("BOXSCORE RESPONSE - FINAL GAME")
    print("=" * 80)
    boxscore = fetch_game_boxscore(game_id)
    if boxscore:
        print(f"Game ID: {boxscore.get('id')}")
        print(f"gameState: {boxscore.get('gameState')}")
        print(f"gameOutcome: {boxscore.get('gameOutcome')}")
        
        print(f"\nHome Team structure:")
        home_box = boxscore.get('homeTeam', {})
        print(f"  Keys: {list(home_box.keys())}")
        print(f"  commonName: {home_box.get('commonName', {}).get('default')}")
        if 'score' in home_box:
            print(f"  score: {home_box['score']}")
        else:
            print(f"  (no 'score' key)")
        
        print(f"\nAway Team structure:")
        away_box = boxscore.get('awayTeam', {})
        print(f"  Keys: {list(away_box.keys())}")
        print(f"  commonName: {away_box.get('commonName', {}).get('default')}")
        if 'score' in away_box:
            print(f"  score: {away_box['score']}")
        else:
            print(f"  (no 'score' key)")
        
        # Check for summary stats
        print(f"\nLooking for scores in other places...")
        print(f"  keys in boxscore: {list(boxscore.keys())}")
        if 'summary' in boxscore:
            print(f"  summary: {boxscore['summary']}")
else:
    print("No FINAL games found")
