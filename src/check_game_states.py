"""Check actual gameState values in recent games."""

from datetime import date, timedelta
from data_collection import fetch_schedule

# Look at games from the last week
game_states = {}
for days_back in range(0, 14):
    check_date = date(2025, 2, 3) - timedelta(days=days_back)
    games = fetch_schedule(start_date=check_date, end_date=check_date)
    
    for game in games:
        state = game.get('gameState')
        if state not in game_states:
            game_states[state] = 0
        game_states[state] += 1
        
        # Print first game of each state we haven't seen yet
        if game_states[state] == 1:
            print(f"\ngameState: '{state}'")
            if game.get('homeTeam'):
                print(f"  Home: {game['homeTeam'].get('commonName', {}).get('default')}")
            if game.get('awayTeam'):
                print(f"  Away: {game['awayTeam'].get('commonName', {}).get('default')}")
            print(f"  gameOutcome: {game.get('gameOutcome')}")

print("\n" + "=" * 60)
print("Summary of gameState values in last 2 weeks:")
for state, count in sorted(game_states.items()):
    print(f"  '{state}': {count} games")
