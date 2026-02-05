"""Check actual gameState values from historical games (Oct 2024 onwards)."""

from datetime import date, timedelta
from data_collection import fetch_schedule

# Start from beginning of season (Oct 2024)
check_date = date(2024, 10, 5)
games = fetch_schedule(start_date=check_date, end_date=check_date)

game_states = {}
for game in games:
    state = game.get('gameState')
    if state not in game_states:
        game_states[state] = 0
    game_states[state] += 1
    
    # Print first game of each state
    if game_states[state] == 1:
        print(f"\ngameState: '{state}'")
        if game.get('homeTeam'):
            print(f"  Home: {game['homeTeam'].get('commonName', {}).get('default')}")
        if game.get('awayTeam'):
            print(f"  Away: {game['awayTeam'].get('commonName', {}).get('default')}")
        print(f"  gameScheduleState: {game.get('gameScheduleState')}")
        print(f"  gameOutcome: {game.get('gameOutcome')}")

print("\n" + "=" * 60)
print(f"Summary of gameState values on {check_date}:")
for state, count in sorted(game_states.items()):
    print(f"  '{state}': {count} games")
