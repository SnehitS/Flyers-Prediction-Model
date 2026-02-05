# Summary of Structure Fixes for Dataset Builder

## Issues Found & Fixed

### 1. **extract_game_outcome() in build_dataset.py**
   **Problem**: Checking for `gameState == 'FINAL'` but looking for scores in wrong location
   - Was checking `game['teams']['home']['score']` (doesn't exist)
   - Actual structure: `game['homeTeam']['score']`
   
   **Fix**: Updated to extract scores from `game['homeTeam']['score']` and `game['awayTeam']['score']`
   - Both schedule and boxscore responses have this structure
   - Now correctly identifies FINAL games and extracts outcomes

### 2. **assemble_features_for_game() in data_collection.py**
   **Problem**: Looking for teams in wrong location
   - Was trying: `game['teams']['home']['team']['id']` (doesn't exist)
   - Actual structure: `game['homeTeam']['id']`
   
   **Fix**: Updated to extract team IDs from `game['homeTeam']['id']` and `game['awayTeam']['id']`
   - Handles both gameDate (boxscore) and startTimeUTC (schedule) fields

### 3. **compute_rest_days() in data_collection.py**
   **Problem**: Same team location issue
   - Was iterating over `game['teams'][side]['team']`
   - Actual: `game['homeTeam']` and `game['awayTeam']` at top level
   
   **Fix**: Extracts team IDs directly from homeTeam/awayTeam dicts
   - Handles both gameDate and startTimeUTC fields

### 4. **recent_form() in data_collection.py**
   **Problem**: Same team and score location issues
   - Was looking in `game['teams'][side]['team']`
   - Was looking for scores in nested teams structure
   
   **Fix**: Extracts teams from homeTeam/awayTeam and scores from `homeTeam['score']`

### 5. **compute_goalie_rest() in data_collection.py**
   **Problem**: Same team structure issue
   - Was iterating over `game['teams'][side]['team']`
   
   **Fix**: Compares against homeTeam/awayTeam IDs directly

## API Response Structure (Actual)

### Schedule Response Format:
```
game['id']              - Game ID
game['gameState']       - 'FINAL', 'OFF', etc.
game['homeTeam']        - { id, commonName, score, ... }
game['awayTeam']        - { id, commonName, score, ... }
game['gameDate']        - Date string (optional in schedule)
game['startTimeUTC']    - DateTime string
```

### Boxscore Response Format:
```
boxscore['id']          - Game ID
boxscore['gameState']   - 'FINAL', 'OFF', etc.
boxscore['homeTeam']    - { id, commonName, score, ... }
boxscore['awayTeam']    - { id, commonName, score, ... }
boxscore['gameDate']    - Date string
boxscore['startTimeUTC'] - DateTime string
```

## Test Results

✓ Games identified as FINAL correctly
✓ Scores extracted: 2-0 (Capitals vs Bruins) 
✓ Outcomes calculated: home_win=1, away_win=0
✓ Feature flattening produces ~23 fields
✓ All functions handle both schedule and boxscore structures

## Ready to Run Full Build

All structural issues have been fixed. The dataset builder should now:
1. Correctly identify completed games (gameState == 'FINAL')
2. Extract outcomes from both schedule and boxscore responses
3. Assemble features for each game
4. Flatten features into a single CSV row
5. Save labeled dataset to data/processed/features.csv
