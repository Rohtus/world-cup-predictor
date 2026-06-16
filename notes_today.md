# Session Notes - 2026-06-16

## Setup
- Git repository configured
- Python venv configured
- Jupyter Notebook working

## Datasets downloaded

### World Cup DB (Fjelstul)
Location:
data/raw/worldcup_db/

Relevant tables:
- matches.csv
- players.csv
- squads.csv
- player_appearances.csv

### StatsBomb Open Data
Location:
data/raw/statsbomb/

Important discoveries:
- FIFA World Cup competition_id = 43
- 2022 season_id = 106
- 2018 season_id = 3

Example match:
- match_id = 3857276

## StatsBomb structure

competitions
→ matches
→ lineups
→ events
→ three-sixty

## Events discovered

- Pass
- Shot
- Pressure
- Carry
- Dribble
- Duel
- Interception
- Ball Recovery
- Block
- Clearance
- Tactical Shift

## Project direction

player_vector
→ team_vector
→ match_vector
→ prediction

Focus:
Represent teams through player characteristics rather than country names.