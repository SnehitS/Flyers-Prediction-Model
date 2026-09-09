"""Build a labeled dataset of historical regular-season games.

Schedule is fetched once (cached weekly API calls). Team records, rest, and
form are computed as-of each game from that list — no per-game boxscore or
standings round-trip.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta
from typing import Any, Dict, List, Optional
import logging
import os

import pandas as pd

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_collection import (
    FORM_WINDOW,
    LOOKBACK_DAYS,
    _game_date,
    _game_pk,
    _is_final,
    _is_regular_season,
    _team_abbr,
    fetch_schedule,
)
import api_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 2024-25 and 2025-26 regular seasons (from NHL standings-season manifest)
SEASON_WINDOWS = (
    (date(2024, 10, 4), date(2025, 4, 17)),
    (date(2025, 10, 7), date(2026, 4, 17)),
)


def fetch_historical_games(start_date: date, end_date: date) -> List[Dict[str, Any]]:
    """Fetch all games in a date range (cached weekly schedule)."""
    logger.info("Fetching games from %s to %s...", start_date, end_date)
    games = fetch_schedule(start_date=start_date, end_date=end_date)
    logger.info("Retrieved %s games", len(games))
    return games


def extract_game_outcome(game: Dict[str, Any]) -> Dict[str, Any]:
    """Win/loss from a completed schedule game (OFF or FINAL)."""
    outcome = {
        "home_win": None,
        "away_win": None,
        "home_score": None,
        "away_score": None,
        "result_type": None,
        "period_type": None,
    }
    try:
        if not _is_final(game):
            return outcome
        home_score = (game.get("homeTeam") or {}).get("score")
        away_score = (game.get("awayTeam") or {}).get("score")
        if home_score is None or away_score is None:
            return outcome
        outcome["home_score"] = int(home_score)
        outcome["away_score"] = int(away_score)
        period = (game.get("gameOutcome") or {}).get("lastPeriodType") or (
            (game.get("periodDescriptor") or {}).get("periodType")
        )
        outcome["period_type"] = period
        if home_score > away_score:
            outcome["home_win"] = 1
            outcome["away_win"] = 0
            outcome["result_type"] = "W" if period == "REG" else period
        elif home_score < away_score:
            outcome["home_win"] = 0
            outcome["away_win"] = 1
            outcome["result_type"] = "L" if period == "REG" else period
        else:
            outcome["home_win"] = 0
            outcome["away_win"] = 0
            outcome["result_type"] = "TIE"
    except Exception as e:
        logger.warning("Failed to extract outcome for game %s: %s", _game_pk(game), e)
    return outcome


def flatten_features(features: Dict[str, Any], outcome: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten nested assemble_features_for_game output (upcoming-game path)."""
    flat = {
        "gamePk": features.get("gamePk"),
        "gameDate": features.get("gameDate"),
        "home_abbrev": features.get("home_abbrev"),
        "away_abbrev": features.get("away_abbrev"),
    }
    flat.update(outcome)

    home_stats = features.get("home_team_stats") or {}
    away_stats = features.get("away_team_stats") or {}
    for prefix, stats in (("home", home_stats), ("away", away_stats)):
        flat[f"{prefix}_wins"] = stats.get("wins")
        flat[f"{prefix}_losses"] = stats.get("losses")
        flat[f"{prefix}_ot_losses"] = stats.get("otLosses")
        flat[f"{prefix}_points"] = stats.get("points")
        flat[f"{prefix}_gp"] = stats.get("gamesPlayed")
        flat[f"{prefix}_gf"] = stats.get("goalFor")
        flat[f"{prefix}_ga"] = stats.get("goalAgainst")

    home_form = features.get("home_recent_form") or {}
    away_form = features.get("away_recent_form") or {}
    if isinstance(home_form, dict):
        flat["home_form_wins"] = home_form.get("wins", 0)
        flat["home_form_losses"] = home_form.get("losses", 0)
        flat["home_form_pct"] = home_form.get("pct")
    if isinstance(away_form, dict):
        flat["away_form_wins"] = away_form.get("wins", 0)
        flat["away_form_losses"] = away_form.get("losses", 0)
        flat["away_form_pct"] = away_form.get("pct")

    flat["home_rest_days"] = features.get("home_rest_days")
    flat["away_rest_days"] = features.get("away_rest_days")
    return flat


def _empty_record() -> Dict[str, int]:
    return {"wins": 0, "losses": 0, "otl": 0, "points": 0, "gp": 0, "gf": 0, "ga": 0}


def _form_stats(results: List[str], window: int = FORM_WINDOW) -> Dict[str, Any]:
    recent = results[-window:]
    wins = recent.count("W")
    losses = recent.count("L")
    total = len(recent)
    return {
        "wins": wins,
        "losses": losses,
        "pct": (wins / total) if total else 0.0,
        "games": total,
    }


def _snapshot_team(
    prefix: str,
    team_id: Optional[int],
    abbrev: Optional[str],
    records: Dict[int, Dict[str, int]],
    form: Dict[int, List[str]],
    last_played: Dict[int, date],
    game_date: date,
) -> Dict[str, Any]:
    tid = int(team_id) if team_id is not None else None
    rec = records.get(tid, _empty_record()) if tid is not None else _empty_record()
    fs = _form_stats(form.get(tid, []) if tid is not None else [])
    last = last_played.get(tid) if tid is not None else None
    rest = (game_date - last).days if last is not None else None
    return {
        f"{prefix}_abbrev": abbrev,
        f"{prefix}_id": tid,
        f"{prefix}_wins": rec["wins"],
        f"{prefix}_losses": rec["losses"],
        f"{prefix}_ot_losses": rec["otl"],
        f"{prefix}_points": rec["points"],
        f"{prefix}_gp": rec["gp"],
        f"{prefix}_gf": rec["gf"],
        f"{prefix}_ga": rec["ga"],
        f"{prefix}_form_wins": fs["wins"],
        f"{prefix}_form_losses": fs["losses"],
        f"{prefix}_form_pct": fs["pct"],
        f"{prefix}_rest_days": rest,
    }


def _apply_game(
    records: Dict[int, Dict[str, int]],
    form: Dict[int, List[str]],
    last_played: Dict[int, date],
    game: Dict[str, Any],
    game_date: date,
    outcome: Dict[str, Any],
) -> None:
    home = game.get("homeTeam") or {}
    away = game.get("awayTeam") or {}
    hid = home.get("id")
    aid = away.get("id")
    hs = outcome["home_score"]
    aws = outcome["away_score"]
    period = outcome.get("period_type")
    ot = period in {"OT", "SO"}

    def _touch(tid: Optional[int], gf: int, ga: int, won: bool) -> None:
        if tid is None:
            return
        tid = int(tid)
        rec = records[tid]
        rec["gp"] += 1
        rec["gf"] += gf
        rec["ga"] += ga
        last_played[tid] = game_date
        if won:
            rec["wins"] += 1
            rec["points"] += 2
            form[tid].append("W")
        elif ot:
            rec["otl"] += 1
            rec["points"] += 1
            form[tid].append("L")
        else:
            rec["losses"] += 1
            form[tid].append("L")

    if hid is not None:
        last_played[int(hid)] = game_date
    if aid is not None:
        last_played[int(aid)] = game_date

    if not _is_regular_season(game):
        return

    home_won = hs > aws
    _touch(hid, hs, aws, home_won)
    _touch(aid, aws, hs, not home_won and hs != aws)


def labeled_rows(
    games: List[Dict[str, Any]],
    start_date: date,
    end_date: date,
) -> List[Dict[str, Any]]:
    """Walk games in order; emit a row for each regular-season game in range."""
    ordered = sorted(
        games,
        key=lambda g: (_game_date(g) or date.min, _game_pk(g) or 0),
    )
    records: Dict[int, Dict[str, int]] = defaultdict(_empty_record)
    form: Dict[int, List[str]] = defaultdict(list)
    last_played: Dict[int, date] = {}
    current_season = None
    rows: List[Dict[str, Any]] = []

    for i, game in enumerate(ordered, start=1):
        gd = _game_date(game)
        if gd is None:
            continue
        outcome = extract_game_outcome(game)
        if outcome.get("home_win") is None:
            continue

        season = game.get("season")
        if season != current_season:
            records = defaultdict(_empty_record)
            form = defaultdict(list)
            current_season = season

        home = game.get("homeTeam") or {}
        away = game.get("awayTeam") or {}
        in_window = start_date <= gd <= end_date and _is_regular_season(game)
        if in_window:
            row = {
                "gamePk": _game_pk(game),
                "gameDate": gd.isoformat(),
                "season": season,
            }
            row.update(
                _snapshot_team(
                    "home", home.get("id"), _team_abbr(home),
                    records, form, last_played, gd,
                )
            )
            row.update(
                _snapshot_team(
                    "away", away.get("id"), _team_abbr(away),
                    records, form, last_played, gd,
                )
            )
            row.update(outcome)
            rows.append(row)

        _apply_game(records, form, last_played, game, gd, outcome)

        if i % 500 == 0:
            logger.info("Processed %s / %s games (%s labeled)", i, len(ordered), len(rows))

    return rows


def build_dataset(start_date: date, end_date: date, output_path: str) -> pd.DataFrame:
    """Fetch the date range (plus lookback), label regular-season games, save CSV."""
    api_cache.reset_stats()
    lookback_start = start_date - timedelta(days=LOOKBACK_DAYS)
    games = fetch_historical_games(lookback_start, end_date)
    if not games:
        logger.error("No games fetched.")
        return pd.DataFrame()

    rows = labeled_rows(games, start_date, end_date)
    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("No valid rows were assembled into dataset.")
        return df

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)

    cache_stats = api_cache.stats()
    logger.info("Saved dataset to %s", output_path)
    logger.info("Dataset shape: %s", df.shape)
    logger.info("Home win rate: %.3f", df["home_win"].mean())
    logger.info("Cache hits=%s misses=%s", cache_stats["hits"], cache_stats["misses"])
    logger.info("Missing values:\n%s", df.isnull().sum())
    return df


if __name__ == "__main__":
    start = SEASON_WINDOWS[0][0]
    end = SEASON_WINDOWS[-1][1]
    output = "data/processed/features.csv"
    logger.info("Building regular-season dataset from %s to %s", start, end)
    df = build_dataset(start, end, output)
    if not df.empty:
        print(f"\nDataset built: {df.shape[0]} games -> {output}")
        print(f"  Date range: {df['gameDate'].min()} to {df['gameDate'].max()}")
        print(f"  Home win rate: {df['home_win'].mean():.1%}")
        print(f"  Rest days median: {df['home_rest_days'].median()}")
        print(f"  Cache: {api_cache.stats()}")
    else:
        print("Failed to build dataset")
