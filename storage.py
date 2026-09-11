"""SQLite persistence layer for the 100K → 1M Investment Governance app."""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DATABASE_PATH = Path(__file__).parent / "data" / "investment_journal.db"


def connection() -> sqlite3.Connection:
    """Open the SQLite database and ensure the data directory exists."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    database = sqlite3.connect(DATABASE_PATH)
    database.row_factory = sqlite3.Row
    return database


def initialize_database() -> None:
    """Create the decisions table if it does not already exist."""
    with connection() as database:
        database.execute(
            """
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                ticker TEXT NOT NULL,
                news_headline TEXT,
                news_source TEXT,
                has_event INTEGER NOT NULL,
                is_verified INTEGER NOT NULL,
                current_price REAL,
                entry_price REAL NOT NULL,
                stop_loss REAL NOT NULL,
                target_price REAL NOT NULL,
                risk_reward REAL NOT NULL,
                cooling_done INTEGER NOT NULL,
                thesis_clear INTEGER NOT NULL,
                no_emotion INTEGER NOT NULL,
                decision TEXT NOT NULL,
                notes TEXT
            )
            """
        )


def save_decision(**values: Any) -> int:
    """Save one governance decision and return its database ID."""
    payload = {
        **values,
        "created_at": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
    }

    columns = ", ".join(payload.keys())
    placeholders = ", ".join(f":{key}" for key in payload)

    with connection() as database:
        cursor = database.execute(
            f"INSERT INTO decisions ({columns}) VALUES ({placeholders})",
            payload,
        )

    return int(cursor.lastrowid)


def get_decisions(limit: int = 250) -> list[dict[str, Any]]:
    """Return the most recent governance decisions."""
    with connection() as database:
        rows = database.execute(
            "SELECT * FROM decisions ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()

    return [dict(row) for row in rows]
