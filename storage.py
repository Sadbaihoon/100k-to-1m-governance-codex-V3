"""Local persistence layer for the personal version of the application.

Keeping data access here lets a future multi-user version replace SQLite with a
managed database without changing the investment-analysis interface.
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATABASE_PATH = Path(__file__).parent / "data" / "investment_journal.db"


def connection() -> sqlite3.Connection:
    DATABASE_PATH.parent.mkdir(exist_ok=True)
    database = sqlite3.connect(DATABASE_PATH)
    database.row_factory = sqlite3.Row
    return database


def initialize_database() -> None:
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
        existing_columns = {
            row["name"]
            for row in database.execute("PRAGMA table_info(decisions)").fetchall()
        }
        if "analysis_snapshot" not in existing_columns:
            database.execute("ALTER TABLE decisions ADD COLUMN analysis_snapshot TEXT")


def save_decision(**values: Any) -> int:
    columns = ", ".join(values)
    placeholders = ", ".join(f":{key}" for key in values)
    payload = {
        **values,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }
    columns = "created_at, " + columns
    placeholders = ":created_at, " + placeholders
    with connection() as database:
        cursor = database.execute(
            f"INSERT INTO decisions ({columns}) VALUES ({placeholders})",
            payload,
        )
    return int(cursor.lastrowid)


def get_decisions(limit: int = 250) -> list[dict[str, Any]]:
    with connection() as database:
        rows = database.execute(
            "SELECT * FROM decisions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(row) for row in rows]
