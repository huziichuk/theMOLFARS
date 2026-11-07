"""Seed PostgreSQL from data/builder_pack.xlsx.

Usage:
    python seed_db.py [path/to/builder_pack.xlsx]
"""
import os
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

ROOT = Path(__file__).parent
DEFAULT_XLSX = ROOT / "data" / "builder_pack.xlsx"
DB_URL = os.environ.get("DATABASE_URL", "postgresql+psycopg://molfars:molfars@localhost:5432/molfars")

SHEETS = [
    "Asset_Master",
    "Scene_Anchors",
    "Current_State",
    "Telemetry_Sample_3d",
    "Alarms_Events",
    "Maintenance_History",
    "Historical_Service_Cases",
    "Knowledge_Docs",
    "RAG_Corpus_Index",
    "Data_Quality",
    "Public_Research",
    "Builder_Questions",
    "API_Contract",
    "Change_Log",
]

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_asset_master_asset_id ON asset_master (asset_id)",
    "CREATE INDEX IF NOT EXISTS idx_current_state_asset_id ON current_state (asset_id)",
    "CREATE INDEX IF NOT EXISTS idx_telemetry_asset_ts ON telemetry_sample_3d (asset_id, timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_alarms_asset_ts ON alarms_events (asset_id, timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_maintenance_asset_opened ON maintenance_history (asset_id, opened_at)",
    "CREATE INDEX IF NOT EXISTS idx_cases_class_issue ON historical_service_cases (asset_class, historical_issue_code)",
    "CREATE INDEX IF NOT EXISTS idx_docs_id_scope ON knowledge_docs (doc_id, scope)",
    "CREATE INDEX IF NOT EXISTS idx_dq_asset_id ON data_quality (asset_id)",
]


def seed(xlsx_path: Path = DEFAULT_XLSX, db_url: str = DB_URL) -> dict[str, int]:
    engine = create_engine(db_url)
    data = pd.read_excel(xlsx_path, sheet_name=SHEETS)
    counts: dict[str, int] = {}
    with engine.begin() as conn:
        for sheet, df in data.items():
            table = sheet.lower()
            df.to_sql(table, conn, if_exists="replace", index=False, method="multi", chunksize=1000)
            counts[table] = len(df)
        for sql in INDEXES:
            conn.execute(text(sql))
    engine.dispose()
    return counts


if __name__ == "__main__":
    xlsx = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_XLSX
    counts = seed(xlsx)
    print(f"Seeded {len(counts)} tables into PostgreSQL ({sum(counts.values())} rows total):")
    for table, n in counts.items():
        print(f"  {table}: {n} rows")
