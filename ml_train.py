"""
ml_train.py — Inkrementell daglig treningspipeline

Brukes slik:
    # Kjøres hver morgen etter at dagens filer er klare:
    python ml_train.py \
        --genus  data/Genus_170926.xlsx \
        --odin   data/Odin_170926.xlsx \
        --af     data/Allfunds_170926.txt \
        --breaks data/Breaks_170926.xlsx

Hva som skjer:
    1. Leser dagens filer via ml_features.py og lagrer feature-rad i feature_store.db
    2. Henter all historikk fra feature_store.db
    3. Walk-forward split: trening = alt t.o.m. i går, validering = i dag
    4. Trener XGBoost-regressor (Lag 1) og Random Forest-klassifikator (Lag 3)
    5. Evaluerer og lagrer modeller med datostempel i models/
    6. Logger metrics til logs/metrics.csv
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
import warnings
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, f1_score,
                              mean_absolute_error)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=UserWarning)

# ml_features.py må ligge i samme mappe
import ml_features as mf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Konfigurasjon
# ---------------------------------------------------------------------------

MODELS_DIR      = Path("models")
LOGS_DIR        = Path("logs")
FEATURE_DB      = Path("feature_store.db")
METRICS_CSV     = LOGS_DIR / "metrics.csv"

# Minimum antall treningsdager før modell trenes
MIN_TRAIN_DAYS  = 30

# Minimum antall merkede brudd for å trene klassifikator
MIN_BREAK_ROWS  = 20

# MAE-terskel: godkjent hvis < 0,5 % av gjennomsnittlig beløp
MAE_PCT_THRESH  = 0.005

# Features som brukes av aggregeringsmodellen (Lag 1)
AGG_FEATURES = [
    "net_flow_nok",
    "gross_buy_nok",
    "gross_sell_nok",
    "order_count",
    "ask_flow_nok",
    "settle_lag",
    "weekday",
    "is_month_end",
    "is_month_start",
    "net_flow_lag1",
    "net_flow_lag2",
    "net_flow_lag3",
    "net_flow_lag4",
    "net_flow_lag5",
    "net_flow_roll_mean_30",
    "net_flow_roll_std_30",
    "settle_date_diff",
]

# Features som brukes av årsaksklassifikatoren (Lag 3)
CLF_FEATURES = [
    "net_flow_nok",
    "gross_buy_nok",
    "gross_sell_nok",
    "order_count",
    "ask_flow_nok",
    "settle_lag",
    "weekday",
    "is_month_end",
    "net_flow_lag1",
    "net_flow_lag2",
    "net_flow_lag3",
    "settle_date_diff",
]


# ---------------------------------------------------------------------------
# Feature store (SQLite)
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS features (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_date      TEXT    NOT NULL,
    isin            TEXT    NOT NULL,
    institution     TEXT    NOT NULL,
    net_flow_nok    REAL,
    gross_buy_nok   REAL,
    gross_sell_nok  REAL,
    order_count     REAL,
    ask_flow_nok    REAL,
    settle_lag      REAL,
    weekday         INTEGER,
    is_month_end    INTEGER,
    is_month_start  INTEGER,
    ext_amount      REAL,
    settle_date_diff REAL,
    net_flow_lag1   REAL,
    net_flow_lag2   REAL,
    net_flow_lag3   REAL,
    net_flow_lag4   REAL,
    net_flow_lag5   REAL,
    net_flow_roll_mean_30 REAL,
    net_flow_roll_std_30  REAL,
    break_type      TEXT,
    is_break        INTEGER,
    ingested_at     TEXT    NOT NULL,
    UNIQUE(trade_date, isin, institution)
);
CREATE INDEX IF NOT EXISTS ix_feat_date ON features(trade_date);
CREATE INDEX IF NOT EXISTS ix_feat_isin ON features(isin);
"""

STORE_COLS = [
    "trade_date", "isin", "institution",
    "net_flow_nok", "gross_buy_nok", "gross_sell_nok",
    "order_count", "ask_flow_nok", "settle_lag",
    "weekday", "is_month_end", "is_month_start",
    "ext_amount", "settle_date_diff",
    "net_flow_lag1", "net_flow_lag2", "net_flow_lag3",
    "net_flow_lag4", "net_flow_lag5",
    "net_flow_roll_mean_30", "net_flow_roll_std_30",
    "break_type", "is_break",
]


@contextmanager
def db_conn():
    c = sqlite3.connect(FEATURE_DB)
    try:
        yield c
        c.commit()
    except Exception:
        c.rollback()
        raise
    finally:
        c.close()


def init_store():
    FEATURE_DB.parent.mkdir(parents=True, exist_ok=True)
    with db_conn() as c:
        c.executescript(SCHEMA)
    log.info("Feature store klar: %s", FEATURE_DB)


def upsert_features(df: pd.DataFrame) -> tuple[int, int]:
    """
    Skriv feature-rader til SQLite.
    Allerede kjente rader (samme trade_date + isin + institution) oppdateres
    slik at break_type kan komme inn senere enn rå features.
    """
    if df.empty:
        return 0, 0

    now = datetime.utcnow().isoformat(timespec="seconds")
    df = df.copy()
    df["trade_date"] = df["trade_date"].astype(str)

    # Bare ta med kolonner som finnes i skjema
    cols = [c for c in STORE_COLS if c in df.columns]

    inserted = updated = 0
    with db_conn() as c:
        for _, row in df[cols].iterrows():
            params = {col: (None if pd.isna(row[col]) else row[col]) for col in cols}
            params["ingested_at"] = now
            try:
                c.execute(
                    f"INSERT INTO features ({','.join(cols)}, ingested_at) "
                    f"VALUES ({','.join(':'+col for col in cols)}, :ingested_at)",
                    params,
                )
                inserted += 1
            except sqlite3.IntegrityError:
                # Oppdater break_type hvis den har kommet inn nå
                c.execute(
                    """UPDATE features
                       SET break_type=:break_type, is_break=:is_break, ingested_at=:ingested_at
                       WHERE trade_date=:trade_date AND isin=:isin AND institution=:institution""",
                    params,
                )
                updated += 1

    log.info("Feature store: %d nye, %d oppdaterte rader", inserted, updated)
    return inserted, updated


def load_all_features() -> pd.DataFrame:
    """Hent all historikk fra feature store."""
    with db_conn() as c:
        df = pd.read_sql_query("SELECT * FROM features ORDER BY trade_date", c)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    log.info("Hentet %d rader fra feature store (%d unike dager)",
             len(df), df["trade_date"].nunique())
    return df


# ---------------------------------------------------------------------------
# Lag 1 – XGBoost aggregeringsmodell
# ---------------------------------------------------------------------------

def train_aggregation_model(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    today_str: str,
) -> tuple[XGBRegressor | None, dict]:
    """
    Tren XGBoost-regressor som predikerer faktisk depotbeløp (ext_amount).
    Returnerer (modell, metrics-dict).
    """
    feature_cols = [c for c in AGG_FEATURES if c in train_df.columns]
    target_col   = "ext_amount"

    # Trenger rader der vi faktisk har et eksternt beløp å trene mot
    train_clean = train_df[train_df[target_col].notna() &
                           (train_df[target_col] != 0)].copy()
    val_clean   = val_df[val_df[target_col].notna()].copy()

    if len(train_clean) < 10:
        log.warning("Lag 1: for få treningsrader (%d) – hopper over", len(train_clean))
        return None, {"lag1_status": "skipped_too_few_rows"}

    X_train = train_clean[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    y_train = pd.to_numeric(train_clean[target_col], errors="coerce")

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
        early_stopping_rounds=20,
    )

    # Valideringssett for early stopping
    if len(val_clean) > 0:
        X_val = val_clean[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        y_val = pd.to_numeric(val_clean[target_col], errors="coerce")
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  verbose=False)
        preds  = model.predict(X_val)
        mae    = mean_absolute_error(y_val, preds)
        mean_a = y_val.abs().mean() or 1
        mae_pct = mae / mean_a
        status  = "ok" if mae_pct < MAE_PCT_THRESH else "warn_mae_high"
        log.info("Lag 1 MAE: %.0f NOK (%.2f%%) – %s", mae, mae_pct * 100, status)
        metrics = {
            "lag1_mae_nok":   round(mae, 2),
            "lag1_mae_pct":   round(mae_pct * 100, 4),
            "lag1_val_rows":  len(val_clean),
            "lag1_train_rows": len(train_clean),
            "lag1_status":    status,
        }
    else:
        model.fit(X_train, y_train, verbose=False)
        metrics = {"lag1_train_rows": len(train_clean), "lag1_status": "no_val"}

    # Lagre modell
    model_path = MODELS_DIR / f"agg_{today_str}.pkl"
    joblib.dump({"model": model, "feature_cols": feature_cols}, model_path)
    log.info("Lag 1-modell lagret: %s", model_path)

    return model, metrics


# ---------------------------------------------------------------------------
# Lag 3 – Random Forest årsaksklassifikator
# ---------------------------------------------------------------------------

def train_classifier(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    today_str: str,
) -> tuple[RandomForestClassifier | None, LabelEncoder | None, dict]:
    """
    Tren Random Forest-klassifikator på merkede brudd.
    Returnerer (modell, label_encoder, metrics-dict).
    """
    feature_cols = [c for c in CLF_FEATURES if c in train_df.columns]

    # Bare bruk rader der vi har ekte brudd (ikke NO_BREAK eller UNLABELLED)
    valid_breaks = train_df[
        train_df["break_type"].notna() &
        ~train_df["break_type"].isin(["NO_BREAK", "UNLABELLED"])
    ].copy()

    if len(valid_breaks) < MIN_BREAK_ROWS:
        log.info("Lag 3: %d merkede brudd – trenger ≥ %d. Hopper over.",
                 len(valid_breaks), MIN_BREAK_ROWS)
        return None, None, {
            "lag3_status": "skipped_too_few_breaks",
            "lag3_labelled_rows": len(valid_breaks),
        }

    # Klasse-fordeling
    dist = valid_breaks["break_type"].value_counts()
    log.info("Lag 3 klasser:\n%s", dist.to_string())

    le = LabelEncoder()
    X  = valid_breaks[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    y  = le.fit_transform(valid_breaks["break_type"])

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)

    # Validering
    val_breaks = val_df[
        val_df["break_type"].notna() &
        ~val_df["break_type"].isin(["NO_BREAK", "UNLABELLED"])
    ].copy()

    if len(val_breaks) > 0:
        X_val  = val_breaks[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        y_val  = le.transform(
            val_breaks["break_type"].map(
                lambda x: x if x in le.classes_ else le.classes_[0]
            )
        )
        preds  = model.predict(X_val)
        f1     = f1_score(y_val, preds, average="macro", zero_division=0)
        report = classification_report(y_val, preds,
                                        target_names=le.classes_,
                                        zero_division=0)
        log.info("Lag 3 F1 makro: %.3f\n%s", f1, report)
        metrics = {
            "lag3_f1_macro":      round(f1, 4),
            "lag3_train_breaks":  len(valid_breaks),
            "lag3_val_breaks":    len(val_breaks),
            "lag3_status":        "ok",
        }
    else:
        metrics = {
            "lag3_train_breaks": len(valid_breaks),
            "lag3_status":       "trained_no_val",
        }

    # Lagre
    model_path = MODELS_DIR / f"clf_{today_str}.pkl"
    joblib.dump({"model": model, "le": le, "feature_cols": feature_cols}, model_path)
    log.info("Lag 3-modell lagret: %s", model_path)

    return model, le, metrics


# ---------------------------------------------------------------------------
# Metrics-logging
# ---------------------------------------------------------------------------

def log_metrics(today_str: str, train_days: int, metrics: dict):
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    row = {
        "date":       today_str,
        "train_days": train_days,
        **metrics,
    }
    row_df = pd.DataFrame([row])
    if METRICS_CSV.exists():
        existing = pd.read_csv(METRICS_CSV)
        # Oppdater hvis dato finnes, ellers legg til
        existing = existing[existing["date"] != today_str]
        row_df = pd.concat([existing, row_df], ignore_index=True)
    row_df.to_csv(METRICS_CSV, index=False)
    log.info("Metrics skrevet til %s", METRICS_CSV)


def print_metrics_summary(metrics: dict):
    print("\n" + "=" * 50)
    print("TRENINGSRESULTAT")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k:<28} {v}")
    print("=" * 50)


# ---------------------------------------------------------------------------
# Finn siste modell
# ---------------------------------------------------------------------------

def latest_model(prefix: str) -> Path | None:
    """Returner nyeste modell med gitt prefix (agg_ eller clf_)."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    candidates = sorted(MODELS_DIR.glob(f"{prefix}*.pkl"))
    return candidates[-1] if candidates else None


# ---------------------------------------------------------------------------
# Hoved-pipeline
# ---------------------------------------------------------------------------

def run_daily(
    genus_path:  Path,
    odin_path:   Path | None,
    af_path:     Path | None,
    breaks_path: Path | None,
    force_date:  str | None = None,
):
    """
    Kjør komplett daglig pipeline for én dag.

    1. Les dagens filer → feature-rad
    2. Lagre i feature store
    3. Hent all historikk
    4. Walk-forward split (trening t.o.m. i går, validering i dag)
    5. Tren modeller
    6. Logg metrics
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    init_store()

    # --- Steg 1: Les dagens filer ---
    log.info("=== STEG 1: Lesing av daglige filer ===")

    genus_df = mf.read_genus(genus_path)
    ext_dfs  = []
    if odin_path and odin_path.exists():
        ext_dfs.append(mf.read_odin(odin_path))
    if af_path and af_path.exists():
        ext_dfs.append(mf.read_allfunds(af_path))

    breaks_df = None
    if breaks_path and breaks_path.exists():
        breaks_df = mf.read_breaks(breaks_path)

    internal = mf.aggregate_internal(genus_df)
    external = mf.aggregate_external(ext_dfs)
    today_features = mf.build_features(internal, external, breaks_df, genus_df)

    if today_features.empty:
        log.error("Ingen features generert for dagens filer – avbryter")
        return

    log.info("Dagens feature-rader: %d", len(today_features))

    # Bestem hva som er "i dag" (trade_date med flest rader)
    today_date = (force_date or
                  today_features["trade_date"].value_counts().idxmax().strftime("%Y-%m-%d"))
    today_str  = today_date.replace("-", "")
    log.info("Behandler dag: %s", today_date)

    # --- Steg 2: Lagre i feature store ---
    log.info("=== STEG 2: Lagrer i feature store ===")
    upsert_features(today_features)

    # --- Steg 3: Hent all historikk ---
    log.info("=== STEG 3: Henter historikk ===")
    all_data = load_all_features()
    n_days   = all_data["trade_date"].nunique()

    if n_days < MIN_TRAIN_DAYS:
        log.info("Bare %d dager i store – trenger ≥ %d. Samler historikk.",
                 n_days, MIN_TRAIN_DAYS)
        log_metrics(today_str, n_days, {"status": f"collecting_{n_days}_of_{MIN_TRAIN_DAYS}_days"})
        print(f"\nSamler historikk: {n_days}/{MIN_TRAIN_DAYS} dager. "
              f"Trening starter om {MIN_TRAIN_DAYS - n_days} dager.")
        return

    # --- Steg 4: Walk-forward split ---
    log.info("=== STEG 4: Walk-forward split ===")
    all_data = all_data.sort_values("trade_date")
    cutoff   = pd.Timestamp(today_date)

    train_df = all_data[all_data["trade_date"] < cutoff].copy()
    val_df   = all_data[all_data["trade_date"] == cutoff].copy()

    log.info("Trening: %d rader (%d dager) | Validering: %d rader",
             len(train_df), train_df["trade_date"].nunique(), len(val_df))

    # --- Steg 5: Tren modeller ---
    all_metrics: dict = {}

    log.info("=== STEG 5a: Lag 1 – XGBoost ===")
    _, agg_metrics = train_aggregation_model(train_df, val_df, today_str)
    all_metrics.update(agg_metrics)

    log.info("=== STEG 5b: Lag 3 – Random Forest ===")
    _, _, clf_metrics = train_classifier(train_df, val_df, today_str)
    all_metrics.update(clf_metrics)

    all_metrics["train_days"] = n_days

    # --- Steg 6: Logg metrics ---
    log.info("=== STEG 6: Logger metrics ===")
    log_metrics(today_str, n_days, all_metrics)
    print_metrics_summary(all_metrics)

    # Vis siste 5 dagers metrics
    if METRICS_CSV.exists():
        hist = pd.read_csv(METRICS_CSV).tail(5)
        print("\nSiste 5 dager:")
        lag1_cols = [c for c in ["date", "train_days", "lag1_mae_pct", "lag1_status",
                                  "lag3_f1_macro", "lag3_status"] if c in hist.columns]
        print(hist[lag1_cols].to_string(index=False))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Daglig ML-treningspipeline")
    parser.add_argument("--genus",  required=True,  help="Genus-fil for dagen (.xlsx)")
    parser.add_argument("--odin",   default=None,   help="ODIN-fil for dagen (.xlsx)")
    parser.add_argument("--af",     default=None,   help="Allfunds-fil for dagen (.txt)")
    parser.add_argument("--breaks", default=None,   help="Breaks-fil for dagen (.xlsx)")
    parser.add_argument("--date",   default=None,
                        help="Overrid dato (YYYY-MM-DD), brukes til testing")
    args = parser.parse_args()

    run_daily(
        genus_path  = Path(args.genus),
        odin_path   = Path(args.odin)   if args.odin   else None,
        af_path     = Path(args.af)     if args.af     else None,
        breaks_path = Path(args.breaks) if args.breaks else None,
        force_date  = args.date,
    )


if __name__ == "__main__":
    main()
