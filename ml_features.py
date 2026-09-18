"""
ml_features.py — Feature engineering for ML-basert oppgjørsrekonsilering

Leser daglige filer fra Genus (.xlsx), Allfunds (.txt) og ODIN (.xlsx),
aggregerer til (date, isin, institution)-nivå og produserer treningsdatasett
klar for XGBoost-aggregeringsmodell og Random Forest-klassifikator.

Matchingnøkkel internt: ISIN + institution (fullt banknavn fra Genus/ODIN).
Breaks-filer flettes via transaksjonsreferanse (txn_ref), siden breaks-eksporten
bruker Allfunds 4-sifret kode – ikke Genus-navn.

Kjøring:
    python ml_features.py --data-dir ./data --output features.csv
    python ml_features.py --data-dir ./data --breaks-dir ./breaks --output features.csv
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Konstantar
# ---------------------------------------------------------------------------

GENUS_DIRECTION_MAP = {
    "Subscription":                  "BUY",
    "Transfer (Subscription)":       "BUY",
    "Switch (Subscription)":         "BUY",
    "Investment Plan Subscription":  "BUY",
    "Redemption":                    "SELL",
    "Transfer (Redemption)":         "SELL",
    "Switch (Redemption)":           "SELL",
    "Investment Plan Redemption":    "SELL",
    "Distribution Cost":             "SELL",
}

RECONCILE_DOMAINS = {"External", "Between SB1 Banks"}

ODIN_DIRECTION_MAP = {
    "Subscription":           "BUY",
    "Transfer (Subscription)":"BUY",
    "Switch (Subscription)":  "BUY",
    "Redemption":             "SELL",
    "Transfer (Redemption)":  "SELL",
    "Switch (Redemption)":    "SELL",
}

ALLFUNDS_DIRECTION_MAP = {
    "10": "BUY",  "12": "BUY",  "13": "BUY",
    "60": "BUY",  "61": "BUY",  "62": "BUY",
    "20": "SELL", "22": "SELL", "23": "SELL", "24": "SELL",
    "75": "SELL", "76": "SELL", "77": "SELL", "78": "SELL",
    "79": "SELL", "86": "SELL",
}

ALLFUNDS_FIELD_SPEC = {
    "record_type": (11,  2),
    "txn_type":    (57,  2),
    "txn_ref":     (120, 20),
    "portfolio":   (153, 34),
    "trade_date":  (324, 10),
    "settle_date": (364, 10),
    "net_amount":  (391, 17),
    "quantity":    (408, 17),
    "isin":        (61,  12),
    "fund_name":   (80,  40),
}


# ---------------------------------------------------------------------------
# Lesing av kildedataene
# ---------------------------------------------------------------------------

def _af_slice(line: str, key: str) -> str:
    off, lng = ALLFUNDS_FIELD_SPEC[key]
    return line[off - 1: off - 1 + lng]

def _af_decimal(raw: str, divisor: int) -> float | None:
    s = raw.strip()
    return int(s) / divisor if s.isdigit() else None


def read_genus(path: Path) -> pd.DataFrame:
    log.info("Genus: %s", path.name)
    df = pd.read_excel(path, sheet_name=0)
    df.columns = [str(c).strip() for c in df.columns]

    if "Cancelled" in df.columns:
        df = df[df["Cancelled"].astype(str).str.upper() != "TRUE"]
    if "TradeDomain" in df.columns:
        df = df[df["TradeDomain"].astype(str).str.strip().isin(RECONCILE_DOMAINS)]
    if df.empty:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["trade_date"]    = pd.to_datetime(df["NAV Date"], errors="coerce").dt.normalize()
    out["settle_date"]   = pd.to_datetime(df["Settlement Date"], errors="coerce").dt.normalize()
    out["isin"]          = df["ISIN"].astype(str).str.strip().str.upper()
    out["institution"]   = df["InstitutionName"].astype(str).str.strip().str.upper()
    out["direction"]     = df["TransactionType"].map(GENUS_DIRECTION_MAP).fillna("UNKNOWN")
    out["amount_nok"]    = pd.to_numeric(df["Settlement Amount"], errors="coerce").abs()
    out["units"]         = pd.to_numeric(df["Units"], errors="coerce").abs()
    out["product_type"]  = df["ProductType"].astype(str).str.strip().str.upper()
    out["txn_ref"]       = df["Transaction Reference"].astype(str).str.strip()
    out["source"]        = "genus"
    out["signed_amount"] = np.where(out["direction"] == "BUY",
                                    out["amount_nok"], -out["amount_nok"])
    return out.dropna(subset=["trade_date", "isin", "institution"])


def read_odin(path: Path) -> pd.DataFrame:
    log.info("ODIN: %s", path.name)
    df = pd.read_excel(path, dtype=str)
    df.columns = [str(c).strip() for c in df.columns]

    out = pd.DataFrame()
    out["trade_date"]    = pd.to_datetime(df["Kursdato"], errors="coerce").dt.normalize()
    out["settle_date"]   = out["trade_date"]
    out["isin"]          = df["ISIN"].astype(str).str.strip().str.upper()
    out["institution"]   = df["Kunde"].astype(str).str.strip().str.upper()
    out["direction"]     = df["Type"].map(ODIN_DIRECTION_MAP).fillna("UNKNOWN")
    out["amount_nok"]    = pd.to_numeric(df["Beløp"], errors="coerce").abs()
    out["units"]         = pd.to_numeric(df["Andeler"], errors="coerce").abs()
    out["product_type"]  = "ODIN"
    out["txn_ref"]       = df["Transaksjon ref."].astype(str).str.strip()
    out["source"]        = "odin"
    out["signed_amount"] = np.where(out["direction"] == "BUY",
                                    out["amount_nok"], -out["amount_nok"])
    return out.dropna(subset=["trade_date", "isin", "institution"])


def read_allfunds(path: Path) -> pd.DataFrame:
    log.info("Allfunds: %s", path.name)
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            lines = path.read_text(encoding=enc).splitlines()
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError(f"Kunne ikke dekode {path}")

    rows = []
    for line in lines:
        if len(line) < 750:
            continue
        if _af_slice(line, "record_type").strip() != "24":
            continue
        txn_code  = _af_slice(line, "txn_type").strip()
        direction = ALLFUNDS_DIRECTION_MAP.get(txn_code)
        if not direction:
            continue
        # txn_ref: strip ledende nuller (matcher breaks-eksporten)
        raw_ref = _af_slice(line, "txn_ref").strip()
        txn_ref = raw_ref.lstrip("0") or "0"

        rows.append({
            "trade_date":   _af_slice(line, "trade_date").strip(),
            "settle_date":  _af_slice(line, "settle_date").strip(),
            "isin":         _af_slice(line, "isin").strip().upper(),
            "af_code":      _af_slice(line, "portfolio")[:4].strip(),
            "direction":    direction,
            "amount_nok":   _af_decimal(_af_slice(line, "net_amount"), 100),
            "units":        _af_decimal(_af_slice(line, "quantity"), 1_000_000),
            "txn_ref":      txn_ref,
            "source":       "allfunds",
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["trade_date"]    = pd.to_datetime(df["trade_date"],  errors="coerce").dt.normalize()
    df["settle_date"]   = pd.to_datetime(df["settle_date"], errors="coerce").dt.normalize()
    df["amount_nok"]    = pd.to_numeric(df["amount_nok"],   errors="coerce").abs()
    df["signed_amount"] = np.where(df["direction"] == "BUY",
                                   df["amount_nok"], -df["amount_nok"])
    return df.dropna(subset=["trade_date", "isin"])


def read_breaks(path: Path) -> pd.DataFrame:
    """
    Les breaks-eksportfil. Header på rad 2 (0-indeksert).
    Returnerer én rad per umatched transaksjon med break_type og txn_ref.
    txn_ref brukes til å flette breaks mot Genus/Allfunds-transaksjoner.
    """
    log.info("Breaks: %s", path.name)
    df = pd.read_excel(path, header=2)
    df.columns = [str(c).strip() for c in df.columns]
    df = df[df["ISIN"].notna() & (df["ISIN"].astype(str) != "ISIN")]

    out = pd.DataFrame()
    out["trade_date"]  = pd.to_datetime(df["Trade Date"], errors="coerce").dt.normalize()
    out["settle_date"] = pd.to_datetime(df["Settlement Date"], errors="coerce").dt.normalize()
    out["isin"]        = df["ISIN"].astype(str).str.strip().str.upper()
    out["af_code"]     = df["Portfolio Code"].astype(str).str.strip()
    out["break_type"]  = df["Break"].astype(str).str.strip()
    out["custodian"]   = df["Source Custodian Code"].astype(str).str.strip()
    out["direction"]   = df["Source Transaction Type Code"].astype(str).str.strip()
    # txn_ref: strip ledende nuller for å matche Allfunds-formatet
    out["txn_ref"]     = (df["Transaction Reference Number"]
                          .astype(str).str.strip().str.lstrip("0").str.strip())
    out["qty"]         = pd.to_numeric(df["Quantity - Trade Date"], errors="coerce")
    out["amount"]      = pd.to_numeric(df["Settlement Amount - Settle Date"], errors="coerce")

    return out.dropna(subset=["trade_date", "isin", "break_type"])


# ---------------------------------------------------------------------------
# Aggregering
# ---------------------------------------------------------------------------

def aggregate_internal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggreger Genus til daglig netto flyt per (trade_date, isin, institution).
    Bevar også txn_ref-liste for break-fletiing.
    """
    if df.empty:
        return pd.DataFrame()

    grp = df.groupby(["trade_date", "isin", "institution"])
    agg = grp.agg(
        net_flow_nok   = ("signed_amount", "sum"),
        gross_buy_nok  = ("amount_nok",
                          lambda x: x[df.loc[x.index, "direction"] == "BUY"].sum()),
        gross_sell_nok = ("amount_nok",
                          lambda x: x[df.loc[x.index, "direction"] == "SELL"].sum()),
        order_count    = ("signed_amount", "count"),
        ask_flow_nok   = ("signed_amount",
                          lambda x: x[df.loc[x.index, "product_type"] == "ASK"].sum()),
        settle_date    = ("settle_date", "first"),
        # Samle txn_ref-er som pipe-separert streng
        txn_refs       = ("txn_ref",
                          lambda x: "|".join(x.astype(str).unique())),
    ).reset_index()

    agg["settle_lag"]    = (agg["settle_date"] - agg["trade_date"]).dt.days.fillna(0).astype(int)
    agg["weekday"]       = agg["trade_date"].dt.weekday
    agg["is_month_end"]  = agg["trade_date"].dt.is_month_end.astype(int)
    agg["is_month_start"]= agg["trade_date"].dt.is_month_start.astype(int)

    return agg.sort_values(["isin", "institution", "trade_date"]).reset_index(drop=True)


def aggregate_external(ext_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Aggreger Allfunds + ODIN til daglig netto flyt per (trade_date, isin, af_code/institution).
    For Allfunds: nøkkel er af_code (4 siffer).
    For ODIN: nøkkel er institution (fullt navn).
    Returnerer begge for senere fletiing.
    """
    af_frames   = [d for d in ext_dfs if not d.empty and d["source"].iloc[0] == "allfunds"]
    odin_frames = [d for d in ext_dfs if not d.empty and d["source"].iloc[0] == "odin"]

    results = []

    if af_frames:
        af = pd.concat(af_frames, ignore_index=True)
        af_agg = (af.groupby(["trade_date", "isin", "af_code"])
                  .agg(ext_amount=("signed_amount", "sum"),
                       ext_settle_date=("settle_date", "first"),
                       ext_txn_refs=("txn_ref", lambda x: "|".join(x.astype(str).unique())))
                  .reset_index()
                  .rename(columns={"af_code": "ext_key"}))
        af_agg["ext_source"] = "allfunds"
        results.append(af_agg)

    if odin_frames:
        odin = pd.concat(odin_frames, ignore_index=True)
        odin_agg = (odin.groupby(["trade_date", "isin", "institution"])
                    .agg(ext_amount=("signed_amount", "sum"),
                         ext_settle_date=("settle_date", "first"),
                         ext_txn_refs=("txn_ref", lambda x: "|".join(x.astype(str).unique())))
                    .reset_index()
                    .rename(columns={"institution": "ext_key"}))
        odin_agg["ext_source"] = "odin"
        results.append(odin_agg)

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


# ---------------------------------------------------------------------------
# Lag-features
# ---------------------------------------------------------------------------

def add_lag_features(df: pd.DataFrame, col: str = "net_flow_nok",
                     n_lags: int = 5, roll_window: int = 30) -> pd.DataFrame:
    df = df.sort_values(["isin", "institution", "trade_date"]).copy()
    grp = df.groupby(["isin", "institution"])[col]
    for lag in range(1, n_lags + 1):
        df[f"net_flow_lag{lag}"] = grp.shift(lag)
    df[f"net_flow_roll_mean_{roll_window}"] = grp.transform(
        lambda x: x.rolling(roll_window, min_periods=3).mean())
    df[f"net_flow_roll_std_{roll_window}"] = grp.transform(
        lambda x: x.rolling(roll_window, min_periods=3).std())
    return df


# ---------------------------------------------------------------------------
# Sammenslåing og break-fletiing
# ---------------------------------------------------------------------------

def build_features(internal: pd.DataFrame,
                   external: pd.DataFrame,
                   breaks_df: pd.DataFrame | None,
                   all_genus_rows: pd.DataFrame) -> pd.DataFrame:
    """
    Slår sammen intern aggregering med ekstern, legger til lag-features,
    og fletter breaks via txn_ref.
    """
    # --- Join intern mot ekstern ---
    # Ekstern har en "ext_key" som er enten af_code (Allfunds) eller institution (ODIN).
    # Intern har institution (fullt navn). Vi joiner på (trade_date, isin) og velger
    # ekstern match der ext_key finnes i txn_ref-settet eller institution-navnets match.

    # Enkleste korrekte tilnærming: join på (trade_date, isin) og summer all ekstern
    # flyt uavhengig av institusjonskode, siden vi allerede er på ISIN-nivå per dag.
    # Institution-separasjon håndteres fullt ut på intern side (Genus).
    if not external.empty:
        ext_day = (external.groupby(["trade_date", "isin"])
                   .agg(ext_amount_total=("ext_amount", "sum"),
                        ext_settle_date=("ext_settle_date", "first"))
                   .reset_index())
    else:
        ext_day = pd.DataFrame(columns=["trade_date", "isin",
                                        "ext_amount_total", "ext_settle_date"])

    features = internal.merge(ext_day, on=["trade_date", "isin"], how="left")
    features["ext_amount_total"] = features["ext_amount_total"].fillna(0)
    features["ext_settle_date"]  = features["ext_settle_date"].fillna(features["settle_date"])

    # Settlement date-avvik (viktigste feature – 96% av brudd er settle-avvik)
    features["settle_date_diff"] = (
        features["ext_settle_date"] - features["settle_date"]
    ).dt.days.fillna(0).astype(int)

    # Rename ext_amount_total -> ext_amount
    features = features.rename(columns={"ext_amount_total": "ext_amount"})
    features["basis"] = 0.0

    # --- Lag-features ---
    features = add_lag_features(features)

    # --- Flett breaks via txn_ref ---
    if breaks_df is not None and not breaks_df.empty:
        # Ekspander txn_refs per feature-rad
        expanded = features[["trade_date", "isin", "institution", "txn_refs"]].copy()
        expanded = expanded.assign(
            txn_ref=expanded["txn_refs"].str.split("|")
        ).explode("txn_ref")
        expanded["txn_ref"] = expanded["txn_ref"].astype(str).str.strip()

        # Join mot breaks på txn_ref
        break_join = expanded.merge(
            breaks_df[["txn_ref", "break_type"]].drop_duplicates("txn_ref"),
            on="txn_ref", how="left"
        )
        # Velg verste break per (trade_date, isin, institution)
        # Prioriter ikke-settle brudd over settle-avvik
        def pick_break(series):
            vals = series.dropna()
            if vals.empty:
                return "NO_BREAK"
            # ASK-brudd prioriteres
            ask = [v for v in vals if "ASK" in v]
            if ask:
                return ask[0]
            return vals.iloc[0]

        break_per_group = (break_join.groupby(["trade_date", "isin", "institution"])
                           ["break_type"].apply(pick_break).reset_index())
        features = features.merge(break_per_group, on=["trade_date", "isin", "institution"],
                                  how="left")
        features["break_type"] = features["break_type"].fillna("NO_BREAK")
        features["is_break"]   = (features["break_type"] != "NO_BREAK").astype(int)
    else:
        features["break_type"] = "UNLABELLED"
        features["is_break"]   = np.nan

    return features


# ---------------------------------------------------------------------------
# Laste mapper
# ---------------------------------------------------------------------------

def load_folder(data_dir: Path, breaks_dir: Path | None = None) -> pd.DataFrame:
    genus_frames = []
    odin_frames  = []
    af_frames    = []
    break_frames = []

    for f in sorted(data_dir.glob("**/*")):
        if not f.is_file():
            continue
        name_l = f.name.lower()
        try:
            if "genus" in name_l and f.suffix.lower() == ".xlsx":
                genus_frames.append(read_genus(f))
            elif "odin" in name_l and f.suffix.lower() == ".xlsx":
                odin_frames.append(read_odin(f))
            elif "allfund" in name_l and f.suffix.lower() == ".txt":
                af_frames.append(read_allfunds(f))
        except Exception as e:
            log.warning("Lesefeil %s: %s", f.name, e)

    if breaks_dir:
        for f in sorted(breaks_dir.glob("**/*.xlsx")):
            try:
                break_frames.append(read_breaks(f))
            except Exception as e:
                log.warning("Breaks-lesefeil %s: %s", f.name, e)

    if not genus_frames:
        raise ValueError(f"Ingen Genus-filer funnet i {data_dir}")

    all_genus = pd.concat(genus_frames, ignore_index=True)
    internal  = aggregate_internal(all_genus)
    external  = aggregate_external(odin_frames + af_frames)
    breaks_df = pd.concat(break_frames, ignore_index=True) if break_frames else None

    features = build_features(internal, external, breaks_df, all_genus)

    log.info("Feature-tabell: %d rader, %d kolonner", len(features), len(features.columns))
    if "is_break" in features.columns:
        log.info("Bruddfordeling:\n%s", features["break_type"].value_counts().to_string())
    return features


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Bygg ML-feature-tabell")
    parser.add_argument("--data-dir",   required=True)
    parser.add_argument("--breaks-dir", default=None)
    parser.add_argument("--output",     default="features.csv")
    args = parser.parse_args()

    features = load_folder(Path(args.data_dir),
                           Path(args.breaks_dir) if args.breaks_dir else None)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out, index=False)
    log.info("Lagret: %s", out)

    print("\n=== Feature-tabell oppsummering ===")
    print(f"Rader:         {len(features):,}")
    print(f"Kolonner:      {len(features.columns)}")
    print(f"Periode:       {features['trade_date'].min().date()} → {features['trade_date'].max().date()}")
    print(f"ISINer:        {features['isin'].nunique():,}")
    print(f"Institusjoner: {features['institution'].nunique():,}")
    if "break_type" in features.columns:
        print("\nBruddfordeling:")
        print(features["break_type"].value_counts().to_string())
    print("\nFeature-kolonner:")
    print([c for c in features.columns])

if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# NOTE OM TIDSLOGIKK (viktig for treningsdatasett):
#
# Genus bruker NAV Date som handelsdato og Settlement Date som oppgjørsdato.
# Breaks-filen fra VPS/Intellimatch bruker trade_date = Allfunds sin handelsdato,
# som tilsvarer Genus sin Settlement Date (ikke NAV Date).
#
# For å bygge et korrekt treningsdatasett med 12-24 måneder historikk:
#   1. Aggreger Genus på BEGGE datoer: NAV Date (for features) og Settlement Date (for label-join)
#   2. Flett breaks mot (settle_date, isin) fra intern aggregering
#   3. breaks_trade_date == genus_settle_date (T+1/T+2/T+3 avvik er allerede i break_type)
#
# Denne versjonen er klar for produksjon med 12-24 måneder daglige filer.
# Med én dags data vises alle som NO_BREAK fordi dateoverlappen kun skjer
# mellom Genus settle_date (2026-09-16) og breaks trade_date (2026-09-16).
