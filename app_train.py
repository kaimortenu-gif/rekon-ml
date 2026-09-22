"""
app_train.py — Streamlit-app for daglig modelltrening med GitHub-lagring

Kjøring lokalt:
    GITHUB_TOKEN=ghp_xxx GITHUB_REPO=bruker/repo streamlit run app_train.py

P� Streamlit Cloud:
    Sett GITHUB_TOKEN, GITHUB_REPO og GITHUB_BRANCH i Streamlit Secrets.
    Appen laster ned feature_store.db og modeller fra GitHub ved oppstart,
    og committer dem tilbake etter trening.
"""
from __future__ import annotations

import io
import logging
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

import ml_features as mf
import ml_train as mt
import github_storage as gs

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Konstanter – lokale stier i Streamlit Cloud-containeren
# ---------------------------------------------------------------------------

LOCAL_DB         = Path("data/feature_store.db")
LOCAL_MODELS_DIR = Path("models")
mt.FEATURE_DB    = LOCAL_DB          # pek ml_train til riktig sti
mt.MODELS_DIR    = LOCAL_MODELS_DIR

# ---------------------------------------------------------------------------
# Sideoppsett
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Oppgjørsrekon – Modelltrening",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Oppgjørsrekonsilering – Daglig modelltrening")
st.caption(
    "Last opp dagens fire filer, se forhåndsvisning og tren modellen. "
    "Feature store og modeller lagres automatisk i GitHub."
)

# ---------------------------------------------------------------------------
# Synkroniser fra GitHub ved oppstart (én gang per sesjon)
# ---------------------------------------------------------------------------

if "github_synced" not in st.session_state:
    with st.spinner("Laster ned historikk og modeller fra GitHub …"):
        try:
            sync_result = gs.sync_from_github(LOCAL_DB, LOCAL_MODELS_DIR, mt.METRICS_CSV)
            st.session_state["github_synced"] = True
            st.session_state["sync_result"]   = sync_result
        except RuntimeError as e:
            st.error(str(e))
            st.info(
                "Sett `GITHUB_TOKEN`, `GITHUB_REPO` og `GITHUB_BRANCH` "
                "i Streamlit Secrets (Settings → Secrets)."
            )
            st.stop()

sync_result = st.session_state.get("sync_result", {})
if sync_result.get("db"):
    st.toast("✅ Historikk lastet fra GitHub", icon="📥")
else:
    st.toast("Starter uten historikk – ny feature store opprettes", icon="🆕")

# ---------------------------------------------------------------------------
# Hjelpefunksjoner
# ---------------------------------------------------------------------------

def _tmp(data: bytes, suffix: str) -> Path:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(data)
    tmp.flush()
    return Path(tmp.name)


def _preview_df(df: pd.DataFrame, label: str, n: int = 5):
    st.markdown(f"**{label}** — {len(df):,} rader")
    st.dataframe(df.head(n), use_container_width=True, hide_index=True)


def _breaks_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (df["break_type"]
            .value_counts()
            .reset_index()
            .rename(columns={"break_type": "Årsak", "count": "Antall"}))


# ---------------------------------------------------------------------------
# Steg 1 – Last opp filer
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("Steg 1 — Last opp dagens filer")

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

with col1:
    genus_file = st.file_uploader("Genus (.xlsx)", type=["xlsx"], key="genus")
with col2:
    af_file = st.file_uploader("Allfunds (.txt)", type=["txt"], key="af")
with col3:
    odin_file = st.file_uploader("ODIN (.xlsx)", type=["xlsx"], key="odin")
with col4:
    breaks_file = st.file_uploader("Breaks (.xlsx)", type=["xlsx"], key="breaks")

all_uploaded = all([genus_file, af_file, odin_file, breaks_file])

if not all_uploaded:
    missing = [n for n, f in [("Genus", genus_file), ("Allfunds", af_file),
                                ("ODIN", odin_file), ("Breaks", breaks_file)] if not f]
    st.info(f"Venter på: {', '.join(missing)}")
    st.stop()

st.success("✅ Alle fire filer lastet opp")

# Les bytes én gang – unngår at Streamlit re-setter file-pointer
genus_bytes  = genus_file.read()
af_bytes     = af_file.read()
odin_bytes   = odin_file.read()
breaks_bytes = breaks_file.read()

# ---------------------------------------------------------------------------
# Steg 2 – Forhåndsvisning (caches på filinnhold)
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("Steg 2 — Forhåndsvisning")

@st.cache_data(show_spinner="Leser filer …")
def load_previews(gb, ab, ob, bb):
    try:
        genus_df  = mf.read_genus(_tmp(gb, ".xlsx"))
        af_df     = mf.read_allfunds(_tmp(ab, ".txt"))
        odin_df   = mf.read_odin(_tmp(ob, ".xlsx"))
        breaks_df = mf.read_breaks(_tmp(bb, ".xlsx"))
        internal  = mf.aggregate_internal(genus_df)
        external  = mf.aggregate_external([af_df, odin_df])
        features  = mf.build_features(internal, external, breaks_df, genus_df)
        return genus_df, af_df, odin_df, breaks_df, features, None
    except Exception as e:
        return None, None, None, None, None, str(e)


genus_df, af_df, odin_df, breaks_df, features_df, load_err = load_previews(
    genus_bytes, af_bytes, odin_bytes, breaks_bytes
)

if load_err:
    st.error(f"Feil ved lesing av filer: {load_err}")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["Genus", "Allfunds", "ODIN", "Breaks"])

with tab1:
    _preview_df(
        genus_df[["trade_date","settle_date","isin","institution","direction","amount_nok"]],
        "Genus – interne ordrer"
    )
with tab2:
    _preview_df(
        af_df[["trade_date","settle_date","isin","af_code","direction","amount_nok"]],
        "Allfunds – eksterne transaksjoner"
    )
with tab3:
    _preview_df(
        odin_df[["trade_date","isin","institution","direction","amount_nok"]],
        "ODIN – eksterne transaksjoner"
    )
with tab4:
    ca, cb = st.columns([2, 1])
    with ca:
        _preview_df(
            breaks_df[["trade_date","isin","af_code","break_type","custodian","qty"]],
            "Breaks – umatchede transaksjoner"
        )
    with cb:
        st.markdown("**Bruddfordeling**")
        st.dataframe(_breaks_summary(breaks_df), use_container_width=True, hide_index=True)

# Feature-sammendrag
fc1, fc2, fc3, fc4 = st.columns(4)
fc1.metric("Feature-rader", f"{len(features_df):,}")
fc2.metric("ISINer", f"{features_df['isin'].nunique():,}")
fc3.metric("Institusjoner", f"{features_df['institution'].nunique():,}")
fc4.metric("Merkede brudd",
           f"{int(features_df['is_break'].sum()) if 'is_break' in features_df.columns else 0:,}")

today_date = (features_df["trade_date"]
              .value_counts().idxmax().strftime("%Y-%m-%d"))
st.caption(f"Behandler dag: **{today_date}**")

# ---------------------------------------------------------------------------
# Steg 3 – Bekreft og tren
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("Steg 3 — Bekreft og tren")

mt.init_store()
try:
    existing = mt.load_all_features()
    n_days   = existing["trade_date"].nunique()
except Exception:
    n_days = 0

col_info, col_btn = st.columns([3, 1])
with col_info:
    if n_days >= mt.MIN_TRAIN_DAYS:
        st.info(
            f"✅ Feature store: **{n_days} dager** historikk — klar for trening"
        )
    else:
        st.info(
            f"⏳ Feature store: **{n_days}/{mt.MIN_TRAIN_DAYS} dager** — "
            f"samler historikk, {mt.MIN_TRAIN_DAYS - n_days} dager gjenstår"
        )

with col_btn:
    train_clicked = st.button(
        f"🚀 Tren  {today_date}",
        type="primary",
        use_container_width=True,
    )

if not train_clicked:
    st.stop()

# ---------------------------------------------------------------------------
# Steg 4 – Kjør trening
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("Steg 4 — Trening pågår")

progress = st.progress(0, text="Starter …")
log_box  = st.expander("Detaljert logg", expanded=False)
log_lines: list[str] = []

def _log(msg: str, pct: int):
    log_lines.append(msg)
    progress.progress(pct, text=msg)
    with log_box:
        st.text("\n".join(log_lines))


today_str   = today_date.replace("-", "")
all_metrics: dict = {}

# Feature engineering
_log("Feature engineering …", 10)
try:
    g2 = mf.read_genus(_tmp(genus_bytes, ".xlsx"))
    a2 = mf.read_allfunds(_tmp(af_bytes, ".txt"))
    o2 = mf.read_odin(_tmp(odin_bytes, ".xlsx"))
    b2 = mf.read_breaks(_tmp(breaks_bytes, ".xlsx"))
    features2 = mf.build_features(
        mf.aggregate_internal(g2),
        mf.aggregate_external([a2, o2]),
        b2, g2
    )
except Exception as e:
    st.error(f"Feature engineering feilet: {e}")
    st.stop()

# Lagre i feature store
_log("Lagrer i feature store …", 25)
mt.upsert_features(features2)

# Hent all historikk
_log("Henter historikk …", 35)
all_data = mt.load_all_features()
n_days2  = all_data["trade_date"].nunique()

if n_days2 < mt.MIN_TRAIN_DAYS:
    progress.progress(100, text="Ferdig")
    st.warning(
        f"**{n_days2}/{mt.MIN_TRAIN_DAYS} dager** i feature store. "
        f"Lagret i GitHub – kom tilbake om {mt.MIN_TRAIN_DAYS - n_days2} dager."
    )
    # Commit db til GitHub selv uten trening
    _log("Committer feature store til GitHub …", 90)
    try:
        gs.commit_to_github(LOCAL_DB, LOCAL_MODELS_DIR, today_str, mt.METRICS_CSV)
        st.toast("✅ Feature store committet til GitHub", icon="📤")
    except Exception as e:
        st.warning(f"GitHub commit feilet: {e}")
    st.stop()

# Walk-forward split
_log("Walk-forward split …", 45)
all_data = all_data.sort_values("trade_date")
cutoff   = pd.Timestamp(today_date)
train_df = all_data[all_data["trade_date"] < cutoff].copy()
val_df   = all_data[all_data["trade_date"] == cutoff].copy()

# Lag 1 – XGBoost
_log("Trener Lag 1 – XGBoost …", 55)
try:
    _, agg_m = mt.train_aggregation_model(train_df, val_df, today_str)
    all_metrics.update(agg_m)
except Exception as e:
    st.error(f"Lag 1 feilet: {e}")
    all_metrics["lag1_status"] = f"error: {e}"

# Lag 3 – Random Forest
_log("Trener Lag 3 – Random Forest …", 70)
try:
    _, _, clf_m = mt.train_classifier(train_df, val_df, today_str)
    all_metrics.update(clf_m)
except Exception as e:
    st.error(f"Lag 3 feilet: {e}")
    all_metrics["lag3_status"] = f"error: {e}"

# Logger metrics
_log("Logger metrics …", 82)
all_metrics["train_days"] = n_days2
mt.log_metrics(today_str, n_days2, all_metrics)

# Commit til GitHub
_log("Committer til GitHub …", 90)
try:
    commit_result = gs.commit_to_github(LOCAL_DB, LOCAL_MODELS_DIR, today_str, mt.METRICS_CSV)
    github_ok = True
except Exception as e:
    st.warning(f"GitHub commit feilet: {e}")
    commit_result = {"db": False, "models": []}
    github_ok = False

progress.progress(100, text="✅ Ferdig")

# ---------------------------------------------------------------------------
# Resultater
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("Resultater")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Treningsdager", n_days2)
m2.metric("Treningsrader", f"{len(train_df):,}")

lag1_pct = all_metrics.get("lag1_mae_pct")
if lag1_pct is not None:
    m3.metric(
        "Lag 1 MAE",
        f"{lag1_pct:.2f}%",
        delta="OK" if lag1_pct < 0.5 else "Høy",
        delta_color="normal" if lag1_pct < 0.5 else "inverse",
    )
else:
    m3.metric("Lag 1 MAE", all_metrics.get("lag1_status", "–"))

lag3_f1 = all_metrics.get("lag3_f1_macro")
m4.metric("Lag 3 F1 makro", f"{lag3_f1:.3f}" if lag3_f1 else all_metrics.get("lag3_status", "–"))

# Bruddfordeling og metrics-historikk
r1, r2 = st.columns(2)

with r1:
    st.markdown("**Bruddfordeling – treningsdata**")
    breaks_in_train = train_df[
        train_df["break_type"].notna() &
        ~train_df["break_type"].isin(["NO_BREAK", "UNLABELLED"])
    ]
    if not breaks_in_train.empty:
        dist = (breaks_in_train["break_type"]
                .value_counts()
                .reset_index()
                .rename(columns={"break_type": "Årsak", "count": "Antall"}))
        st.dataframe(dist, use_container_width=True, hide_index=True)
    else:
        st.caption("Ingen merkede brudd i treningsdata ennå.")

with r2:
    st.markdown("**Metrics-historikk (siste 10 dager)**")
    if mt.METRICS_CSV.exists():
        hist = pd.read_csv(mt.METRICS_CSV).tail(10)
        show_cols = [c for c in ["date","train_days","lag1_mae_pct",
                                  "lag1_status","lag3_f1_macro","lag3_status"]
                     if c in hist.columns]
        st.dataframe(hist[show_cols], use_container_width=True, hide_index=True)
    else:
        st.caption("Ingen historikk ennå.")

# GitHub-status
st.markdown("---")
if github_ok:
    committed = ", ".join(commit_result.get("models", []))
    metrics_ok = commit_result.get("metrics", False)
    st.success(
        f"✅ Committet til GitHub:  \n"
        f"- `data/feature_store.db`  \n"
        + (f"- `models/{committed}`  \n" if committed else "- (ingen nye modeller)  \n")
        + (f"- `logs/metrics.csv`" if metrics_ok else "")
    )
else:
    st.warning("⚠️ GitHub commit feilet – modeller er kun lagret lokalt i denne sesjonen.")
