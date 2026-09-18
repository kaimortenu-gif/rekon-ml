"""
github_storage.py — Persistent lagring via GitHub for Streamlit Cloud

Brukes av app_train.py til å:
  - Laste ned feature_store.db og models/*.pkl ved oppstart
  - Committe oppdatert db og nye modeller tilbake etter trening

Autentisering via Streamlit Secrets:
    GITHUB_TOKEN  = "ghp_..."
    GITHUB_REPO   = "brukernavn/rekon-ml"
    GITHUB_BRANCH = "main"

Filstørrelsesgrense i GitHub: 100 MB per fil. SQLite-databasen og
pkl-filer er typisk langt under dette. Når databasen vokser over
~50 MB bør du vurdere å bytte til Azure Blob Storage.
"""
from __future__ import annotations

import base64
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)


def _client():
    """Returner autentisert GitHub-klient via PyGithub."""
    try:
        from github import Github
        import streamlit as st
        token = st.secrets["GITHUB_TOKEN"]
    except Exception:
        # Fallback: miljøvariabel (nyttig for lokal testing)
        token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        raise RuntimeError(
            "GITHUB_TOKEN ikke satt. "
            "Legg det til i Streamlit Secrets eller som miljøvariabel."
        )
    from github import Github
    return Github(token)


def _repo():
    import streamlit as st
    try:
        repo_name = st.secrets["GITHUB_REPO"]
        branch    = st.secrets.get("GITHUB_BRANCH", "main")
    except Exception:
        repo_name = os.environ.get("GITHUB_REPO", "")
        branch    = os.environ.get("GITHUB_BRANCH", "main")
    g = _client()
    return g.get_repo(repo_name), branch


def download_file(remote_path: str, local_path: Path) -> bool:
    """
    Last ned én fil fra GitHub til lokal sti.
    Returnerer True hvis filen fantes, False hvis ikke.
    """
    try:
        repo, branch = _repo()
        content = repo.get_contents(remote_path, ref=branch)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(base64.b64decode(content.content))
        log.info("Lastet ned %s → %s", remote_path, local_path)
        return True
    except Exception as e:
        if "404" in str(e):
            log.info("Fant ikke %s i GitHub – starter uten", remote_path)
            return False
        raise


def upload_file(local_path: Path, remote_path: str, commit_msg: str) -> None:
    """
    Last opp / oppdater én fil i GitHub.
    Oppretter filen hvis den ikke finnes, oppdaterer den hvis den finnes.
    """
    repo, branch = _repo()
    content = local_path.read_bytes()
    encoded = base64.b64encode(content).decode()

    try:
        existing = repo.get_contents(remote_path, ref=branch)
        repo.update_file(
            path=remote_path,
            message=commit_msg,
            content=content,
            sha=existing.sha,
            branch=branch,
        )
        log.info("Oppdaterte %s i GitHub", remote_path)
    except Exception as e:
        if "404" in str(e):
            repo.create_file(
                path=remote_path,
                message=commit_msg,
                content=content,
                branch=branch,
            )
            log.info("Opprettet %s i GitHub", remote_path)
        else:
            raise


def sync_from_github(local_db: Path, local_models_dir: Path) -> dict:
    """
    Last ned feature_store.db og alle modeller fra GitHub ved oppstart.
    Returnerer en dict med hva som ble lastet ned.
    """
    result = {"db": False, "models": []}

    # feature_store.db
    result["db"] = download_file("data/feature_store.db", local_db)

    # Alle modeller i models/
    try:
        repo, branch = _repo()
        try:
            contents = repo.get_contents("models", ref=branch)
            if not isinstance(contents, list):
                contents = [contents]
            for f in contents:
                if f.name.endswith(".pkl"):
                    local_path = local_models_dir / f.name
                    if download_file(f"models/{f.name}", local_path):
                        result["models"].append(f.name)
        except Exception as e:
            if "404" not in str(e):
                raise
    except Exception as e:
        log.warning("Kunne ikke liste models/ i GitHub: %s", e)

    return result


def commit_to_github(
    local_db: Path,
    local_models_dir: Path,
    today_str: str,
) -> dict:
    """
    Commit feature_store.db og nye modeller tilbake til GitHub.
    Returnerer dict med hva som ble committet.
    """
    result = {"db": False, "models": []}
    msg    = f"rekon-ml: daglig oppdatering {today_str}"

    # feature_store.db
    if local_db.exists():
        upload_file(local_db, "data/feature_store.db", msg)
        result["db"] = True

    # Modeller for denne dagen
    if local_models_dir.exists():
        for pkl in sorted(local_models_dir.glob(f"*{today_str}*.pkl")):
            upload_file(pkl, f"models/{pkl.name}", msg)
            result["models"].append(pkl.name)

    return result
