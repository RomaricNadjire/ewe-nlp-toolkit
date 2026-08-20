"""Utilitaires communs aux scripts de téléchargement de corpus (src/download/).

Fonctions partagées : détection automatique des colonnes de langue, écriture
JSONL au format projet, et génération d'un `manifest.json` par source.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

# Racine du dépôt = deux niveaux au-dessus de ce fichier (src/download/_common.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_TRANSLATION = REPO_ROOT / "data" / "raw" / "translation"
RAW_ASR = REPO_ROOT / "data" / "raw" / "asr"

# Codes NLLB cibles -> motifs de noms de colonnes (minuscules) à reconnaître.
LANG_PATTERNS: dict[str, tuple[str, ...]] = {
    "ewe_Latn": ("ewe", "éwé", "ewe_latn", "ee"),
    "eng_Latn": ("eng", "english", "anglais", "eng_latn", "en"),
    "fra_Latn": ("fra", "french", "français", "francais", "fra_latn", "fr"),
}


def detect_lang_columns(columns: Iterable[str]) -> dict[str, str]:
    """Associe code NLLB -> nom de colonne, par heuristique sur les noms.

    Les correspondances exactes (`ee`, `en`, `fr`) priment sur les sous-chaînes
    pour éviter qu'« en » dans « sentence » ne crée un faux positif.
    """
    cols = list(columns)
    lower = {c: str(c).strip().lower() for c in cols}
    mapping: dict[str, str] = {}
    for code, patterns in LANG_PATTERNS.items():
        chosen = None
        # 1) égalité exacte d'abord
        for c in cols:
            if lower[c] in patterns:
                chosen = c
                break
        # 2) sinon, sous-chaîne (motifs >= 3 lettres pour limiter le bruit)
        if chosen is None:
            for c in cols:
                if any(p in lower[c] for p in patterns if len(p) >= 3):
                    chosen = c
                    break
        if chosen is not None and chosen not in mapping.values():
            mapping[code] = chosen
    return mapping


def row_to_translation(row: dict, colmap: dict[str, str]) -> dict | None:
    """Construit `{"translation": {code: texte, ...}}` à partir d'une ligne.

    Conserve une éventuelle colonne `similarity`/`score` en clé sœur pour le
    filtrage ultérieur (corpus minés). Retourne None si l'éwé est absent/vide.
    """
    trans: dict[str, str] = {}
    for code, col in colmap.items():
        val = row.get(col)
        if val is None:
            continue
        val = str(val).strip()
        if val:
            trans[code] = val
    if not trans.get("ewe_Latn"):
        return None
    rec: dict = {"translation": trans}
    for sim_col in ("similarity", "score", "laser_score", "cosine"):
        if sim_col in row and row[sim_col] is not None:
            try:
                rec["similarity"] = float(row[sim_col])
            except (TypeError, ValueError):
                pass
            break
    return rec


def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    """Écrit les enregistrements en JSONL (UTF-8). Retourne le nombre de lignes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def write_manifest(dest_dir: Path, **fields) -> None:
    """Écrit `dest_dir/manifest.json` (provenance, licence, format, comptes, date)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    fields.setdefault("downloaded_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    (dest_dir / "manifest.json").write_text(
        json.dumps(fields, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def require(module: str, pip_name: str | None = None):
    """Importe `module` ou termine avec un message d'installation clair."""
    try:
        return __import__(module)
    except ImportError:
        pip = pip_name or module
        sys.exit(
            f"[ERREUR] Module '{module}' introuvable.\n"
            f"         Installez-le :  pip install {pip}"
        )
