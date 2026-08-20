#!/usr/bin/env python3
"""Télécharge des datasets éwé depuis Kaggle et normalise les corpus parallèles.

Nécessite des identifiants Kaggle dans `~/.kaggle/kaggle.json`
(Compte Kaggle -> Settings -> API -> « Create New Token »), puis `chmod 600`.

Chaque dataset est téléchargé/dézippé dans `data/raw/translation/<nom>/` (fichiers
bruts conservés). Si des colonnes éwé + (anglais|français) sont détectées dans un
CSV, un `all.jsonl` normalisé est aussi écrit. Un `manifest.json` est généré.

Exemples :
    python src/download/download_kaggle.py --list
    python src/download/download_kaggle.py --only tchaye59_ewe_english
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _common import (
    RAW_TRANSLATION,
    detect_lang_columns,
    row_to_translation,
    write_jsonl,
    write_manifest,
)

SOURCES = [
    {"name": "kaggle_tchaye59_ewe_english", "slug": "tchaye59/eweenglish-bilingual-pairs",
     "notes": "Paires bilingues ee-en."},
    {"name": "kaggle_yvicherita_ewe_corpus", "slug": "yvicherita/ewe-language-corpus",
     "notes": "Corpus éwé (parallèle ou monolingue selon colonnes)."},
    {"name": "kaggle_ghana_maternal_health_ewe",
     "slug": "ghanaairesnet/ghana-maternal-health-q-and-a-dataset-ewe",
     "notes": "Q/R santé maternelle en éwé (probablement monolingue/QA)."},
]


def get_kaggle_api():
    """Importe et authentifie l'API Kaggle, avec messages d'aide explicites."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: PLC0415
    except ImportError:
        sys.exit("[ERREUR] Module 'kaggle' introuvable.  pip install kaggle")
    except Exception as e:  # auth tentée à l'import si kaggle.json manquant  # noqa: BLE001
        sys.exit(f"[ERREUR] Authentification Kaggle impossible : {e}\n"
                 "         Placez vos identifiants dans ~/.kaggle/kaggle.json (chmod 600).")
    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as e:  # noqa: BLE001
        sys.exit(f"[ERREUR] Authentification Kaggle échouée : {e}\n"
                 "         Vérifiez ~/.kaggle/kaggle.json (Settings -> API -> Create New Token).")
    return api


def normalize_csvs(dest: Path) -> tuple[int, dict, list[str]]:
    """Cherche des CSV parallèles dans `dest`, écrit all.jsonl. Retourne (n, colmap, raw_files)."""
    pd = __import__("pandas")
    raw_files = sorted(p.name for p in dest.iterdir() if p.is_file() and p.suffix.lower() == ".csv")
    records: list[dict] = []
    colmap_used: dict = {}
    for csv_name in raw_files:
        try:
            df = pd.read_csv(dest / csv_name)
        except Exception as e:  # noqa: BLE001
            print(f"  [csv] lecture impossible {csv_name} : {e}")
            continue
        colmap = detect_lang_columns(df.columns)
        if "ewe_Latn" not in colmap or len(colmap) < 2:
            print(f"  [csv] {csv_name} : pas de paire éwé↔(en|fr) (colonnes={list(df.columns)})")
            continue
        colmap_used = colmap
        for row in df.to_dict(orient="records"):
            rec = row_to_translation(row, colmap)
            if rec is not None:
                records.append(rec)
    n = write_jsonl(dest / "all.jsonl", records) if records else 0
    return n, colmap_used, raw_files


def process_source(src: dict) -> dict:
    api = get_kaggle_api()
    name, slug = src["name"], src["slug"]
    dest = RAW_TRANSLATION / name
    dest.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {name}  (Kaggle {slug}) ===")
    try:
        api.dataset_download_files(slug, path=str(dest), unzip=True, quiet=False)
    except Exception as e:  # noqa: BLE001
        print(f"  [ÉCHEC] téléchargement : {e}")
        return {"name": name, "status": "échec", "error": str(e)}

    n, colmap, raw_files = normalize_csvs(dest)
    if n:
        print(f"  -> all.jsonl normalisé ({n} lignes parallèles)")
    else:
        print("  (aucune paire parallèle normalisée — fichiers bruts conservés)")

    write_manifest(
        dest,
        source=name,
        provenance="kaggle",
        slug=slug,
        license="voir page Kaggle",
        format="jsonl + bruts" if n else "bruts (monolingue/QA ?)",
        columns_detected=colmap,
        raw_files=raw_files,
        rows={"all": n},
        rows_total=n,
        notes=src.get("notes", ""),
    )
    return {"name": name, "status": "ok", "rows": n}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", help="nom(s) séparés par des virgules")
    args = ap.parse_args()

    if args.list:
        for s in SOURCES:
            print(f"  {s['name']:<36} {s['slug']}")
        return

    selected = SOURCES
    if args.only:
        wanted = {x.strip() for x in args.only.split(",")}
        selected = [s for s in SOURCES if s["name"] in wanted]
        if not selected:
            raise SystemExit(f"Aucune source ne correspond à --only {args.only!r}")

    results = [process_source(s) for s in selected]

    print("\n===== RÉSUMÉ =====")
    for r in results:
        mark = "✅" if r["status"] == "ok" else "❌"
        info = f"{r['rows']} lignes parallèles" if r["status"] == "ok" else r.get("error", "")
        print(f"  {mark} {r['name']:<36} {info}")


if __name__ == "__main__":
    main()
