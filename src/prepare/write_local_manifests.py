#!/usr/bin/env python3
"""Génère un `manifest.json` pour chaque source locale pré-existante.

Les scripts de `src/download/` et `src/scraping/` écrivent déjà leur manifeste
au moment du téléchargement. Ce script complète les sources déjà présentes
*avant* la mise en place de l'outillage (OPUS exporté à la main, CSV Kaggle,
pivot fr-en, audio Bible), en calculant les comptes réels depuis les fichiers.

Idempotent : peut être relancé sans risque (réécrit les manifestes).

Exemple :
    python src/prepare/write_local_manifests.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "download"))
from _common import RAW_ASR, RAW_TRANSLATION, write_manifest  # noqa: E402


def _count_csv_rows(path: Path) -> int:
    """Compte les lignes de données d'un CSV (gère les champs multi-lignes)."""
    with path.open(encoding="utf-8", errors="replace", newline="") as f:
        return max(sum(1 for _ in csv.reader(f)) - 1, 0)


def _count_lines(path: Path) -> int:
    with path.open(encoding="utf-8", errors="replace") as f:
        return sum(1 for line in f if line.strip())


def manifest_francais_anglais() -> None:
    d = RAW_TRANSLATION / "francais_anglais"
    train = _count_csv_rows(d / "dataset_train_reste_fr_eng.csv")
    test = _count_csv_rows(d / "dataset_test_2000_fr_eng.csv")
    write_manifest(
        d, source="francais_anglais", provenance="fourni (pivot)",
        language="eng_Latn↔fra_Latn (PAS d'éwé)",
        license="à vérifier (corpus en-fr fourni)",
        format="csv",
        rows={"train_reste": train, "test_2000": test},
        rows_total=train + test,
        notes="Pivot anglais-français pour le multilingue. Ne contient pas d'éwé.",
    )
    print(f"francais_anglais : train={train}, test={test}")


def manifest_kaggle_ewe_english() -> None:
    d = RAW_TRANSLATION / "kaggle_ewe_english"
    n = _count_csv_rows(d / "EWE_ENGLISH.csv")
    write_manifest(
        d, source="kaggle_ewe_english", provenance="Kaggle",
        language="ewe_Latn↔eng_Latn",
        license="voir page Kaggle source", format="csv",
        rows_total=n,
        notes="Paires éwé-anglais (colonnes EWE/ENGLISH).",
    )
    print(f"kaggle_ewe_english : {n}")


def manifest_opus(name: str, lang_file: str, version: str) -> None:
    d = RAW_TRANSLATION / name
    n = _count_lines(d / lang_file)
    write_manifest(
        d, source=name, provenance="OPUS (export manuel moses)",
        language="ewe_Latn↔eng_Latn",
        license="voir fichier LICENSE du paquet OPUS",
        format="moses (.ee/.en) + csv + xml",
        version=version, rows_total=n,
        notes=f"Bitexte aligné OPUS. {n} segments (lignes du fichier .ee).",
    )
    print(f"{name} : {n}")


def manifest_bible_ewe_asr() -> None:
    d = RAW_ASR / "bible_ewe"
    rows = {}
    total = 0
    for split in ("train", "validation", "test"):
        meta = d / split / "metadata.jsonl"
        c = _count_lines(meta) if meta.exists() else 0
        rows[split] = c
        total += c
    write_manifest(
        d, source="bible_ewe", provenance="fourni (audio Bible éwé)",
        language="ewe_Latn (audio + transcription)",
        license="à vérifier", format="audiofolder (metadata.jsonl + wav)",
        rows=rows, rows_total=total,
        notes="Corpus ASR Bible éwé. metadata.jsonl : {file_name, sentence}.",
    )
    print(f"bible_ewe (ASR) : {rows}")


def main() -> None:
    manifest_francais_anglais()
    manifest_kaggle_ewe_english()
    manifest_opus("opus_bible_ee_en", "bible-uedin.ee-en.ee", "bible-uedin")
    manifest_opus("opus_qed_ee_en", "QED.ee-en.ee", "QED")
    manifest_bible_ewe_asr()
    print("\nManifestes locaux générés.")


if __name__ == "__main__":
    main()
