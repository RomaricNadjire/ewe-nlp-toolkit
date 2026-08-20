"""
add_fr_en_to_nllb.py
====================
Ajoute les paires anglais↔français des CSV ``Français-Anglais/`` au dataset NLLB
existant (``nllb_translation/``), au format JSONL :

    {"translation": {"eng_Latn": "...", "fra_Latn": "..."}}

Le dataset éwé existant est PRÉSERVÉ : seules les anciennes lignes fr-en
(eng_Latn + fra_Latn, sans ewe_Latn) sont régénérées à partir des CSV, ce qui
rend le script idempotent (ré-exécutable sans créer de doublons).

Sources :
    Français-Anglais/dataset_test_2000_fr_eng.csv     -> split test
    Français-Anglais/dataset_train_reste_fr_eng.csv   -> split train + validation

Colonnes attendues : ``English words/sentences`` , ``French words/sentences``.

Une seule ligne BILINGUE est écrite par paire : les deux directions
(en→fr et fr→en) sont générées au moment du fine-tuning, pas ici.

Usage :
    python3 add_fr_en_to_nllb.py
    python3 add_fr_en_to_nllb.py --dry-run
    python3 add_fr_en_to_nllb.py --val-size 5000 --seed 42
    python3 add_fr_en_to_nllb.py --src "Français-Anglais" --dst nllb_translation
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import unicodedata
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("[ERREUR] pandas requis : pip install pandas", file=sys.stderr)
    sys.exit(1)


EN_COL = "English words/sentences"
FR_COL = "French words/sentences"
EN_CODE = "eng_Latn"
FR_CODE = "fra_Latn"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean(text: str) -> str:
    """Normalisation NFKC + strip (même convention que restructure_translation.py)."""
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize("NFKC", text).strip()


def is_valid(en: str, fr: str, min_chars: int = 1) -> bool:
    """Filtre les paires vides (ou plus courtes que min_chars)."""
    return len(en) >= min_chars and len(fr) >= min_chars


def load_csv_pairs(path: Path, min_chars: int = 1) -> list[tuple[str, str]]:
    """Charge un CSV (en,fr) et retourne une liste de tuples (anglais, français) nettoyés."""
    if not path.exists():
        print(f"  [ERREUR] Introuvable : {path}", file=sys.stderr)
        return []
    # dtype=str + na_filter=False : ne jamais convertir "NA"/"None"/"" en NaN.
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)
    if EN_COL not in df.columns or FR_COL not in df.columns:
        print(f"  [ERREUR] Colonnes '{EN_COL}'/'{FR_COL}' introuvables dans {path}")
        print(f"           Colonnes disponibles : {list(df.columns)}")
        return []
    pairs: list[tuple[str, str]] = []
    for en_raw, fr_raw in zip(df[EN_COL], df[FR_COL]):
        en, fr = clean(en_raw), clean(fr_raw)
        if is_valid(en, fr, min_chars):
            pairs.append((en, fr))
    return pairs


def dedup_pairs(pairs: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], int]:
    """Déduplique sur la paire exacte (en, fr) en préservant l'ordre."""
    seen: set[tuple[str, str]] = set()
    unique: list[tuple[str, str]] = []
    for p in pairs:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique, len(pairs) - len(unique)


def make_line(en: str, fr: str) -> dict:
    """Construit une ligne JSONL NLLB bilingue en-fr."""
    return {"translation": {EN_CODE: en, FR_CODE: fr}}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def is_ewe_line(obj: dict) -> bool:
    """Vrai si l'exemple contient de l'éwé (à préserver)."""
    return "ewe_Latn" in obj.get("translation", {})


def write_jsonl(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def count_kinds(lines: list[dict]) -> tuple[int, int]:
    """Retourne (nb_lignes_ewe, nb_lignes_fr_en)."""
    ewe = sum(1 for o in lines if is_ewe_line(o))
    return ewe, len(lines) - ewe


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ajoute les paires fr↔en au dataset NLLB (préserve l'éwé)."
    )
    parser.add_argument("--src-root", default=".", help="Racine du projet (défaut : .)")
    parser.add_argument(
        "--src", default="Français-Anglais",
        help="Dossier contenant les CSV fr-en.",
    )
    parser.add_argument(
        "--dst", default="nllb_translation",
        help="Dossier du dataset NLLB à mettre à jour.",
    )
    parser.add_argument(
        "--train-csv", default="dataset_train_reste_fr_eng.csv",
        help="Nom du CSV d'entraînement (split train + validation).",
    )
    parser.add_argument(
        "--test-csv", default="dataset_test_2000_fr_eng.csv",
        help="Nom du CSV de test.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Graine pour le shuffle.")
    parser.add_argument(
        "--val-size", type=int, default=5000,
        help="Nombre de paires prélevées du train pour la validation.",
    )
    parser.add_argument(
        "--min-chars", type=int, default=1,
        help="Longueur minimale (en caractères) de chaque côté.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Simulation : affiche les stats sans écrire.",
    )
    args = parser.parse_args()

    src_root = Path(args.src_root)
    src_dir = src_root / args.src
    dst_dir = src_root / args.dst
    train_csv = src_dir / args.train_csv
    test_csv = src_dir / args.test_csv

    mode = "[DRY-RUN] " if args.dry_run else ""
    print(f"{mode}Ajout des paires fr↔en au dataset NLLB")
    print(f"Source : {src_dir.resolve()}")
    print(f"Dest.  : {dst_dir.resolve()}")
    print("=" * 64)

    # -----------------------------------------------------------------------
    # 1. Chargement des CSV
    # -----------------------------------------------------------------------
    print("\n[1/5] Chargement des CSV")
    test_raw = load_csv_pairs(test_csv, args.min_chars)
    train_raw = load_csv_pairs(train_csv, args.min_chars)
    if not test_raw or not train_raw:
        print("[ERREUR] CSV vide ou introuvable — abandon.", file=sys.stderr)
        return 1
    print(f"  {args.test_csv:<40} {len(test_raw):>7} paires")
    print(f"  {args.train_csv:<40} {len(train_raw):>7} paires")

    # -----------------------------------------------------------------------
    # 2. Déduplication interne
    # -----------------------------------------------------------------------
    print("\n[2/5] Déduplication interne (paire exacte en+fr)")
    test_pairs, dup_test = dedup_pairs(test_raw)
    train_pairs, dup_train = dedup_pairs(train_raw)
    print(f"  test  : {len(test_raw):>7} -> {len(test_pairs):>7} ({dup_test} doublons)")
    print(f"  train : {len(train_raw):>7} -> {len(train_pairs):>7} ({dup_train} doublons)")

    # -----------------------------------------------------------------------
    # 3. Anti-fuite : retirer du pool train les paires présentes dans le test
    # -----------------------------------------------------------------------
    print("\n[3/5] Anti-fuite + split validation")
    test_keys = set(test_pairs)
    train_pool = [p for p in train_pairs if p not in test_keys]
    leaked = len(train_pairs) - len(train_pool)
    print(f"  Paires train aussi dans test, retirées : {leaked}")

    rng = random.Random(args.seed)
    rng.shuffle(train_pool)
    val_size = min(args.val_size, max(0, len(train_pool) - 1))
    val_pairs = train_pool[:val_size]
    final_train_pairs = train_pool[val_size:]
    print(f"  validation : {len(val_pairs)} paires (prélevées du train)")
    print(f"  train      : {len(final_train_pairs)} paires")

    # Construction des lignes JSONL bilingues fr-en
    new_by_split = {
        "train": [make_line(en, fr) for en, fr in final_train_pairs],
        "validation": [make_line(en, fr) for en, fr in val_pairs],
        "test": [make_line(en, fr) for en, fr in test_pairs],
    }

    # -----------------------------------------------------------------------
    # 4. Fusion avec l'existant (préserver l'éwé, régénérer le fr-en)
    # -----------------------------------------------------------------------
    print("\n[4/5] Fusion avec le dataset existant (éwé préservé)")
    outputs: dict[str, list[dict]] = {}
    for split, new_lines in new_by_split.items():
        existing = load_jsonl(dst_dir / f"{split}.jsonl")
        ewe_before, fren_before = count_kinds(existing)
        ewe_lines = [o for o in existing if is_ewe_line(o)]
        combined = ewe_lines + new_lines
        outputs[split] = combined
        print(
            f"  {split:<11} avant: {len(existing):>6} "
            f"(éwé={ewe_before}, fr-en={fren_before})  ->  "
            f"après: {len(combined):>6} (éwé={len(ewe_lines)}, fr-en={len(new_lines)})"
        )

    # -----------------------------------------------------------------------
    # 5. Vérifications anti-fuite globales
    # -----------------------------------------------------------------------
    print("\n[5/5] Vérifications")
    train_keys = {p for p in final_train_pairs}
    val_keys = {p for p in val_pairs}
    inter_tv = test_keys & (train_keys | val_keys)
    inter_train_val = train_keys & val_keys
    print(f"  test ∩ (train ∪ val) : {len(inter_tv)} (attendu 0)")
    print(f"  train ∩ val          : {len(inter_train_val)} (attendu 0)")
    if inter_tv or inter_train_val:
        print("  [ERREUR] Fuite détectée entre les splits — abandon.", file=sys.stderr)
        return 1

    if args.dry_run:
        print("\n[DRY-RUN] Aucun fichier écrit.")
        return 0

    for split, lines in outputs.items():
        write_jsonl(dst_dir / f"{split}.jsonl", lines)
        print(f"  ✔ écrit {split}.jsonl ({len(lines)} lignes)")

    print("\nTerminé.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
