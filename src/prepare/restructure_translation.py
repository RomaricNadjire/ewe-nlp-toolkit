"""
restructure_translation.py
==========================
Fusionne toutes les sources de paires éwé↔anglais et éwé↔français
au format JSONL NLLB, prêt pour le fine-tuning de :
    facebook/nllb-200-distilled-600M

Sources :
    opus/bible_ee_en.csv            éwé ↔ anglais  (Bible)
    opus/qed_ee_en.csv              éwé ↔ anglais  (sous-titres)
    kaggle/EWE_ENGLISH.csv          éwé ↔ anglais
    kaggle/ewe_english_train.csv    éwé ↔ anglais
    peterlin/ewe_dataset.csv        éwé ↔ anglais
    peterlin/ewe_english_dictionary.csv  éwé ↔ anglais (mots)
    peterlin/ewe_phrases.csv        éwé ↔ anglais  (phrases)
    zenodo/French_to_ewe_dataset.xlsx    français ↔ éwé

Format de sortie (NLLB) :
    {"translation": {"ewe_Latn": "...", "eng_Latn": "..."}}
    {"translation": {"ewe_Latn": "...", "fra_Latn": "..."}}

Usage :
    python3 restructure_translation.py
    python3 restructure_translation.py --dst nllb_translation --seed 42
    python3 restructure_translation.py --dry-run
    python3 restructure_translation.py --move-originals
"""

import argparse
import json
import random
import shutil
import sys
import unicodedata
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("[ERREUR] pandas requis : pip install pandas openpyxl", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Configuration des sources
# ---------------------------------------------------------------------------
# Chaque source : (chemin, col_ewe, col_target, lang_code_target, read_kwargs)
SOURCES_EN = [
    (
        "data/raw/translation/opus_bible_ee_en/bible_ee_en.csv",
        "ewe", "english", "eng_Latn",
        {},
    ),
    (
        "data/raw/translation/opus_qed_ee_en/qed_ee_en.csv",
        "ewe", "english", "eng_Latn",
        {},
    ),
    (
        "data/raw/translation/kaggle_ewe_english/EWE_ENGLISH.csv",
        "EWE", "ENGLISH", "eng_Latn",
        {"index_col": 0},
    ),
    (
        "data/raw/translation/kaggle_ewe_english/ewe_english_train.csv",
        "Ewe", "English", "eng_Latn",
        {},
    ),
    (
        "data/raw/translation/peterlin/ewe_dataset.csv",
        "Ewe", "English", "eng_Latn",
        {},
    ),
    (
        "data/raw/translation/peterlin/ewe_english_dictionary.csv",
        "ewe", "english", "eng_Latn",
        {},
    ),
    (
        "data/raw/translation/peterlin/ewe_phrases.csv",
        "Ewe", "English", "eng_Latn",
        {},
    ),
]

ZENODO_XLSX = "data/raw/translation/zenodo/French_to_ewe_dataset.xlsx"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean(text: str) -> str:
    """Normalisation NFKC + strip."""
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize("NFKC", text).strip()


def is_valid(ewe: str, target: str, min_chars: int = 2) -> bool:
    """Filtre les paires vides ou trop courtes."""
    return len(ewe) >= min_chars and len(target) >= min_chars


def load_csv_source(
    path: Path,
    col_ewe: str,
    col_target: str,
    lang_code: str,
    read_kwargs: dict,
) -> list[dict]:
    """Charge un CSV et retourne une liste de paires NLLB."""
    if not path.exists():
        print(f"  [SKIP] Introuvable : {path}")
        return []
    try:
        df = pd.read_csv(path, **read_kwargs)
    except Exception as exc:
        print(f"  [ERREUR] Lecture {path} : {exc}")
        return []

    # Normaliser les noms de colonnes en minuscules pour la comparaison
    col_map = {c: c for c in df.columns}
    ewe_col = col_map.get(col_ewe)
    tgt_col = col_map.get(col_target)

    if ewe_col is None or tgt_col is None:
        print(f"  [ERREUR] Colonnes '{col_ewe}'/'{col_target}' introuvables dans {path}")
        print(f"           Colonnes disponibles : {list(df.columns)}")
        return []

    pairs = []
    for _, row in df.iterrows():
        ewe = clean(str(row[ewe_col]))
        target = clean(str(row[tgt_col]))
        if is_valid(ewe, target):
            pairs.append({"translation": {"ewe_Latn": ewe, lang_code: target}})
    return pairs


def load_xlsx_source(path: Path) -> list[dict]:
    """
    Charge le xlsx zenodo (français↔éwé).
    Structure multi-feuilles :
      - Feuille 'French'   : (index, texte_français)
      - Feuille 'Ewe'      : (index, texte_éwé)
    Les deux feuilles sont alignées par l'index numérique de la 1ère colonne.
    """
    if not path.exists():
        print(f"  [SKIP] Introuvable : {path}")
        return []
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        print("  [ERREUR] openpyxl requis : pip install openpyxl")
        return []

    try:
        xl = pd.ExcelFile(path, engine="openpyxl")
        sheets = xl.sheet_names
        print(f"  Feuilles détectées : {sheets}")

        # Cherche les feuilles French et Ewe (insensible à la casse)
        sheets_lower = {s.lower(): s for s in sheets}
        sheet_fr  = sheets_lower.get("french")
        sheet_ewe = sheets_lower.get("ewe")

        if sheet_fr is None or sheet_ewe is None:
            print(f"  [ERREUR] Feuilles 'French' et/ou 'Ewe' introuvables dans {path}")
            return []

        df_fr  = xl.parse(sheet_fr,  header=0, index_col=0)
        df_ewe = xl.parse(sheet_ewe, header=0, index_col=0)

        # La colonne de texte est la première (et unique) colonne de chaque feuille
        col_fr_name  = df_fr.columns[0]
        col_ewe_name = df_ewe.columns[0]
        print(f"  Colonnes utilisées : éwé='{col_ewe_name}', français='{col_fr_name}'")

        # Jointure sur l'index
        merged = df_ewe[[col_ewe_name]].join(df_fr[[col_fr_name]], how="inner")

    except Exception as exc:
        print(f"  [ERREUR] Lecture {path} : {exc}")
        return []

    pairs = []
    for _, row in merged.iterrows():
        ewe = clean(str(row[col_ewe_name]))
        fr  = clean(str(row[col_fr_name]))
        if is_valid(ewe, fr):
            pairs.append({"translation": {"ewe_Latn": ewe, "fra_Latn": fr}})
    return pairs


def deduplicate(pairs: list[dict]) -> tuple[list[dict], int]:
    """Déduplique sur la clé (ewe_Latn, valeur_cible)."""
    seen: set[tuple] = set()
    unique = []
    for p in pairs:
        t = p["translation"]
        ewe = t["ewe_Latn"]
        target_val = next(v for k, v in t.items() if k != "ewe_Latn")
        key = (ewe, target_val)
        if key not in seen:
            seen.add(key)
            unique.append(p)
    removed = len(pairs) - len(unique)
    return unique, removed


def split_dataset(
    pairs: list[dict],
    seed: int,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Shuffle + split 80/10/10."""
    rng = random.Random(seed)
    shuffled = pairs.copy()
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    train = shuffled[:n_train]
    val   = shuffled[n_train:n_train + n_val]
    test  = shuffled[n_train + n_val:]
    return train, val, test


def write_jsonl(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def move_originals(src_root: Path, dst_root: Path, folders: list[str]) -> None:
    """Déplace les dossiers sources vers dst_root."""
    print("\n" + "=" * 60)
    print("DÉPLACEMENT DES DOSSIERS ORIGINAUX")
    print("=" * 60)
    for folder in folders:
        src = src_root / folder
        dst = dst_root / folder
        if not src.exists():
            print(f"  [SKIP] {src} introuvable")
            continue
        if dst.exists():
            print(f"  [SKIP] {dst} existe déjà — ignoré pour éviter l'écrasement")
            continue
        shutil.move(str(src), str(dst))
        print(f"  ✔ {src.name}/ → {dst.parent}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Restructure les données de traduction éwé au format NLLB JSONL."
    )
    parser.add_argument("--src-root", default=".", help="Racine du projet (défaut : .)")
    parser.add_argument("--dst", default="data/processed/translation", help="Dossier de sortie")
    parser.add_argument("--seed", type=int, default=42, help="Graine pour le shuffle")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Simulation : affiche les stats sans écrire.",
    )
    parser.add_argument(
        "--move-originals", action="store_true",
        help="Déplace opus/, kaggle/, peterlin/, zenodo/ vers ../datasets/ après génération.",
    )
    args = parser.parse_args()

    src_root = Path(args.src_root)
    dst_root = Path(args.dst)

    mode_label = "[DRY-RUN] " if args.dry_run else ""
    print(f"{mode_label}Restructuration des données de traduction éwé")
    print(f"Source : {src_root.resolve()}")
    print(f"Dest.  : {dst_root.resolve()}")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # 1. Chargement de toutes les sources
    # -----------------------------------------------------------------------
    all_pairs_en: list[dict] = []
    all_pairs_fr: list[dict] = []

    print("\n[1/5] Chargement des sources éwé ↔ anglais")
    for path_str, col_ewe, col_tgt, lang_code, kwargs in SOURCES_EN:
        path = src_root / path_str
        pairs = load_csv_source(path, col_ewe, col_tgt, lang_code, kwargs)
        print(f"  {path_str:<45} {len(pairs):>6} paires")
        all_pairs_en.extend(pairs)

    print(f"\n  Sous-total brut éwé↔anglais : {len(all_pairs_en)}")

    print("\n[2/5] Chargement de la source éwé ↔ français (zenodo)")
    xlsx_path = src_root / ZENODO_XLSX
    pairs_fr = load_xlsx_source(xlsx_path)
    print(f"  {ZENODO_XLSX:<45} {len(pairs_fr):>6} paires")
    all_pairs_fr.extend(pairs_fr)
    print(f"\n  Sous-total brut éwé↔français : {len(all_pairs_fr)}")

    # -----------------------------------------------------------------------
    # 2. Déduplication séparée par paire de langues
    # -----------------------------------------------------------------------
    print("\n[3/5] Déduplication")
    pairs_en_dedup, removed_en = deduplicate(all_pairs_en)
    pairs_fr_dedup, removed_fr = deduplicate(all_pairs_fr)
    print(f"  éwé↔anglais : {len(all_pairs_en)} → {len(pairs_en_dedup)} ({removed_en} doublons supprimés)")
    print(f"  éwé↔français: {len(all_pairs_fr)} → {len(pairs_fr_dedup)} ({removed_fr} doublons supprimés)")

    # -----------------------------------------------------------------------
    # 3. Split 80/10/10 par paire de langues
    # -----------------------------------------------------------------------
    print("\n[4/5] Split 80/10/10 (seed={})".format(args.seed))
    train_en, val_en, test_en = split_dataset(pairs_en_dedup, args.seed)
    train_fr, val_fr, test_fr = split_dataset(pairs_fr_dedup, args.seed)

    train = train_en + train_fr
    val   = val_en   + val_fr
    test  = test_en  + test_fr

    # Mélanger les deux langues dans chaque split
    rng = random.Random(args.seed)
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    total = len(train) + len(val) + len(test)
    print(f"  train      : {len(train):>6}")
    print(f"  validation : {len(val):>6}")
    print(f"  test       : {len(test):>6}")
    print(f"  TOTAL      : {total:>6}")

    # -----------------------------------------------------------------------
    # 4. Écriture JSONL
    # -----------------------------------------------------------------------
    print("\n[5/5] Écriture JSONL")
    if args.dry_run:
        print("  [DRY-RUN] Aucun fichier écrit.")
    else:
        write_jsonl(dst_root / "train.jsonl",      train)
        write_jsonl(dst_root / "validation.jsonl", val)
        write_jsonl(dst_root / "test.jsonl",       test)
        print(f"  ✔ {dst_root}/train.jsonl      ({len(train)} lignes)")
        print(f"  ✔ {dst_root}/validation.jsonl ({len(val)} lignes)")
        print(f"  ✔ {dst_root}/test.jsonl       ({len(test)} lignes)")

    # -----------------------------------------------------------------------
    # 5. (Optionnel) Déplacement des originaux
    # -----------------------------------------------------------------------
    if args.move_originals and not args.dry_run:
        datasets_dir = src_root.resolve().parent / "datasets"
        if not datasets_dir.exists():
            print(f"\n[ERREUR] Dossier cible introuvable : {datasets_dir}", file=sys.stderr)
        else:
            move_originals(
                src_root=src_root,
                dst_root=datasets_dir,
                folders=["opus", "kaggle", "peterlin", "zenodo"],
            )

    # -----------------------------------------------------------------------
    # Résumé final
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RÉSUMÉ")
    print("=" * 60)
    print(f"  Paires éwé↔anglais (après dédup) : {len(pairs_en_dedup)}")
    print(f"  Paires éwé↔français (après dédup): {len(pairs_fr_dedup)}")
    print(f"  Total                             : {total}")
    if not args.dry_run:
        print(f"\n  Dataset prêt dans : {dst_root.resolve()}")
        print("\n  Chargement HuggingFace :")
        print("    from datasets import load_dataset")
        print(f'    ds = load_dataset("json", data_files={{')
        print(f'        "train":      "{args.dst}/train.jsonl",')
        print(f'        "validation": "{args.dst}/validation.jsonl",')
        print(f'        "test":       "{args.dst}/test.jsonl",')
        print(f'    }})')
        if not args.move_originals:
            print("\n  Pour déplacer les dossiers sources vers ../datasets/ :")
            print("    python3 restructure_translation.py --move-originals")


if __name__ == "__main__":
    main()
