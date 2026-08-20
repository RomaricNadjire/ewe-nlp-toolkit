#!/usr/bin/env python3
"""Télécharge des corpus de traduction éwé depuis le Hugging Face Hub.

Chaque source est normalisée au format projet
    {"translation": {"ewe_Latn": ..., "eng_Latn"|"fra_Latn": ...}}
et écrite dans `data/raw/translation/<nom>/<split>.jsonl` avec un `manifest.json`.

Les colonnes de langue sont détectées automatiquement (cf. `_common.detect_lang_columns`),
ce qui rend le script robuste aux variations de schéma entre datasets.

Exemples :
    python src/download/download_hf_translation.py --list
    python src/download/download_hf_translation.py --only adaboubvincent_eng_fra_ewe
    python src/download/download_hf_translation.py --limit 5000        # échantillon de test
    python src/download/download_hf_translation.py --skip-mined        # ignore les corpus minés
"""
from __future__ import annotations

import argparse

from _common import (
    RAW_TRANSLATION,
    detect_lang_columns,
    require,
    row_to_translation,
    write_jsonl,
    write_manifest,
)

# Catalogue des sources HF de traduction (cf. data/CATALOG.md §1.2).
# `eval_only=True` => à réserver à l'évaluation (jamais dans le train).
# `mined=True`     => corpus bruité, à filtrer par similarité en P2.
SOURCES = [
    {
        "name": "adaboubvincent_eng_fra_ewe",
        "hf_id": "adaboubvincent/translation-eng-fra-ewe",
        "license": "à vérifier",
        # Format long : une ligne = (src, src_lang, tgt, target_lang).
        "long_format": {
            "src": "src", "src_lang": "src_lang",
            "tgt": "tgt", "tgt_lang": "target_lang",
            "lang_map": {"ewe": "ewe_Latn", "ee": "ewe_Latn",
                         "eng": "eng_Latn", "en": "eng_Latn",
                         "fra": "fra_Latn", "fr": "fra_Latn"},
        },
        "notes": "Parallèle 3 voies éwé/anglais/français (~311k, format long, doublons inverses).",
    },
    {
        "name": "ghananlp_ewe_english_4m",
        "hf_id": "ghananlpcommunity/english-ewe-sentence-pairs-4m",
        "license": "à vérifier (GhanaNLP)",
        "mined": True,
        "notes": "~4.4M paires ee-en MINÉES (bruitées) — colonne similarity conservée.",
    },
    {
        "name": "michsethowusu_ewe_french",
        "hf_id": "michsethowusu/ewe-french_sentence-pairs",
        "license": "à vérifier",
        "mined": True,
        "notes": "~1-10M paires ee-fr MINÉES (bruitées).",
    },
    {
        "name": "ghananlp_ewe_english_parallel",
        "hf_id": "Ghana-NLP/EWE_ENGLISH_PARALLEL_TEXT",
        "license": "à vérifier",
        # Schéma id/text/label : text = anglais, label = éwé.
        "colmap": {"eng_Latn": "text", "ewe_Latn": "label"},
        "notes": "Texte parallèle ee-en Ghana-NLP (text=en, label=ee).",
    },
    {
        "name": "ghananlp_navigation_ewe",
        "hf_id": "ghananlpcommunity/navigation-corpus-ewe",
        "license": "à vérifier",
        # english = anglais, translated = éwé.
        "colmap": {"eng_Latn": "english", "ewe_Latn": "translated"},
        "notes": "Domaine navigation/itinéraires ee-en (translated=ee).",
    },
    {
        "name": "glk360_fr_ewe",
        "hf_id": "glk360/fr-ewe-corpus",
        "license": "domaine public (DUDH)",
        "notes": "~60 paires ee-fr (Déclaration universelle des droits de l'homme).",
    },
    {
        "name": "mafand_en_ewe",
        "hf_id": "masakhane/mafand",
        "config": "en-ewe",
        "license": "CC-BY-NC",
        "eval_only": True,
        "notes": "MAFAND presse ee-en (train 2026 / dev 1414 / test 1563). CC-BY-NC => éval.",
    },
]


def process_source(src: dict, limit: int | None) -> dict:
    """Charge une source HF, normalise et écrit JSONL + manifest. Retourne un résumé."""
    datasets = require("datasets")
    name, hf_id = src["name"], src["hf_id"]
    config = src.get("config")
    dest = RAW_TRANSLATION / name

    print(f"\n=== {name}  ({hf_id}{' / ' + config if config else ''}) ===")
    try:
        ds = datasets.load_dataset(hf_id, config) if config else datasets.load_dataset(hf_id)
    except Exception as e:  # noqa: BLE001 — on veut continuer sur les autres sources
        print(f"  [ÉCHEC] chargement impossible : {e}")
        return {"name": name, "status": "échec", "error": str(e)}

    # Normalise en DatasetDict (certains datasets renvoient un Dataset simple).
    if not hasattr(ds, "items"):
        ds = {"train": ds}

    split_counts: dict[str, int] = {}
    colmap_used: dict = {}
    long_spec = src.get("long_format")
    fixed_colmap = src.get("colmap")
    for split, dset in ds.items():
        cols = list(dset.column_names)

        if long_spec:
            colmap_used = {"format": "long", **{k: v for k, v in long_spec.items() if k != "lang_map"}}

            def _records(dset=dset):
                it = dset if limit is None else dset.select(range(min(limit, len(dset))))
                lm = long_spec["lang_map"]
                for row in it:
                    sc = lm.get(str(row.get(long_spec["src_lang"])).strip().lower())
                    tc = lm.get(str(row.get(long_spec["tgt_lang"])).strip().lower())
                    if not sc or not tc:
                        continue
                    trans = {}
                    sv = str(row.get(long_spec["src"]) or "").strip()
                    tv = str(row.get(long_spec["tgt"]) or "").strip()
                    if sv:
                        trans[sc] = sv
                    if tv:
                        trans[tc] = tv
                    if trans.get("ewe_Latn") and len(trans) >= 2:
                        yield {"translation": trans}
        else:
            colmap = fixed_colmap or detect_lang_columns(cols)
            if "ewe_Latn" not in colmap:
                print(f"  [IGNORÉ] split '{split}' : aucune colonne éwé détectée (colonnes={cols})")
                continue
            missing = [c for c in colmap.values() if c not in cols]
            if missing:
                print(f"  [IGNORÉ] split '{split}' : colonnes {missing} absentes (présentes={cols})")
                continue
            colmap_used = colmap

            def _records(dset=dset, colmap=colmap):
                it = dset if limit is None else dset.select(range(min(limit, len(dset))))
                for row in it:
                    rec = row_to_translation(row, colmap)
                    if rec is not None:
                        yield rec

        out = dest / f"{split}.jsonl"
        n = write_jsonl(out, _records())
        split_counts[split] = n
        print(f"  {split:<12} -> {out.relative_to(RAW_TRANSLATION.parent.parent)}  ({n} lignes)")

    if not split_counts:
        print("  [ÉCHEC] aucune colonne éwé exploitable.")
        return {"name": name, "status": "échec", "error": "pas de colonne éwé"}

    write_manifest(
        dest,
        source=name,
        provenance="huggingface",
        hf_id=hf_id,
        config=config,
        license=src.get("license", "à vérifier"),
        format="jsonl ({'translation': {code: texte}})",
        columns_detected=colmap_used,
        rows=split_counts,
        rows_total=sum(split_counts.values()),
        mined=bool(src.get("mined")),
        eval_only=bool(src.get("eval_only")),
        limited_sample=limit,
        notes=src.get("notes", ""),
    )
    return {"name": name, "status": "ok", "rows": sum(split_counts.values())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="liste les sources et quitte")
    ap.add_argument("--only", help="nom(s) de source séparés par des virgules")
    ap.add_argument("--limit", type=int, default=None, help="n max de lignes par split (test)")
    ap.add_argument("--skip-mined", action="store_true", help="ignore les corpus minés")
    ap.add_argument("--skip-eval-only", action="store_true", help="ignore les corpus éval. seule")
    args = ap.parse_args()

    if args.list:
        for s in SOURCES:
            tags = []
            if s.get("mined"):
                tags.append("miné")
            if s.get("eval_only"):
                tags.append("éval-seule")
            print(f"  {s['name']:<32} {s['hf_id']}{'  [' + ','.join(tags) + ']' if tags else ''}")
        return

    selected = SOURCES
    if args.only:
        wanted = {x.strip() for x in args.only.split(",")}
        selected = [s for s in SOURCES if s["name"] in wanted]
        if not selected:
            raise SystemExit(f"Aucune source ne correspond à --only {args.only!r}")
    if args.skip_mined:
        selected = [s for s in selected if not s.get("mined")]
    if args.skip_eval_only:
        selected = [s for s in selected if not s.get("eval_only")]

    results = [process_source(s, args.limit) for s in selected]

    print("\n===== RÉSUMÉ =====")
    for r in results:
        if r["status"] == "ok":
            print(f"  ✅ {r['name']:<32} {r['rows']} lignes")
        else:
            print(f"  ❌ {r['name']:<32} {r.get('error', '')}")


if __name__ == "__main__":
    main()
