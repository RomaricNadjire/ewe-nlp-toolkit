#!/usr/bin/env python3
"""Phase 0 — Pré-filtrage local des paires minées (heuristiques + langid).

Objectif : réduire le bruit GROSSIER des corpus minés AVANT le re-scoring
sémantique SONAR (Phase 1, sur GPU Kaggle). Ce script ne juge PAS l'alignement
sémantique (réservé à SONAR) ; il ne fait que rejeter les paires manifestement
inexploitables :

  - longueurs incohérentes (ratio de mots / de caractères trop élevé) ;
  - balisage résiduel (HTML, URL, entités, wiki) ;
  - copies exactes / champs vides / contenu non-textuel ;
  - langid (fasttext ``lid.176``) :
      * côté cible (en/fr) : exiger une détection positive (conf >= seuil) ;
      * côté éwé : rejeter si confidemment classé comme anglais/français
        (fuite de la langue cible dans le champ éwé).

`lid.176` ne couvre pas l'éwé : on ne peut donc pas confirmer POSITIVEMENT
l'éwé localement (réservé à `lid218e` sur Kaggle) ; on l'utilise uniquement
pour détecter les fuites de langue européenne.

Entrées :
    data/raw/translation/ghananlp_ewe_english_4m/*.jsonl   (éwé-anglais)
    data/raw/translation/michsethowusu_ewe_french/*.jsonl  (éwé-français)

Sorties :
    data/interim/mined_candidates/ewe_en.jsonl
    data/interim/mined_candidates/ewe_fr.jsonl
    data/interim/mined_candidates/phase0_report.json

Exemples :
    .venv/bin/python src/prepare/filter_mined_phase0.py
    .venv/bin/python src/prepare/filter_mined_phase0.py --limit 100000   # test
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW = REPO_ROOT / "data" / "raw" / "translation"
OUT = REPO_ROOT / "data" / "interim" / "mined_candidates"
MODEL = REPO_ROOT / "data" / "interim" / "models" / "lid.176.bin"

SOURCES = (
    # (dossier source, langue cible, fichier de sortie)
    ("ghananlp_ewe_english_4m", "eng_Latn", "ewe_en.jsonl"),
    ("michsethowusu_ewe_french", "fra_Latn", "ewe_fr.jsonl"),
)
# Code lid.176 attendu pour chaque langue cible.
TGT_LID = {"eng_Latn": "en", "fra_Latn": "fr"}
# Côté éwé : rejet si confidemment détecté comme l'une de ces langues (fuite).
EWE_LEAK_LANGS = {"en", "fr"}

_WS = re.compile(r"\s+")
TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://|www\.")
ENTITY_RE = re.compile(r"&[a-zA-Z#0-9]{2,8};")
WIKI_RE = re.compile(r"\[\[|\]\]|\{\{|\}\}|\|\|")


def norm_text(s) -> str:
    """NFC, suppression du BOM, espaces compactés."""
    if not s:
        return ""
    s = unicodedata.normalize("NFC", str(s)).replace("\ufeff", "")
    return _WS.sub(" ", s).strip()


def letter_ratio(s: str) -> float:
    """Fraction de caractères alphabétiques parmi les non-espaces."""
    chars = [c for c in s if not c.isspace()]
    if not chars:
        return 0.0
    return sum(c.isalpha() for c in chars) / len(chars)


def heuristic_reject(ewe: str, tgt: str, args) -> str | None:
    """Renvoie la raison du rejet (str) ou None si la paire passe."""
    if not ewe or not tgt:
        return "vide"
    if ewe == tgt:
        return "copie"
    we, wt = ewe.split(), tgt.split()
    nwe, nwt = len(we), len(wt)
    if nwe == 0 or nwt == 0:
        return "vide"
    wr = max(nwe, nwt) / min(nwe, nwt)
    if wr > args.max_word_ratio:
        return "ratio_mots"
    cr = max(len(ewe), len(tgt)) / max(1, min(len(ewe), len(tgt)))
    if cr > args.max_char_ratio:
        return "ratio_char"
    if TAG_RE.search(ewe) or TAG_RE.search(tgt):
        return "balise"
    if URL_RE.search(ewe) or URL_RE.search(tgt):
        return "url"
    if ENTITY_RE.search(ewe) or ENTITY_RE.search(tgt):
        return "entite"
    if WIKI_RE.search(ewe) or WIKI_RE.search(tgt):
        return "wiki"
    if letter_ratio(ewe) < args.min_letter_ratio or letter_ratio(tgt) < args.min_letter_ratio:
        return "peu_lettres"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-word-ratio", type=float, default=2.0)
    ap.add_argument("--max-char-ratio", type=float, default=3.0)
    ap.add_argument("--min-letter-ratio", type=float, default=0.5)
    ap.add_argument("--tgt-min-conf", type=float, default=0.50,
                    help="conf. minimale pour valider la langue cible (en/fr)")
    ap.add_argument("--ewe-reject-conf", type=float, default=0.55,
                    help="conf. au-delà de laquelle on rejette une fuite en/fr côté éwé")
    ap.add_argument("--model", type=Path, default=MODEL)
    ap.add_argument("--limit", type=int, default=0,
                    help="ne traiter que N lignes par source (0 = tout) pour tester")
    ap.add_argument("--progress-every", type=int, default=200_000)
    args = ap.parse_args()

    if not args.model.exists():
        print(f"ERREUR : modèle langid introuvable : {args.model}", file=sys.stderr)
        return 1

    import fasttext
    print(f"Chargement du modèle langid : {args.model}")
    lid = fasttext.load_model(str(args.model))

    def predict(text: str):
        lab, conf = lid.predict(text.replace("\n", " "), k=1)
        return lab[0].replace("__label__", ""), float(conf[0])

    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = {"params": vars(args).copy(), "sources": {}}
    report["params"]["model"] = str(args.model)

    grand_total = grand_kept = 0
    for folder, tgt_lang, out_name in SOURCES:
        src_dir = RAW / folder
        files = sorted(src_dir.glob("*.jsonl"))
        if not files:
            print(f"  (aucun fichier dans {src_dir}, ignoré)")
            continue
        tgt_code = TGT_LID[tgt_lang]
        out_path = OUT / out_name
        reasons = Counter()
        total = kept = 0
        print(f"\n=== {folder} -> {out_name} (cible {tgt_lang}/{tgt_code}) ===")
        with out_path.open("w", encoding="utf-8") as fout:
            for fp in files:
                with fp.open(encoding="utf-8") as fin:
                    for line in fin:
                        if args.limit and total >= args.limit:
                            break
                        total += 1
                        if total % args.progress_every == 0:
                            print(f"  …{total:,} lus, {kept:,} retenus", flush=True)
                        line = line.strip()
                        if not line:
                            reasons["json_vide"] += 1
                            continue
                        try:
                            obj = json.loads(line)
                            tr = obj["translation"]
                            ewe = norm_text(tr.get("ewe_Latn"))
                            tgt = norm_text(tr.get(tgt_lang))
                        except (json.JSONDecodeError, KeyError, AttributeError):
                            reasons["json_invalide"] += 1
                            continue

                        r = heuristic_reject(ewe, tgt, args)
                        if r:
                            reasons[r] += 1
                            continue

                        # langid côté cible : détection positive requise
                        tlab, tconf = predict(tgt)
                        if tlab != tgt_code or tconf < args.tgt_min_conf:
                            reasons["cible_langid"] += 1
                            continue
                        # langid côté éwé : rejet si fuite en/fr confiante
                        elab, econf = predict(ewe)
                        if elab in EWE_LEAK_LANGS and econf >= args.ewe_reject_conf:
                            reasons["ewe_fuite"] += 1
                            continue

                        rec = {"translation": {"ewe_Latn": ewe, tgt_lang: tgt}}
                        if "similarity" in obj:
                            rec["similarity"] = obj["similarity"]
                        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        kept += 1
                if args.limit and total >= args.limit:
                    break

        rate = (kept / total * 100) if total else 0.0
        print(f"  total={total:,}  retenus={kept:,} ({rate:.1f}%)")
        print("  rejets :", dict(reasons.most_common()))
        report["sources"][folder] = {
            "target": tgt_lang,
            "output": str(out_path.relative_to(REPO_ROOT)),
            "total": total,
            "kept": kept,
            "kept_pct": round(rate, 2),
            "rejections": dict(reasons),
        }
        grand_total += total
        grand_kept += kept

    report["grand_total"] = grand_total
    report["grand_kept"] = grand_kept
    report["grand_kept_pct"] = round((grand_kept / grand_total * 100) if grand_total else 0.0, 2)
    rep_path = OUT / "phase0_report.json"
    rep_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=== TOTAL : {grand_kept:,}/{grand_total:,} retenus "
          f"({report['grand_kept_pct']}%) ===")
    print(f"Rapport : {rep_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
