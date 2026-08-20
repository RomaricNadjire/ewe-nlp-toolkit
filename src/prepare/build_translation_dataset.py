#!/usr/bin/env python3
"""P2 — Nettoyage, déduplication et fusion des corpus de traduction éwé.

Lit toutes les sources de `data/raw/translation/`, les normalise en paires
multilingues (éwé/anglais/français), nettoie et filtre (filtrage par score de
similarité pour les corpus minés), déduplique, exclut tout ce qui est dans
validation/test, puis écrit :

    data/processed/translation/train.jsonl         sources PROPRES (non minées)
    data/processed/translation/train_mined.jsonl   minés filtrés (optionnel)
    data/processed/translation/build_report.json   statistiques par source

`validation.jsonl` / `test.jsonl` existants sont conservés tels quels (held-out) ;
le train est dédupliqué contre eux pour éviter toute fuite. L'ancien `train.jsonl`
est déplacé vers `data/interim/translation_prev/` et réinjecté comme source propre
(aucune donnée n'est perdue).

Exemples :
    python src/prepare/build_translation_dataset.py
    python src/prepare/build_translation_dataset.py --mined-min-sim 1.07
    python src/prepare/build_translation_dataset.py --no-mined
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW = REPO_ROOT / "data" / "raw" / "translation"
PROC = REPO_ROOT / "data" / "processed" / "translation"
INTERIM = REPO_ROOT / "data" / "interim" / "translation_prev"

LANGS = ("ewe_Latn", "eng_Latn", "fra_Latn")

# Caractères turcs absents de l'orthographe éwé (détectent le bruit OPUS/QED).
SUSPECT_EWE = set("ıİşğŞĞ")
# Plages Unicode d'autres écritures : si l'« éwé » en contient, c'est du bruit.
_BAD_RANGES = (
    (0x0400, 0x04FF),  # cyrillique
    (0x0590, 0x05FF),  # hébreu
    (0x0600, 0x06FF),  # arabe
    (0x3040, 0x30FF),  # hiragana/katakana
    (0x4E00, 0x9FFF),  # CJK
    (0xAC00, 0xD7A3),  # hangul
)
_WS = re.compile(r"\s+")


# ---------------------------------------------------------------- normalisation
def norm_text(s) -> str:
    """NFC, suppression du BOM, espaces compactés."""
    if not s:
        return ""
    s = unicodedata.normalize("NFC", str(s)).replace("\ufeff", "")
    return _WS.sub(" ", s).strip()


def bad_script(s: str) -> bool:
    """Vrai si `s` contient des caractères d'une autre écriture (bruit)."""
    for ch in s[:200]:
        if ch in SUSPECT_EWE:
            return True
        o = ord(ch)
        for lo, hi in _BAD_RANGES:
            if lo <= o <= hi:
                return True
    return False


def letter_ratio(s: str) -> float:
    """Proportion de lettres parmi les caractères non-espaces."""
    chars = [c for c in s if not c.isspace()]
    if not chars:
        return 0.0
    return sum(c.isalpha() for c in chars) / len(chars)


# ---------------------------------------------------------------------- lecteurs
def read_jsonl(src_dir: Path):
    for path in sorted(src_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def read_csv_cols(src_dir: Path, csv_name: str, cols: dict[str, str]):
    paths = list(src_dir.glob(csv_name))
    if not paths:
        return
    with paths[0].open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            trans = {code: row.get(col, "") for code, col in cols.items()}
            yield {"translation": trans}


def read_yvicherita(src_dir: Path):
    """Export PHPMyAdmin : table `eweenglishsentence` avec ee_sentence / en_sentence."""
    paths = list(src_dir.glob("*.json"))
    if not paths:
        return
    try:
        blocks = json.loads(paths[0].read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    for blk in blocks if isinstance(blocks, list) else []:
        if not (isinstance(blk, dict) and isinstance(blk.get("data"), list)):
            continue
        for rec in blk["data"]:
            ewe = rec.get("ee_sentence") or rec.get("ewe") or ""
            eng = (rec.get("en_sentence") or rec.get("english")
                   or rec.get("eng_sentence") or rec.get("translation") or "")
            if ewe and eng:
                yield {"translation": {"ewe_Latn": ewe, "eng_Latn": eng}}


# ------------------------------------------------------------------- catalogue
def _src(name, **kw):
    kw["name"] = name
    return kw


SOURCES = [
    # --- sources propres (non minées) ---
    _src("adaboubvincent_eng_fra_ewe", kind="jsonl"),
    _src("ghananlp_ewe_english_parallel", kind="jsonl"),
    _src("ghananlp_navigation_ewe", kind="jsonl"),
    _src("peterlin", kind="jsonl"),
    _src("opus_tatoeba_ee_en", kind="jsonl"),
    _src("kaggle_tchaye59_ewe_english", kind="jsonl"),
    _src("kaggle_ewe_english", kind="csv",
         csv_name="EWE_ENGLISH.csv", cols={"ewe_Latn": "EWE", "eng_Latn": "ENGLISH"}),
    _src("opus_bible_ee_en", kind="csv",
         csv_name="bible_ee_en.csv", cols={"ewe_Latn": "ewe", "eng_Latn": "english"}),
    _src("kaggle_yvicherita_ewe_corpus", kind="yvicherita"),
    _src("francais_anglais", kind="csv",
         csv_name="dataset_train_reste_fr_eng.csv",
         cols={"eng_Latn": "English words/sentences", "fra_Latn": "French words/sentences"}),
    # --- corpus minés (filtrés par similarité) ---
    _src("ghananlp_ewe_english_4m", kind="jsonl", mined=True),
    _src("michsethowusu_ewe_french", kind="jsonl", mined=True),
    # --- exclus (avec justification) ---
    _src("opus_qed_ee_en", exclude="désaligné (turc dans la colonne éwé)"),
    _src("opus_qed_ee_fr", exclude="désaligné (turc dans la colonne éwé)"),
    _src("glk360_fr_ewe", exclude="éwé monolingue (pas de paires parallèles)"),
    _src("kaggle_ghana_maternal_health_ewe", exclude="pas de paires parallèles (batches bruts)"),
]


def iter_source(src: dict):
    d = RAW / src["name"]
    kind = src.get("kind")
    if kind == "jsonl":
        yield from read_jsonl(d)
    elif kind == "csv":
        yield from read_csv_cols(d, src["csv_name"], src["cols"])
    elif kind == "yvicherita":
        yield from read_yvicherita(d)


# -------------------------------------------------------------------- nettoyage
def clean_pairs(rec: dict, mined: bool, min_sim: float, max_len: int):
    """Génère des tuples (langueA, texteA, langueB, texteB, statut) pour chaque paire
    de langues présente (éwé/anglais/français). statut='ok' ou raison de rejet."""
    trans = rec.get("translation") or {}
    vals = {}
    for lang in LANGS:
        v = norm_text(trans.get(lang, ""))
        if v:
            vals[lang] = v

    if mined:
        sim = rec.get("similarity")
        if sim is None or (isinstance(sim, float) and sim != sim):  # None ou NaN
            yield (None, None, None, None, "sim_absente")
            return
        if float(sim) < min_sim:
            yield (None, None, None, None, "sim_basse")
            return

    present = [l for l in LANGS if l in vals]
    if len(present) < 2:
        yield (None, None, None, None, "pas_de_paire")
        return

    lo_b, hi_b = (0.4, 2.5) if mined else (0.2, 5.0)
    for i in range(len(present)):
        for j in range(i + 1, len(present)):
            a, b = present[i], present[j]
            ta, tb = vals[a], vals[b]
            if bad_script(ta) or bad_script(tb):
                yield (None, None, None, None, "ecriture_etrangere")
                continue
            if ta == tb:
                yield (None, None, None, None, "copie")
                continue
            na, nb = len(ta), len(tb)
            if na > max_len or nb > max_len:
                yield (None, None, None, None, "trop_long")
                continue
            ratio = na / nb if nb else 99
            if not (lo_b <= ratio <= hi_b):
                yield (None, None, None, None, "ratio")
                continue
            if mined and (letter_ratio(ta) < 0.5 or letter_ratio(tb) < 0.5):
                yield (None, None, None, None, "trop_de_symboles")
                continue
            yield (a, ta, b, tb, "ok")


def pair_key(la: str, ta: str, lb: str, tb: str) -> bytes:
    """Clé canonique d'une paire, indépendante de l'ordre des deux langues."""
    x = f"{la}\x1f{ta.lower()}"
    y = f"{lb}\x1f{tb.lower()}"
    lo, hi = (x, y) if x <= y else (y, x)
    h = hashlib.blake2b(digest_size=16)
    h.update(lo.encode())
    h.update(b"\x1e")
    h.update(hi.encode())
    return h.digest()


def load_eval_keys() -> set[bytes]:
    keys: set[bytes] = set()
    for split in ("validation.jsonl", "test.jsonl"):
        p = PROC / split
        if not p.exists():
            continue
        with p.open(encoding="utf-8") as f:
            for line in f:
                try:
                    trans = json.loads(line).get("translation", {})
                except json.JSONDecodeError:
                    continue
                present = [(l, norm_text(trans.get(l, ""))) for l in LANGS]
                present = [(l, v) for l, v in present if v]
                for i in range(len(present)):
                    for j in range(i + 1, len(present)):
                        la, ta = present[i]
                        lb, tb = present[j]
                        keys.add(pair_key(la, ta, lb, tb))
    return keys


# ------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mined-min-sim", type=float, default=1.06,
                    help="score de similarité minimal pour garder une paire minée (défaut 1.06)")
    ap.add_argument("--max-len", type=int, default=1000, help="longueur max (caractères) par côté")
    ap.add_argument("--no-mined", action="store_true", help="ignore complètement les corpus minés")
    args = ap.parse_args()

    PROC.mkdir(parents=True, exist_ok=True)
    INTERIM.mkdir(parents=True, exist_ok=True)

    # Préserver l'ancien train comme source propre (déplacé, pas perdu).
    prev_train = INTERIM / "train_prev.jsonl"
    if (PROC / "train.jsonl").exists():
        shutil.move(str(PROC / "train.jsonl"), str(prev_train))
        print(f"Ancien train -> {prev_train.relative_to(REPO_ROOT)}")

    print("Chargement des clés validation/test (held-out)...")
    eval_keys = load_eval_keys()
    print(f"  {len(eval_keys)} clés held-out")

    seen: set[bytes] = set()
    report: dict[str, dict] = {}

    out_clean = (PROC / "train.jsonl").open("w", encoding="utf-8")
    out_mined = (PROC / "train_mined.jsonl").open("w", encoding="utf-8")
    n_clean = n_mined = 0

    # 1) ancienne base propre (réinjectée en premier)
    work = []
    if prev_train.exists():
        work.append(_src("prev_processed", kind="jsonl_path", path=prev_train))
    work += [s for s in SOURCES if not s.get("exclude") and not (args.no_mined and s.get("mined"))]

    for src in work:
        name = src["name"]
        mined = bool(src.get("mined"))
        stats = {"lu": 0, "ok": 0, "doublon": 0, "held_out": 0}
        drops: dict[str, int] = {}

        if src.get("kind") == "jsonl_path":
            records = _read_one_jsonl(src["path"])
        else:
            records = iter_source(src)

        for rec in records:
            stats["lu"] += 1
            for la, ta, lb, tb, status in clean_pairs(rec, mined, args.mined_min_sim, args.max_len):
                if status != "ok":
                    drops[status] = drops.get(status, 0) + 1
                    continue
                k = pair_key(la, ta, lb, tb)
                if k in eval_keys:
                    stats["held_out"] += 1
                    continue
                if k in seen:
                    stats["doublon"] += 1
                    continue
                seen.add(k)
                stats["ok"] += 1
                line = json.dumps({"translation": {la: ta, lb: tb}}, ensure_ascii=False)
                if mined:
                    out_mined.write(line + "\n"); n_mined += 1
                else:
                    out_clean.write(line + "\n"); n_clean += 1

        stats["rejets"] = drops
        report[name] = stats
        print(f"  {name:<34} lu={stats['lu']:>8}  gardé={stats['ok']:>8}  "
              f"doublons={stats['doublon']:>7}  held-out={stats['held_out']:>5}")

    out_clean.close()
    out_mined.close()
    if n_mined == 0:
        (PROC / "train_mined.jsonl").unlink(missing_ok=True)

    summary = {
        "train_clean": n_clean,
        "train_mined": n_mined,
        "held_out_keys": len(eval_keys),
        "mined_min_sim": args.mined_min_sim,
        "max_len": args.max_len,
        "sources_exclues": {s["name"]: s["exclude"] for s in SOURCES if s.get("exclude")},
        "par_source": report,
    }
    (PROC / "build_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\n===== RÉSUMÉ =====")
    print(f"  train.jsonl (propre) : {n_clean} paires")
    print(f"  train_mined.jsonl    : {n_mined} paires (sim >= {args.mined_min_sim})")
    print(f"  held-out (val+test)  : {len(eval_keys)} clés exclues du train")
    print(f"  rapport              : data/processed/translation/build_report.json")


def _read_one_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


if __name__ == "__main__":
    main()
