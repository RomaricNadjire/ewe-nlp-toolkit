#!/usr/bin/env python3
"""Télécharge des corpus parallèles depuis OPUS (https://opus.nlpl.eu).

Cible en priorité les paires **éwé-français** (ee-fr) absentes du jeu local, plus
quelques compléments ee-en. Utilise l'API OPUS pour résoudre le lien « moses » le
plus récent, puis normalise au format projet
    {"translation": {"ewe_Latn": ..., "eng_Latn"|"fra_Latn": ...}}
dans `data/raw/translation/opus_<corpus>_<paire>/` avec un `manifest.json`.

Dépendances : requests (déjà présent). Aucun besoin d'opustools.

Exemples :
    python src/download/download_opus.py --list
    python src/download/download_opus.py --only nllb_ee_fr
    python src/download/download_opus.py --skip-mined
"""
from __future__ import annotations

import argparse
import io
import zipfile

from _common import RAW_TRANSLATION, require, write_jsonl, write_manifest

OPUS_API = "https://opus.nlpl.eu/opusapi/"

# Code langue OPUS (ISO) -> code NLLB.
LANG = {"ee": "ewe_Latn", "en": "eng_Latn", "fr": "fra_Latn"}

# (corpus OPUS, langue source, langue cible). OPUS ordonne alphabétiquement (ee<en<fr).
SOURCES = [
    {"name": "opus_nllb_ee_fr", "corpus": "NLLB", "src": "ee", "tgt": "fr",
     "mined": True, "notes": "NLLB miné ee-fr (~1 039 385) — à filtrer en P2."},
    {"name": "opus_bible_ee_fr", "corpus": "bible-uedin", "src": "ee", "tgt": "fr",
     "notes": "Bible alignée ee-fr (~8 004)."},
    {"name": "opus_qed_ee_fr", "corpus": "QED", "src": "ee", "tgt": "fr",
     "notes": "Sous-titres éducatifs ee-fr (~166, CC-BY)."},
    {"name": "opus_tatoeba_ee_en", "corpus": "Tatoeba", "src": "ee", "tgt": "en",
     "notes": "Phrases communautaires ee-en (CC-BY)."},
]


def resolve_moses_url(corpus: str, src: str, tgt: str) -> str | None:
    """Interroge l'API OPUS pour le lien d'archive moses le plus récent."""
    requests = require("requests")
    params = {"corpus": corpus, "source": src, "target": tgt,
              "preprocessing": "moses", "version": "latest"}
    try:
        r = requests.get(OPUS_API, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
    except Exception as e:  # noqa: BLE001
        print(f"  [API] échec ({e})")
        return None
    for entry in data.get("corpora", []):
        url = entry.get("url", "")
        if url.endswith(".zip"):
            return url
    return None


def process_source(src: dict, limit: int | None) -> dict:
    requests = require("requests")
    name, corpus = src["name"], src["corpus"]
    s, t = src["src"], src["tgt"]
    dest = RAW_TRANSLATION / name
    print(f"\n=== {name}  (OPUS {corpus} {s}-{t}) ===")

    url = resolve_moses_url(corpus, s, t)
    if not url:
        print("  [ÉCHEC] aucune archive moses trouvée.")
        return {"name": name, "status": "échec", "error": "pas d'archive moses"}
    print(f"  archive : {url}")

    try:
        resp = requests.get(url, timeout=600)
        resp.raise_for_status()
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
    except Exception as e:  # noqa: BLE001
        print(f"  [ÉCHEC] téléchargement/zip : {e}")
        return {"name": name, "status": "échec", "error": str(e)}

    # Fichiers parallèles : <corpus>.<s>-<t>.<s> et .<t>
    names = zf.namelist()
    f_src = next((n for n in names if n.endswith(f".{s}")), None)
    f_tgt = next((n for n in names if n.endswith(f".{t}")), None)
    if not (f_src and f_tgt):
        print(f"  [ÉCHEC] fichiers .{s}/.{t} introuvables dans {names}")
        return {"name": name, "status": "échec", "error": "fichiers parallèles absents"}

    with zf.open(f_src) as fa, zf.open(f_tgt) as fb:
        src_lines = io.TextIOWrapper(fa, encoding="utf-8")
        tgt_lines = io.TextIOWrapper(fb, encoding="utf-8")

        def _records():
            code_s, code_t = LANG[s], LANG[t]
            for i, (a, b) in enumerate(zip(src_lines, tgt_lines)):
                if limit is not None and i >= limit:
                    break
                a, b = a.strip(), b.strip()
                if a and b:
                    yield {"translation": {code_s: a, code_t: b}}

        out = dest / "all.jsonl"
        n = write_jsonl(out, _records())

    print(f"  -> {out.relative_to(RAW_TRANSLATION.parent.parent)}  ({n} lignes)")
    write_manifest(
        dest,
        source=name,
        provenance="opus",
        corpus=corpus,
        pair=f"{s}-{t}",
        url=url,
        license="conditions OPUS / corpus amont",
        format="jsonl ({'translation': {code: texte}})",
        rows={"all": n},
        rows_total=n,
        mined=bool(src.get("mined")),
        limited_sample=limit,
        notes=src.get("notes", ""),
    )
    return {"name": name, "status": "ok", "rows": n}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", help="nom(s) séparés par des virgules")
    ap.add_argument("--limit", type=int, default=None, help="n max de lignes (test)")
    ap.add_argument("--skip-mined", action="store_true")
    args = ap.parse_args()

    if args.list:
        for s in SOURCES:
            tag = "  [miné]" if s.get("mined") else ""
            print(f"  {s['name']:<24} OPUS {s['corpus']} {s['src']}-{s['tgt']}{tag}")
        return

    selected = SOURCES
    if args.only:
        wanted = {x.strip() for x in args.only.split(",")}
        selected = [s for s in SOURCES if s["name"] in wanted]
        if not selected:
            raise SystemExit(f"Aucune source ne correspond à --only {args.only!r}")
    if args.skip_mined:
        selected = [s for s in selected if not s.get("mined")]

    results = [process_source(s, args.limit) for s in selected]

    print("\n===== RÉSUMÉ =====")
    for r in results:
        mark = "✅" if r["status"] == "ok" else "❌"
        info = f"{r['rows']} lignes" if r["status"] == "ok" else r.get("error", "")
        print(f"  {mark} {r['name']:<24} {info}")


if __name__ == "__main__":
    main()
