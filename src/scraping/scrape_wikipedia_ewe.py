#!/usr/bin/env python3
"""Scrape le Wikipédia en éwé (ee.wikipedia.org) — corpus monolingue éwé.

Utilise l'API MediaWiki (action=query, generator=allpages, prop=extracts) pour
récupérer le texte brut de tous les articles de l'espace principal.

Sortie dans `data/raw/monolingual/ewe_wikipedia/` :
  - articles.jsonl   ({pageid, title, text})        — un article par ligne
  - sentences.jsonl  ({text})                        — phrases éwé segmentées
  - manifest.json

Utile pour : modèles de langue, TTS, augmentation monolingue. Licence CC-BY-SA 4.0
(attribution Wikipédia requise). Dépendance : requests.

Exemples :
    python src/scraping/scrape_wikipedia_ewe.py --list
    python src/scraping/scrape_wikipedia_ewe.py --limit 200
    python src/scraping/scrape_wikipedia_ewe.py            # tout (~1300 articles)
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "download"))
from _common import REPO_ROOT, require, write_jsonl, write_manifest  # noqa: E402

API_URL = "https://ee.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "EweCorpusBot/1.0 (https://github.com/RomaricNadjire/TP_Traduction; recherche académique, corpus monolingue éwé)"}
DEST = REPO_ROOT / "data" / "raw" / "monolingual" / "ewe_wikipedia"
DELAY = 1.0            # politesse entre requêtes
MAX_RETRIES = 5       # tentatives sur 429/503/maxlag
MIN_CHARS = 20        # ignorer les ébauches/textes trop courts
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def _api_get(session, params: dict) -> dict:
    """Appel API avec backoff sur 429/503 (Retry-After) et erreurs maxlag."""
    base = {"format": "json", "maxlag": "5"}
    base.update(params)
    for attempt in range(MAX_RETRIES):
        res = session.get(API_URL, params=base, headers=HEADERS, timeout=60)
        if res.status_code in (429, 503):
            wait = int(res.headers.get("Retry-After", 0)) or (2 ** attempt)
            print(f"  [throttle] {res.status_code} -> pause {wait}s")
            time.sleep(wait)
            continue
        res.raise_for_status()
        data = res.json()
        # Erreur maxlag : serveur surchargé, on attend et on réessaie.
        if isinstance(data, dict) and data.get("error", {}).get("code") == "maxlag":
            wait = int(res.headers.get("Retry-After", 0)) or (2 ** attempt)
            print(f"  [maxlag] pause {wait}s")
            time.sleep(wait)
            continue
        return data
    raise RuntimeError(f"Échec API après {MAX_RETRIES} tentatives (throttling).")


def site_stats(session) -> dict:
    data = _api_get(session, {"action": "query", "meta": "siteinfo", "siprop": "statistics"})
    return data["query"]["statistics"]


def fetch_all_pageids(session) -> list[int]:
    """Liste tous les pageid de l'espace principal (requête peu coûteuse)."""
    ids: list[int] = []
    params = {
        "action": "query", "list": "allpages", "apnamespace": 0,
        "aplimit": "max", "apfilterredir": "nonredirects",
    }
    while True:
        data = _api_get(session, params)
        for p in data.get("query", {}).get("allpages", []):
            ids.append(p["pageid"])
        cont = data.get("continue")
        if not cont:
            break
        params.update(cont)
        time.sleep(DELAY)
    return ids


def fetch_extracts(session, pageids: list[int]) -> list[dict]:
    """Récupère le texte brut d'un lot de pages (gère `excontinue`)."""
    result: dict[int, dict] = {}
    params = {
        "action": "query", "prop": "extracts", "explaintext": 1,
        "exlimit": "max", "pageids": "|".join(str(p) for p in pageids),
    }
    while True:
        data = _api_get(session, params)
        for page in data.get("query", {}).get("pages", {}).values():
            text = (page.get("extract") or "").strip()
            if text:
                pid = page.get("pageid")
                result[pid] = {"pageid": pid, "title": page.get("title"), "text": text}
        cont = data.get("continue")
        if not cont:
            break
        params.update(cont)
        time.sleep(DELAY)
    return list(result.values())


def iter_articles(session, limit: int | None = None):
    """Itère les articles (espace 0) avec leur texte brut.

    Phase 1 : liste de tous les pageid (peu coûteux). Phase 2 : extraits par
    lots de 20 (limite de l'API extracts), ce qui évite les bugs de pagination
    du couple generator+extracts.
    """
    ids = fetch_all_pageids(session)
    if limit:
        ids = ids[:limit]
    for i in range(0, len(ids), 20):
        for art in fetch_extracts(session, ids[i:i + 20]):
            if len(art["text"]) >= MIN_CHARS:
                yield art
        time.sleep(DELAY)


def split_sentences(text: str) -> list[str]:
    out = []
    for chunk in _SENT_SPLIT.split(text):
        s = chunk.strip()
        if len(s) >= MIN_CHARS:
            out.append(s)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None, help="nombre maximal d'articles")
    ap.add_argument("--list", action="store_true", help="afficher les statistiques du site et quitter")
    args = ap.parse_args()

    requests = require("requests")
    session = requests.Session()

    stats = site_stats(session)
    print(f"ee.wikipedia : {stats.get('articles')} articles, {stats.get('pages')} pages.")
    if args.list:
        return

    articles, all_sentences = [], []
    for art in iter_articles(session, limit=args.limit):
        articles.append(art)
        for s in split_sentences(art["text"]):
            all_sentences.append({"text": s})
        if len(articles) % 100 == 0:
            print(f"  ... {len(articles)} articles récupérés")

    n_art = write_jsonl(DEST / "articles.jsonl", articles)
    n_sent = write_jsonl(DEST / "sentences.jsonl", all_sentences)
    print(f"Écrit : {n_art} articles, {n_sent} phrases.")

    write_manifest(
        DEST,
        source="ewe_wikipedia",
        provenance="scraping",
        base_url=API_URL,
        license="CC-BY-SA 4.0 (attribution Wikipédia requise)",
        language="ewe_Latn (monolingue)",
        format="jsonl (articles + phrases)",
        rows={"articles": n_art, "sentences": n_sent},
        rows_total=n_sent,
        notes="Corpus monolingue éwé pour LM/TTS/augmentation. Texte brut via "
              "API MediaWiki prop=extracts explaintext.",
    )
    print("manifest.json écrit.")


if __name__ == "__main__":
    main()
