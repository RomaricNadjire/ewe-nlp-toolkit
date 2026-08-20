#!/usr/bin/env python3
"""Scrape les corpus éwé↔anglais du site peterlin.pl (dictionnaire, paragraphes, phrases).

Refactorisation de `notebooks/exploration/scraping-Ewe-English.ipynb`. Écrit dans
`data/raw/translation/peterlin/` :
  - ewe_english_dictionary.csv   (mots : ewe, english)
  - ewe_dataset.csv              (paragraphes : Subject, Ewe, English, Source_URL)
  - ewe_phrases.csv              (phrases : Category, Ewe, English)
  - all.jsonl                    (paires normalisées dictionnaire + phrases)
  - manifest.json

Dépendances : requests, beautifulsoup4 (déjà présents).

Exemple :
    python src/scraping/scrape_peterlin.py
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

# Réutilise les utilitaires de src/download/_common.py (manifest, jsonl, chemins).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "download"))
from _common import RAW_TRANSLATION, require, write_jsonl, write_manifest  # noqa: E402

BASE = "http://www.peterlin.pl/ewe/"
DICT_URL = BASE + "words.html"
PHRASES_URL = BASE + "phrases.html"
PARAGRAPH_URLS = [
    BASE + "djoubogbe.html",
    BASE + "names-weekdays.html",
    BASE + "child-naming.html",
    BASE + "marriage.html",
    BASE + "funeral.html",
    BASE + "grandfather.html",
]
HEADERS = {"User-Agent": "EweCorpusBot/1.0 (recherche académique)"}
DELAY = 1.0  # politesse entre requêtes


def _get_soup(url: str):
    requests = require("requests")
    bs4 = require("bs4", "beautifulsoup4")
    res = requests.get(url, headers=HEADERS, timeout=60)
    res.raise_for_status()
    res.encoding = "utf-8"
    return bs4.BeautifulSoup(res.text, "html.parser")


def scrape_dictionary() -> list[dict]:
    soup = _get_soup(DICT_URL)
    rows = []
    for block in soup.find_all("dl", class_="letter"):
        for ewe, eng in zip(block.find_all("dt"), block.find_all("dd")):
            e = ewe.get_text(strip=True)
            g = eng.get_text(strip=True)
            if e and g:
                rows.append({"ewe": e, "english": g})
    return rows


def scrape_paragraphs() -> list[dict]:
    rows = []
    for url in PARAGRAPH_URLS:
        soup = _get_soup(url)
        for h2 in soup.find_all("h2"):
            subject = h2.get_text(strip=True)
            ewe_text, english_text, mode = "", "", None
            current = h2.find_next_sibling()
            while current and current.name != "h2":
                if current.name == "h3":
                    title = current.get_text(strip=True).lower()
                    if "ewe version" in title:
                        mode = "ewe"
                    elif "english translation" in title:
                        mode = "english"
                elif current.name == "p":
                    classes = current.get("class", [])
                    if "text" in classes or "poem" in classes:
                        text = current.get_text(separator="\n", strip=True)
                        if mode == "ewe":
                            ewe_text += text + "\n"
                        elif mode == "english":
                            english_text += text + "\n"
                current = current.find_next_sibling()
            ewe_text, english_text = ewe_text.strip(), english_text.strip()
            if ewe_text or english_text:
                rows.append({"Subject": subject, "Ewe": ewe_text,
                             "English": english_text, "Source_URL": url})
        time.sleep(DELAY)
    return rows


def scrape_phrases() -> list[dict]:
    soup = _get_soup(PHRASES_URL)
    rows = []
    for section in soup.find_all("dl", class_="letter"):
        header = section.find_previous("h3", class_="letter-head")
        category = header.get_text(strip=True) if header else "Unknown"
        for dt, dd in zip(section.find_all("dt"), section.find_all("dd")):
            e = dt.get_text(" ", strip=True)
            g = dd.get_text(" ", strip=True)
            if e and g:
                rows.append({"Category": category, "Ewe": e, "English": g})
    return rows


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # LF + UTF-8 sans BOM : sortie propre et stable (diffs minimaux).
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-paragraphs", action="store_true", help="ne pas scraper les paragraphes")
    args = ap.parse_args()

    dest = RAW_TRANSLATION / "peterlin"
    print("== Dictionnaire ==")
    dico = scrape_dictionary()
    _write_csv(dest / "ewe_english_dictionary.csv", dico, ["ewe", "english"])
    print(f"  {len(dico)} entrées -> ewe_english_dictionary.csv")

    print("== Phrases ==")
    phrases = scrape_phrases()
    _write_csv(dest / "ewe_phrases.csv", phrases, ["Category", "Ewe", "English"])
    print(f"  {len(phrases)} phrases -> ewe_phrases.csv")

    paras = []
    if not args.skip_paragraphs:
        print("== Paragraphes ==")
        paras = scrape_paragraphs()
        _write_csv(dest / "ewe_dataset.csv", paras,
                   ["Subject", "Ewe", "English", "Source_URL"])
        print(f"  {len(paras)} sections -> ewe_dataset.csv")

    # Paires normalisées (niveau mot/phrase) pour la traduction.
    records = []
    for r in dico:
        records.append({"translation": {"ewe_Latn": r["ewe"], "eng_Latn": r["english"]}})
    for r in phrases:
        records.append({"translation": {"ewe_Latn": r["Ewe"], "eng_Latn": r["English"]}})
    n = write_jsonl(dest / "all.jsonl", records)
    print(f"== all.jsonl : {n} paires normalisées (dictionnaire + phrases) ==")

    write_manifest(
        dest,
        source="peterlin",
        provenance="scraping",
        base_url=BASE,
        license="usage recherche (peterlin.pl)",
        format="csv (3) + jsonl normalisé",
        rows={"dictionary": len(dico), "phrases": len(phrases),
              "paragraphs": len(paras), "all_pairs": n},
        rows_total=n,
        notes="Refactor de notebooks/exploration/scraping-Ewe-English.ipynb. "
              "Paragraphes = textes longs (doc-level), non inclus dans all.jsonl.",
    )
    print("manifest.json écrit.")


if __name__ == "__main__":
    main()
