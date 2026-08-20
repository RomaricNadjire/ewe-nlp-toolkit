#!/usr/bin/env python3
"""Audit « par source » des scores d'alignement SONAR (Phase 1).

Contexte
--------
La cascade Stage-1 a échoué parce que le filtrage SONAR appliquait un **seuil
global** (le percentile 5 de la distribution des paires *propres*), bien trop
permissif : 66 % (ee-en) et 85 % (ee-fr) des candidats l'ont franchi. Ce script
remplace cette logique aveugle par un **diagnostic source par source** :

  ee_en  ->  ghananlp-4M        (ewe -> eng)
  ee_fr  ->  michsethowusu      (ewe -> fra)

Étape 1 de la procédure « par source » (audit) :
  * récupère les cosinus SONAR de CHAQUE source (Hub `sonar_scores/` ou local) ;
  * calcule la distribution (quantiles, moyenne, écart-type) ;
  * détecte la séparation bruit/signal par la méthode d'Otsu (bimodalité) ;
  * trace l'histogramme et la courbe « volume retenu vs seuil » ;
  * propose plusieurs seuils candidats par source et compare au seuil actuel.

Le script NE recalcule PAS SONAR (pas de GPU requis) : il exploite les scores
déjà produits par `kaggle/kernels/sonar-rescore-mined/`.

Sorties (dans --out, défaut data/interim/mined_audit/) :
    audit_report.json          stats + seuils proposés par source
    hist_<source>.png          histogramme annoté des seuils
    volume_<source>.png        compromis volume/qualité

Exemples
--------
    # via le Hub (token dans .env : HF_TOKEN_READ)
    set -a; source ./.env; set +a
    .venv/bin/python src/prepare/audit_mined_sources.py

    # si les scores ont déjà été téléchargés localement
    .venv/bin/python src/prepare/audit_mined_sources.py --scores-dir data/interim/sonar_scores
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "data" / "interim" / "mined_audit"

CAND_REPO = "romaricnadjire/ewe-mined-candidates"
SCORES_SUBDIR = "sonar_scores"
REALIGNED_SUBDIR = "realigned"

# Préfixe -> libellé lisible (chaque préfixe = UNE source minée).
SOURCES = {
    "ee_en": "ghananlp-4M (ewe->eng)",
    "ee_fr": "michsethowusu (ewe->fra)",
}


# --------------------------------------------------------------------------- IO
def _token() -> str | None:
    return (os.environ.get("HF_TOKEN_READ")
            or os.environ.get("HF_TOKEN")
            or os.environ.get("HF_TOKEN_WRITE"))


def fetch_scores_from_hub(prefix: str, work: Path) -> np.ndarray | None:
    """Télécharge et concatène les cosinus SONAR d'une source depuis le Hub."""
    from huggingface_hub import hf_hub_download, list_repo_files
    tok = _token()
    try:
        files = list_repo_files(CAND_REPO, repo_type="dataset", token=tok)
    except Exception as e:  # noqa: BLE001
        print(f"  [Hub] inaccessible ({type(e).__name__}: {e})", file=sys.stderr)
        return None
    score_files = sorted(
        f for f in files
        if f.startswith(f"{SCORES_SUBDIR}/scores_{prefix}_") and f.endswith(".jsonl")
    )
    if not score_files:
        return None
    work.mkdir(parents=True, exist_ok=True)
    cos: list[float] = []
    for sf in score_files:
        p = hf_hub_download(CAND_REPO, sf, repo_type="dataset",
                            local_dir=str(work), token=tok)
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    cos.append(float(json.loads(line)["cos"]))
        print(f"  {sf}: {len(cos):,} scores cumulés", flush=True)
    return np.asarray(cos, dtype=np.float64) if cos else None


def load_scores_local(scores_dir: str, prefix: str) -> np.ndarray | None:
    """Lit des scores SONAR locaux (fichiers {idx, cos} ou paires {sonar_cos})."""
    d = Path(scores_dir)
    files = sorted(d.glob(f"scores_{prefix}_*.jsonl")) or sorted(d.glob(f"*{prefix}*.jsonl"))
    cos: list[float] = []
    for p in files:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                o = json.loads(line)
                v = o.get("cos", o.get("sonar_cos"))
                if v is not None:
                    cos.append(float(v))
    return np.asarray(cos, dtype=np.float64) if cos else None


def fetch_current_thresholds(work: Path) -> dict:
    """Récupère thresholds.json (seuils p5-propre actuels) si présent sur le Hub."""
    from huggingface_hub import hf_hub_download
    try:
        p = hf_hub_download(CAND_REPO, f"{REALIGNED_SUBDIR}/thresholds.json",
                            repo_type="dataset", local_dir=str(work), token=_token())
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


# -------------------------------------------------------------------- analyse
def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Seuil d'Otsu : maximise la variance inter-classe (séparation bimodale)."""
    hist, edges = np.histogram(values, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    p = hist.astype(float) / max(hist.sum(), 1)
    w = np.cumsum(p)                      # poids cumulés de la classe basse
    mu = np.cumsum(p * centers)           # moyenne cumulée
    mu_t = mu[-1]
    denom = w * (1.0 - w)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b2 = np.where(denom > 0, (mu_t * w - mu) ** 2 / denom, 0.0)
    return float(centers[int(np.nanargmax(sigma_b2))])


def quantiles(cos: np.ndarray) -> dict:
    return {f"p{q}": round(float(np.percentile(cos, q)), 4)
            for q in (1, 5, 10, 25, 50, 75, 90, 95, 99)}


def analyze_source(prefix: str, cos: np.ndarray, current_thr: float | None,
                   out_dir: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    desc = SOURCES.get(prefix, prefix)
    otsu_thr = otsu_threshold(cos)
    candidates = {
        "otsu": otsu_thr,
        "p50": float(np.percentile(cos, 50)),
        "p75": float(np.percentile(cos, 75)),
    }

    def kept_pct(thr: float) -> float:
        return round(100.0 * float((cos >= thr).mean()), 2)

    stats: dict = {
        "source": desc,
        "n_scored": int(cos.size),
        "mean": round(float(cos.mean()), 4),
        "std": round(float(cos.std()), 4),
        "min": round(float(cos.min()), 4),
        "max": round(float(cos.max()), 4),
        "quantiles": quantiles(cos),
        "otsu_threshold": round(otsu_thr, 4),
        "current_threshold_p5clean": (round(current_thr, 4) if current_thr is not None else None),
        "proposed_thresholds": {
            name: {"threshold": round(t, 4), "kept_pct": kept_pct(t)}
            for name, t in candidates.items()
        },
    }
    if current_thr is not None:
        stats["current_kept_pct"] = kept_pct(current_thr)

    # --- Histogramme annoté ---
    plt.figure(figsize=(9, 5))
    plt.hist(cos, bins=120, color="#4C72B0", alpha=0.85)
    colors = {"otsu": "orange", "p50": "green", "p75": "purple"}
    for name, t in candidates.items():
        plt.axvline(t, color=colors[name], linestyle="--", label=f"{name} = {t:.3f}")
    if current_thr is not None:
        plt.axvline(current_thr, color="red", linewidth=2,
                    label=f"seuil actuel (p5 propre) = {current_thr:.3f}")
    plt.title(f"Distribution cosinus SONAR — {desc} (n={cos.size:,})")
    plt.xlabel("cosinus SONAR"); plt.ylabel("nb paires")
    plt.legend(fontsize=8); plt.grid(alpha=0.3)
    hist_png = out_dir / f"hist_{prefix}.png"
    plt.savefig(hist_png, dpi=120, bbox_inches="tight"); plt.close()

    # --- Courbe volume retenu vs seuil ---
    grid = np.linspace(float(cos.min()), float(cos.max()), 200)
    vol = [100.0 * float((cos >= t).mean()) for t in grid]
    plt.figure(figsize=(9, 5))
    plt.plot(grid, vol, color="#55A868")
    plt.axvline(otsu_thr, color="orange", linestyle="--", label=f"Otsu = {otsu_thr:.3f}")
    if current_thr is not None:
        plt.axvline(current_thr, color="red", label=f"actuel = {current_thr:.3f}")
    plt.title(f"Volume retenu vs seuil — {desc}")
    plt.xlabel("seuil cosinus"); plt.ylabel("% paires retenues")
    plt.legend(); plt.grid(alpha=0.3)
    vol_png = out_dir / f"volume_{prefix}.png"
    plt.savefig(vol_png, dpi=120, bbox_inches="tight"); plt.close()

    stats["plots"] = {
        "histogram": str(hist_png.relative_to(REPO_ROOT)),
        "volume_curve": str(vol_png.relative_to(REPO_ROOT)),
    }
    return stats


# ------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores-dir",
                    help="dossier local des scores SONAR (sinon téléchargement Hub)")
    ap.add_argument("--sources", default="ee_en,ee_fr",
                    help="sources à auditer (préfixes séparés par des virgules)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    current = {} if args.scores_dir else fetch_current_thresholds(out / "_meta")

    report: dict = {"repo": CAND_REPO, "sources": {}}
    for prefix in [s.strip() for s in args.sources.split(",") if s.strip()]:
        print(f"\n=== {prefix} ({SOURCES.get(prefix, '?')}) ===", flush=True)
        cos = (load_scores_local(args.scores_dir, prefix) if args.scores_dir
               else fetch_scores_from_hub(prefix, out / "_dl"))
        if cos is None or cos.size == 0:
            print(f"  Aucun score trouvé pour « {prefix} ». "
                  f"Lancer d'abord le notebook SONAR, ou fournir --scores-dir.",
                  file=sys.stderr)
            continue
        thr = current.get(prefix) or current.get(prefix.replace("ee_", "ee_"))
        s = analyze_source(prefix, cos, thr, out)
        report["sources"][prefix] = s
        q = s["quantiles"]
        print(f"  n={s['n_scored']:,}  moy={s['mean']}  méd(p50)={q['p50']}  "
              f"otsu={s['otsu_threshold']}  seuil_actuel={s['current_threshold_p5clean']}")
        print(f"  -> Otsu retiendrait {s['proposed_thresholds']['otsu']['kept_pct']}% "
              f"(actuel : {s.get('current_kept_pct', '?')}%)")

    rep_path = out / "audit_report.json"
    rep_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRapport : {rep_path.relative_to(REPO_ROOT)}")
    if not report["sources"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
