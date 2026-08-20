#!/usr/bin/env python3
"""
Push trained model artifacts to Hugging Face model repos in PRIVATE mode only.

This script is designed for artifacts downloaded from Kaggle kernel outputs,
without retraining anything.

Expected kernel output folders under --base-dir:
- asr-ewe-mms/
- traduction-multilingue/
- traduction-ewe-fra/
- nllb-ewe-fine-tuning/

Usage:
  python3 push_models_private.py --username <hf_user>
  python3 push_models_private.py --base-dir ./models --dry-run
  python3 push_models_private.py --models asr-ewe-mms traduction-ewe-fra
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError
except Exception:  # pragma: no cover - fallback for dry-run without HF deps
    HfApi = None  # type: ignore[assignment]

    class HfHubHTTPError(Exception):
        """Fallback exception when huggingface_hub is unavailable."""


@dataclass(frozen=True)
class ModelSpec:
    key: str
    kernel_dir: str
    repo_name: str
    candidates: tuple[str, ...]
    required_files_any: tuple[str, ...]
    required_files_all: tuple[str, ...] = ()


MODEL_SPECS: dict[str, ModelSpec] = {
    "asr-ewe-mms": ModelSpec(
        key="asr-ewe-mms",
        kernel_dir="asr-ewe-mms",
        repo_name="mms-ewe-asr",
        candidates=(
            "output/mms-ewe",
            "mms-ewe",
            ".",
        ),
        required_files_any=(
            "adapter.ewe.safetensors",
            "model.safetensors",
            "pytorch_model.bin",
        ),
        required_files_all=("processor_config.json",),
    ),
    "traduction-multilingue": ModelSpec(
        key="traduction-multilingue",
        kernel_dir="traduction-multilingue",
        repo_name="nllb-ewe-multilingual-lora",
        candidates=(
            "output/nllb-ewe-multi/adapter",
            "nllb-ewe-multi/adapter",
            "adapter",
        ),
        required_files_any=("adapter_model.safetensors",),
        required_files_all=("adapter_config.json",),
    ),
    "traduction-ewe-fra": ModelSpec(
        key="traduction-ewe-fra",
        kernel_dir="traduction-ewe-fra",
        repo_name="nllb-ewe-fra-lora",
        candidates=(
            "output/nllb-ewe-fra/adapter",
            "nllb-ewe-fra/adapter",
            "adapter",
        ),
        required_files_any=("adapter_model.safetensors",),
        required_files_all=("adapter_config.json",),
    ),
    "nllb-ewe-fine-tuning": ModelSpec(
        key="nllb-ewe-fine-tuning",
        kernel_dir="nllb-ewe-fine-tuning",
        repo_name="nllb-ewe-eng-lora",
        candidates=(
            "output/nllb-ewe-eng/adapter",
            "nllb-ewe-eng/adapter",
            "adapter",
        ),
        required_files_any=("adapter_model.safetensors",),
        required_files_all=("adapter_config.json",),
    ),
}


# Real evaluation metrics extracted from Kaggle training logs / output JSON.
MODEL_CARDS: dict[str, str] = {
    "asr-ewe-mms": """---
license: cc-by-nc-4.0
language:
- ee
library_name: transformers
pipeline_tag: automatic-speech-recognition
tags:
- automatic-speech-recognition
- mms
- wav2vec2
- ewe
- ctc
base_model: facebook/mms-1b-all
metrics:
- wer
- cer
---

# MMS-1B-all fine-tuned for Ewe ASR (mms-ewe-asr)

Fine-tuned [`facebook/mms-1b-all`](https://huggingface.co/facebook/mms-1b-all)
(Wav2Vec2 CTC) for **automatic speech recognition (ASR) in Ewe (`ee`)**.

## Model details
- **Developed by:** Romaric Nadjire (academic project, INF2229)
- **Model type:** Wav2Vec2 / MMS CTC encoder for speech-to-text
- **Language:** Ewe (`ee`)
- **Finetuned from:** `facebook/mms-1b-all`
- **License:** CC-BY-NC-4.0 (non-commercial; academic use)

## Uses
- **Direct use:** Transcription of Ewe speech audio (16 kHz) to text.
- **Out-of-scope:** Languages other than Ewe; noisy/far-field audio far from the
  training distribution; any commercial use.

## Evaluation

| Split | WER (%) | CER (%) |
|-------|--------:|--------:|
| Baseline (validation, before fine-tuning) | 100.00 | 236.73 |
| Fine-tuned (validation) | 23.92 | 5.99 |
| **Fine-tuned (test)** | **13.52** | **2.11** |

Lower is better. WER = Word Error Rate, CER = Character Error Rate.

## How to get started
```python
from transformers import Wav2Vec2ForCTC, AutoProcessor
import torch, torchaudio

repo = "romaricnadjire/mms-ewe-asr"
processor = AutoProcessor.from_pretrained(repo)
model = Wav2Vec2ForCTC.from_pretrained(repo)

speech, sr = torchaudio.load("audio.wav")
if sr != 16000:
    speech = torchaudio.functional.resample(speech, sr, 16000)
inputs = processor(speech.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
pred_ids = torch.argmax(logits, dim=-1)
print(processor.batch_decode(pred_ids)[0])
```

## Training
- **Base model:** `facebook/mms-1b-all`
- **Task:** CTC speech recognition
- **Hardware:** Kaggle GPU (T4 x2)
- **Framework:** Transformers
""",
    "nllb-ewe-fine-tuning": """---
license: cc-by-nc-4.0
language:
- ee
- en
library_name: peft
pipeline_tag: translation
tags:
- translation
- nllb
- lora
- peft
- ewe
base_model: facebook/nllb-200-distilled-600M
metrics:
- bleu
- chrf
---

# NLLB Ewe->English LoRA adapter (nllb-ewe-eng-lora)

LoRA adapter for [`facebook/nllb-200-distilled-600M`](https://huggingface.co/facebook/nllb-200-distilled-600M)
fine-tuned for **Ewe -> English** translation (`ewe_Latn` -> `eng_Latn`).

## Model details
- **Developed by:** Romaric Nadjire (academic project, INF2229)
- **Model type:** PEFT/LoRA adapter for a seq2seq translation model
- **Languages:** Ewe (`ee`) -> English (`en`)
- **Finetuned from:** `facebook/nllb-200-distilled-600M`
- **License:** CC-BY-NC-4.0 (non-commercial; academic use)

## Evaluation (test set, n=4336)

| Model | BLEU | chrF++ |
|-------|-----:|-------:|
| Baseline NLLB-600M | 13.83 | 33.09 |
| **+ LoRA (this model)** | **27.47** | **46.09** |

Higher is better.

## How to get started
```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

base = "facebook/nllb-200-distilled-600M"
tok = AutoTokenizer.from_pretrained(base, src_lang="ewe_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(base)
model = PeftModel.from_pretrained(model, "romaricnadjire/nllb-ewe-eng-lora")

inputs = tok("Ndi nyuie", return_tensors="pt")
out = model.generate(**inputs, forced_bos_token_id=tok.convert_tokens_to_ids("eng_Latn"))
print(tok.batch_decode(out, skip_special_tokens=True)[0])
```

## Training hyperparameters (LoRA)
- r=16, lora_alpha=32, lora_dropout=0.05, rslora=True
- target_modules: q_proj, v_proj
- task_type: SEQ_2_SEQ_LM
- PEFT 0.18.1
""",
    "traduction-ewe-fra": """---
license: cc-by-nc-4.0
language:
- ee
- fr
library_name: peft
pipeline_tag: translation
tags:
- translation
- nllb
- lora
- peft
- ewe
base_model: facebook/nllb-200-distilled-600M
metrics:
- bleu
- chrf
---

# NLLB Ewe->French LoRA adapter (nllb-ewe-fra-lora)

LoRA adapter for [`facebook/nllb-200-distilled-600M`](https://huggingface.co/facebook/nllb-200-distilled-600M)
fine-tuned for **Ewe -> French** translation (`ewe_Latn` -> `fra_Latn`).

## Model details
- **Developed by:** Romaric Nadjire (academic project, INF2229)
- **Model type:** PEFT/LoRA adapter for a seq2seq translation model
- **Languages:** Ewe (`ee`) -> French (`fr`)
- **Finetuned from:** `facebook/nllb-200-distilled-600M`
- **License:** CC-BY-NC-4.0 (non-commercial; academic use)

## Evaluation (test set, n=2362)

| Model | BLEU | chrF++ |
|-------|-----:|-------:|
| Baseline NLLB-600M | 3.15 | 18.72 |
| **+ LoRA (this model)** | **6.94** | **25.42** |

Higher is better. The Ewe->French pair is harder and lower-resource than
Ewe->English, which explains the lower absolute scores.

## How to get started
```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

base = "facebook/nllb-200-distilled-600M"
tok = AutoTokenizer.from_pretrained(base, src_lang="ewe_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(base)
model = PeftModel.from_pretrained(model, "romaricnadjire/nllb-ewe-fra-lora")

inputs = tok("Ndi nyuie", return_tensors="pt")
out = model.generate(**inputs, forced_bos_token_id=tok.convert_tokens_to_ids("fra_Latn"))
print(tok.batch_decode(out, skip_special_tokens=True)[0])
```

## Training hyperparameters (LoRA)
- r=16, lora_alpha=32, lora_dropout=0.05, rslora=True
- target_modules: q_proj, v_proj
- task_type: SEQ_2_SEQ_LM
- PEFT 0.18.1
""",
    "traduction-multilingue": """---
license: cc-by-nc-4.0
language:
- ee
- en
- fr
library_name: peft
pipeline_tag: translation
tags:
- translation
- nllb
- lora
- peft
- ewe
- multilingual
base_model: facebook/nllb-200-distilled-600M
metrics:
- bleu
- chrf
---

# NLLB Ewe->{English,French} multilingual LoRA adapter (nllb-ewe-multilingual-lora)

A single LoRA adapter for [`facebook/nllb-200-distilled-600M`](https://huggingface.co/facebook/nllb-200-distilled-600M)
trained jointly on **Ewe -> English** and **Ewe -> French**.

## Model details
- **Developed by:** Romaric Nadjire (academic project, INF2229)
- **Model type:** PEFT/LoRA adapter for a seq2seq translation model
- **Languages:** Ewe (`ee`) -> English (`en`) / French (`fr`)
- **Finetuned from:** `facebook/nllb-200-distilled-600M`
- **License:** CC-BY-NC-4.0 (non-commercial; academic use)

## Evaluation (test sets)

| Direction | n | BLEU | chrF++ |
|-----------|--:|-----:|-------:|
| Ewe -> English | 4336 | 27.42 | 46.05 |
| Ewe -> French | 2362 | 6.87 | 25.09 |

Higher is better. The single multilingual adapter nearly matches the dedicated
per-language adapters while covering both directions.

## How to get started
```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

base = "facebook/nllb-200-distilled-600M"
tok = AutoTokenizer.from_pretrained(base, src_lang="ewe_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(base)
model = PeftModel.from_pretrained(model, "romaricnadjire/nllb-ewe-multilingual-lora")

# Choose target: "eng_Latn" or "fra_Latn"
inputs = tok("Ndi nyuie", return_tensors="pt")
out = model.generate(**inputs, forced_bos_token_id=tok.convert_tokens_to_ids("eng_Latn"))
print(tok.batch_decode(out, skip_special_tokens=True)[0])
```

## Training hyperparameters (LoRA)
- r=16, lora_alpha=32, lora_dropout=0.05, rslora=True
- target_modules: q_proj, v_proj
- task_type: SEQ_2_SEQ_LM
- PEFT 0.18.1
""",
}


def eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def has_any_file(path: Path, names: Iterable[str]) -> bool:
    return any((path / n).exists() for n in names)


def has_all_files(path: Path, names: Iterable[str]) -> bool:
    return all((path / n).exists() for n in names)


def resolve_artifact_dir(base_dir: Path, spec: ModelSpec) -> Path:
    kernel_root = base_dir / spec.kernel_dir
    if not kernel_root.exists():
        raise FileNotFoundError(f"Kernel output folder not found: {kernel_root}")

    for rel in spec.candidates:
        candidate = (kernel_root / rel).resolve()
        if not candidate.exists() or not candidate.is_dir():
            continue
        if not has_all_files(candidate, spec.required_files_all):
            continue
        if not has_any_file(candidate, spec.required_files_any):
            continue
        return candidate

    searched = "\n".join(str((kernel_root / c).resolve()) for c in spec.candidates)
    raise FileNotFoundError(
        f"No valid artifact directory found for '{spec.key}'.\n"
        f"Searched:\n{searched}\n"
        f"Expected all: {spec.required_files_all}\n"
        f"Expected any: {spec.required_files_any}"
    )


def ensure_private_repo(api, repo_id: str, dry_run: bool) -> None:
    if dry_run:
        print(f"[DRY-RUN] Would ensure private repo exists: {repo_id}")
        return

    try:
        info = api.repo_info(repo_id=repo_id, repo_type="model")
        if not getattr(info, "private", False):
            raise RuntimeError(
                f"Repo exists but is PUBLIC: {repo_id}. "
                "This script only allows private repos."
            )
        print(f"[OK] Repo already private: {repo_id}")
        return
    except HfHubHTTPError as err:
        if err.response is None or err.response.status_code != 404:
            raise

    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    print(f"[OK] Created private repo: {repo_id}")


def write_model_card(spec: ModelSpec, local_dir: Path, dry_run: bool) -> None:
    card = MODEL_CARDS.get(spec.key)
    if not card:
        return
    readme = local_dir / "README.md"
    if dry_run:
        print(f"[DRY-RUN] Would write model card: {readme}")
        return
    readme.write_text(card, encoding="utf-8")
    print(f"[OK] Wrote model card: {readme}")


def upload_model_folder(api, repo_id: str, local_dir: Path, dry_run: bool) -> None:
    if dry_run:
        print(f"[DRY-RUN] Would upload {local_dir} -> {repo_id}")
        return

    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Upload artifacts from {local_dir.name} (private)",
    )
    print(f"[OK] Uploaded to https://huggingface.co/{repo_id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Push model artifacts to Hugging Face private repos only."
    )
    parser.add_argument(
        "--base-dir",
        default="./models",
        help="Local directory containing Kaggle kernel outputs (default: ./models)",
    )
    parser.add_argument(
        "--username",
        default="romaricnadjire",
        help="Hugging Face username or org (default: romaricnadjire)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_SPECS.keys()),
        default=sorted(MODEL_SPECS.keys()),
        help="Subset of models to upload",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and validate paths without creating/uploading repos",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_dir = Path(args.base_dir).expanduser().resolve()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if HfApi is None:
        if not args.dry_run:
            raise RuntimeError(
                "huggingface_hub is not installed. Install it or run in dry-run mode."
            )
        api = None
    else:
        api = HfApi(token=token)

    print("== Private model publishing ==")
    print(f"Base dir  : {base_dir}")
    print(f"HF user   : {args.username}")
    print(f"Models    : {', '.join(args.models)}")
    print(f"Mode      : {'DRY-RUN' if args.dry_run else 'LIVE'}")

    failed = False
    for key in args.models:
        spec = MODEL_SPECS[key]
        repo_id = f"{args.username}/{spec.repo_name}"
        print("\n" + "-" * 72)
        print(f"Model key : {spec.key}")
        try:
            artifact_dir = resolve_artifact_dir(base_dir, spec)
            print(f"Artifact  : {artifact_dir}")

            write_model_card(spec, artifact_dir, dry_run=args.dry_run)
            ensure_private_repo(api, repo_id, dry_run=args.dry_run)
            upload_model_folder(api, repo_id, artifact_dir, dry_run=args.dry_run)
        except Exception as exc:
            failed = True
            eprint(f"[ERROR] {spec.key}: {exc}")

    print("\nDone.")
    if failed:
        eprint("One or more models failed.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
