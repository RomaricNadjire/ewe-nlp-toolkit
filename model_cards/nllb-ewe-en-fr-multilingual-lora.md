---
license: cc-by-nc-4.0
language:
- ee
- en
- fr
library_name: peft
pipeline_tag: translation
base_model: facebook/nllb-200-distilled-600M
tags:
- translation
- nllb
- lora
- peft
- ewe
- low-resource
metrics:
- bleu
- chrf
---

# nllb-ewe-en-fr-multilingual-lora — Ewe ⇄ English ⇄ French (NLLB LoRA)

A **LoRA adapter** for [`facebook/nllb-200-distilled-600M`](https://huggingface.co/facebook/nllb-200-distilled-600M)
fine-tuned for **multi-directional translation** between **Ewe** (`ewe_Latn`),
**English** (`eng_Latn`) and **French** (`fra_Latn`). A single adapter handles all
six directions (Ewe as both source and target).

## Model details

- **Developed by:** Romaric Nadjire — academic project (M1 AI, INF2229)
- **Type:** PEFT / LoRA adapter for an NLLB-200 seq2seq translation model
- **Languages:** Ewe (`ee`), English (`en`), French (`fr`)
- **Fine-tuned from:** `facebook/nllb-200-distilled-600M`
- **License:** **CC-BY-NC-4.0** (inherited from the base model — non-commercial, academic use)

## Evaluation

Test-set BLEU / chrF++ (sacreBLEU), per direction:

| Direction | BLEU ↑ | chrF++ ↑ | n |
|---|--:|--:|--:|
| ewe → eng | 26.16 | 44.88 | 4336 |
| eng → ewe | 23.29 | 43.18 | 4340 |
| ewe → fra | 6.27 | 24.61 | 2362 |
| fra → ewe | 4.12 | 29.09 | 2363 |
| eng → fra | 41.55 | 56.43 | 1988 |
| fra → eng | 46.39 | 65.45 | 1988 |

Ewe↔English is usable; **Ewe↔French is weak** (much less parallel French↔Ewe
data) and should be considered experimental.

## How to get started

```python
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

BASE = "facebook/nllb-200-distilled-600M"
ADAPTER = "romaricnadjire/nllb-ewe-en-fr-multilingual-lora"
CODE = {"ee": "ewe_Latn", "en": "eng_Latn", "fr": "fra_Latn"}

tokenizer = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForSeq2SeqLM.from_pretrained(BASE)
model = PeftModel.from_pretrained(model, ADAPTER).eval()

def translate(text, src, tgt, num_beams=4):
    tokenizer.src_lang = CODE[src]
    inputs = tokenizer(text, return_tensors="pt")
    out = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(CODE[tgt]),
        max_new_tokens=128, num_beams=num_beams,
    )
    return tokenizer.batch_decode(out, skip_special_tokens=True)[0]

print(translate("Ŋdi na mi", "ee", "fr"))
```

## Training

- **Base:** `facebook/nllb-200-distilled-600M` (frozen) + LoRA adapter.
- **Data:** trilingual parallel corpus (Ewe / English / French), deduplicated with
  a canonical order-independent key and filtered (foreign-script, length ratio,
  copies, max length); held-out validation/test excluded from training (anti-leak).
- Trained on Kaggle GPU.

## Data sources & attribution

- **OPUS `bible-uedin`** (ee↔en, ee↔fr) — public domain / OPUS terms.
- **GhanaNLP** navigation corpus (ee↔en) — CC-BY 4.0.
- **Kaggle `tchaye59`** Ewe–English pairs (mixed sources incl. religious texts) — research use.
- English↔French pivot data and public-domain UDHR (glk360).
- Base model **NLLB-200** (`facebook/nllb-200-distilled-600M`) by Meta AI — **CC-BY-NC 4.0**.
- `masakhane/mafand` (CC-BY-NC) was used **for evaluation only**, never for training.

## License & citation

This adapter is a derivative of `facebook/nllb-200-distilled-600M` and is released
under **CC-BY-NC-4.0** (non-commercial). Please credit the base model (Meta NLLB-200)
and the data sources above.

> Academic / portfolio project. Not legal advice; verify licenses before any
> non-research or commercial use.
