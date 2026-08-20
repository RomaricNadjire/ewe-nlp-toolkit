---
license: cc-by-nc-4.0
language:
- ee
library_name: transformers
pipeline_tag: automatic-speech-recognition
base_model: facebook/mms-1b-all
tags:
- automatic-speech-recognition
- mms
- wav2vec2
- ctc
- ewe
- low-resource
metrics:
- wer
- cer
model-index:
- name: mms-ewe-asr-mixed
  results:
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: Ewe mixed multi-domain speech (16 kHz)
      type: audiofolder
      split: validation
    metrics:
    - name: WER
      type: wer
      value: 29.19
    - name: CER
      type: cer
      value: 6.82
---

# mms-ewe-asr-mixed — Ewe speech recognition (MMS-1B fine-tune)

Fine-tuned version of [`facebook/mms-1b-all`](https://huggingface.co/facebook/mms-1b-all)
(Wav2Vec2 / CTC) for **automatic speech recognition (ASR) in Ewe** (`ee`,
`ewe_Latn`), a low-resource Gbe language spoken in Togo and Ghana.

Only the language-specific MMS adapter and the CTC head are trained; the model
loads with plain `Wav2Vec2ForCTC` (the Ewe adapter is baked into the weights).

## Model details

- **Developed by:** Romaric Nadjire — academic project (M1 AI, INF2229)
- **Task / type:** Speech-to-text, Wav2Vec2 CTC (MMS)
- **Language:** Ewe (`ee`)
- **Fine-tuned from:** `facebook/mms-1b-all`
- **License:** **CC-BY-NC-4.0** (inherited from the base model — non-commercial, academic use)

## Intended uses & limitations

- **Direct use:** transcribing Ewe speech (mono, **16 kHz**) to text.
- **Out of scope:** other languages; far-field / very noisy audio; any commercial use.
- **Bias & limitations:** training data is dominated by **read speech** (Bible
  readings and navigation prompts), so accuracy drops on spontaneous,
  conversational or dialectal speech. Numbers, code-switching and named entities
  are error-prone. Reported metrics are on the **validation** split of the mixed set.

## Evaluation

| Model | Split | WER (%) ↓ | CER (%) ↓ |
|---|---|--:|--:|
| Baseline `mms-1b-all` (before fine-tuning) | validation | 100.00 | 236.73 |
| **This model (fine-tuned)** | validation | **29.19** | **6.82** |

WER = Word Error Rate, CER = Character Error Rate (lower is better). The mixed
multi-domain validation set is deliberately harder than a Bible-only test set.

## How to get started

```python
import torch, soundfile as sf, soxr
from transformers import AutoProcessor, Wav2Vec2ForCTC

repo = "romaricnadjire/mms-ewe-asr-mixed"
processor = AutoProcessor.from_pretrained(repo)   # tokenizer target_lang="ewe" is baked in
model = Wav2Vec2ForCTC.from_pretrained(repo).eval()

audio, sr = sf.read("audio.wav", dtype="float32")
if audio.ndim == 2:
    audio = audio.mean(axis=1)
if sr != 16000:
    audio = soxr.resample(audio, sr, 16000)

inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
with torch.inference_mode():
    logits = model(inputs.input_values).logits
pred_ids = torch.argmax(logits, dim=-1)
print(processor.batch_decode(pred_ids, skip_special_tokens=True)[0])
```

## Training

- **Base:** `facebook/mms-1b-all` (MMS adapter + CTC head trained; `ctc_zero_infinity=True`).
- **Data:** a mixed 16 kHz Ewe corpus combining **BibleTTS** Ewe readings and
  **GhanaNLP** navigation-domain speech.
- **Setup:** 8000 steps, lr 3e-4, effective batch size 32 (4 × 8 grad. accum.),
  linear schedule, 500 warmup steps, AMP. Trained on Kaggle GPU.

## Data sources & attribution

- **BibleTTS (Ewe)** — [BibleTTS](https://masakhane-io.github.io/bibleTTS/),
  licensed **CC-BY-SA 4.0**. Attribution required.
- **GhanaNLP** Ewe speech corpora (navigation / Bible audio).
- Base model **MMS** (`facebook/mms-1b-all`) by Meta AI — **CC-BY-NC 4.0**.

## License & citation

This model is a derivative of `facebook/mms-1b-all` and is released under
**CC-BY-NC-4.0** (non-commercial). Please credit the base model (Meta MMS) and
the data sources above (notably BibleTTS, CC-BY-SA 4.0).

> Academic / portfolio project. Not legal advice; verify licenses before any
> non-research or commercial use.
