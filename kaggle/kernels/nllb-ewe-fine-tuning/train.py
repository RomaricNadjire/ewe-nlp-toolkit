"""
train.py — Fine-tuning NLLB éwé→anglais sur Kaggle (arrière-plan / script mode)
=================================================================================
• GPU T4 x2 recommandé (activer via Notebook Settings → Accelerator)
• Internet doit être activé (pour télécharger le modèle et le dataset HF)
• Ajouter le secret "HF_TOKEN" dans Add-ons → Secrets avant de lancer
• Résultats persistés dans /kaggle/working/ et poussés sur HF Hub à la fin
"""

# ── 0. Paquets supplémentaires ────────────────────────────────────────────────
import subprocess, sys

def pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])

pip("evaluate", "sacrebleu", "peft", "accelerate", "datasets", "transformers")

# ── 1. Imports ────────────────────────────────────────────────────────────────
import json, os, re
from pathlib import Path

import numpy as np
import torch
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

# ── 2. Environnement & token HF ───────────────────────────────────────────────
IS_KAGGLE = os.path.exists("/kaggle/working")

if IS_KAGGLE:
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
else:
    HF_TOKEN = os.environ.get("HF_TOKEN", "")

os.environ["HF_TOKEN"] = HF_TOKEN

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device : {device}  |  Kaggle : {IS_KAGGLE}")

# ── 3. Configuration ──────────────────────────────────────────────────────────
MODEL_NAME   = "facebook/nllb-200-distilled-600M"
OUTPUT_DIR   = "/kaggle/working/output/nllb-ewe-eng" if IS_KAGGLE else "./output/nllb-ewe-eng"
ADAPTER_DIR  = os.path.join(OUTPUT_DIR, "adapter")
RESULTS_FILE = os.path.join(OUTPUT_DIR, "resultats_evaluation.json")
HF_REPO_ID   = "romaricnadjire/nllb-ewe-lora-adapter"   # dépôt de l'adaptateur

SRC_LANG = "ewe_Latn"
TGT_LANG = "eng_Latn"

MAX_INPUT_LEN  = 128
MAX_TARGET_LEN = 128

LEARNING_RATE     = 3e-4
BATCH_SIZE_TRAIN  = 8
BATCH_SIZE_EVAL   = 16
GRAD_ACCUM_STEPS  = 2
NUM_EPOCHS        = 3
WARMUP_RATIO      = 0.06
WEIGHT_DECAY      = 0.01

LORA_R              = 16
LORA_ALPHA          = 32
LORA_DROPOUT        = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]

BASELINE_SAMPLE = 200

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output dir : {OUTPUT_DIR}")

# ── 4. Chargement du dataset ──────────────────────────────────────────────────
ds_raw = load_dataset(
    "romaricnadjire/ewe-nllb-translation",
    data_files={
        "train":      "train.jsonl",
        "validation": "validation.jsonl",
        "test":       "test.jsonl",
    },
    token=True,
)
print(ds_raw)

# ── 5. Nettoyage du bruit ─────────────────────────────────────────────────────
EWE_CHARS    = set("ŋɖɔɛʋƒãẽĩõũ")
BIBLE_REF_RE = re.compile(r"^\s*\d{1,3}:\d{1,3}(?:-\d{1,3})?\s*$")

def is_noisy(example):
    t   = example["translation"]
    src = (t.get(SRC_LANG) or "").strip()
    tgt = (t.get(TGT_LANG) or "").strip()
    if src and tgt and src == tgt:
        return True
    if BIBLE_REF_RE.match(src) and src == tgt:
        return True
    if TGT_LANG != "ewe_Latn" and sum(ch in EWE_CHARS for ch in tgt) >= 2:
        return True
    return False

raw_before = ds_raw
ds_raw = ds_raw.filter(lambda ex: not is_noisy(ex))
for split in ["train", "validation", "test"]:
    removed = len(raw_before[split]) - len(ds_raw[split])
    print(f"  {split:<10} supprimés={removed} | restants={len(ds_raw[split])}")

# ── 6. Métriques ──────────────────────────────────────────────────────────────
sacrebleu_metric = evaluate.load("sacrebleu")
chrf_metric      = evaluate.load("chrf")

# ── 7. Évaluation baseline ───────────────────────────────────────────────────
def translate_batch(model, tokenizer, sources, src_lang, tgt_lang,
                    batch_size=16, max_new_tokens=128):
    tokenizer.src_lang = src_lang
    forced_bos = tokenizer.convert_tokens_to_ids(tgt_lang)
    all_preds  = []
    for i in range(0, len(sources), batch_size):
        batch  = sources[i : i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=MAX_INPUT_LEN,
        ).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs, forced_bos_token_id=forced_bos,
                max_new_tokens=max_new_tokens, num_beams=4,
            )
        all_preds.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
    return all_preds


def compute_metrics_on_split(model, tokenizer, dataset_split, n_samples=None):
    if n_samples:
        dataset_split = dataset_split.select(range(min(n_samples, len(dataset_split))))
    pairs = [
        (ex["translation"].get(SRC_LANG), ex["translation"].get(TGT_LANG))
        for ex in dataset_split
        if ex["translation"].get(SRC_LANG) and ex["translation"].get(TGT_LANG)
    ]
    sources, references = zip(*pairs)
    print(f"  Génération de {len(sources)} traductions…")
    predictions = translate_batch(model, tokenizer, list(sources), SRC_LANG, TGT_LANG)
    bleu = sacrebleu_metric.compute(predictions=predictions, references=[[r] for r in references])
    chrf = chrf_metric.compute(predictions=predictions, references=[[r] for r in references], word_order=2)
    return {"bleu": round(bleu["score"], 2), "chrf++": round(chrf["score"], 2), "n": len(sources)}


print("Chargement du modèle baseline…")
tokenizer_base = AutoTokenizer.from_pretrained(MODEL_NAME)
model_base     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
model_base.eval()

print("=== BASELINE — validation rapide ===")
baseline_val = compute_metrics_on_split(model_base, tokenizer_base, ds_raw["validation"], BASELINE_SAMPLE)
print(f"  BLEU   : {baseline_val['bleu']}")
print(f"  chrF++ : {baseline_val['chrf++']}")

print("=== BASELINE — test complet ===")
baseline_test = compute_metrics_on_split(model_base, tokenizer_base, ds_raw["test"])
print(f"  BLEU   : {baseline_test['bleu']}")
print(f"  chrF++ : {baseline_test['chrf++']}")

results = {
    "modele": MODEL_NAME, "paire": f"{SRC_LANG} → {TGT_LANG}",
    "baseline": {"validation_sample": baseline_val, "test": baseline_test},
    "fine_tune": {},
}
with open(RESULTS_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

del model_base, tokenizer_base
if device == "cuda":
    torch.cuda.empty_cache()

# ── 8. Tokenisation ───────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.src_lang    = SRC_LANG
FORCED_BOS_TOKEN_ID   = tokenizer.convert_tokens_to_ids(TGT_LANG)

def preprocess(batch):
    sources = [ex[SRC_LANG] or "" for ex in batch["translation"]]
    targets = [ex[TGT_LANG] or "" for ex in batch["translation"]]
    model_inputs = tokenizer(
        sources, text_target=targets, max_length=MAX_INPUT_LEN, truncation=True,
    )
    model_inputs["labels"] = [ids[:MAX_TARGET_LEN] for ids in model_inputs["labels"]]
    return model_inputs

tokenized = ds_raw.map(
    preprocess, batched=True,
    remove_columns=ds_raw["train"].column_names, desc="Tokenisation",
)
print(tokenized)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer, model=None, label_pad_token_id=-100, pad_to_multiple_of=8,
)

# ── 9. LoRA ───────────────────────────────────────────────────────────────────
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, r=LORA_R, lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT, target_modules=LORA_TARGET_MODULES,
    bias="none", use_rslora=True,
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# ── 10. Entraînement ──────────────────────────────────────────────────────────
def compute_metrics(eval_preds):
    pred_ids, label_ids = eval_preds
    label_ids   = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)
    predictions = tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
    references  = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    bleu = sacrebleu_metric.compute(predictions=predictions, references=[[r] for r in references])
    chrf = chrf_metric.compute(predictions=predictions, references=[[r] for r in references], word_order=2)
    return {"bleu": round(bleu["score"], 2), "chrf++": round(chrf["score"], 2)}

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    eval_steps=500, save_steps=500, logging_steps=100,
    per_device_train_batch_size=BATCH_SIZE_TRAIN,
    per_device_eval_batch_size=BATCH_SIZE_EVAL,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    gradient_checkpointing=True,
    learning_rate=LEARNING_RATE,
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type="cosine",
    weight_decay=WEIGHT_DECAY,
    fp16=(device == "cuda"),
    predict_with_generate=True,
    generation_max_length=MAX_TARGET_LEN,
    eval_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="chrf++",
    greater_is_better=True,
    save_total_limit=2,
    report_to="none",
)

trainer = Seq2SeqTrainer(
    model=model, args=training_args,
    train_dataset=tokenized["train"], eval_dataset=tokenized["validation"],
    processing_class=tokenizer, data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Reprendre depuis le dernier checkpoint si disponible
last_ckpt = None
output_path = Path(OUTPUT_DIR)
if output_path.is_dir():
    ckpts = sorted(
        [d for d in output_path.iterdir()
         if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[-1]),
    )
    if ckpts:
        last_ckpt = str(ckpts[-1])
        print(f"Reprise depuis : {last_ckpt}")

print("Démarrage de l'entraînement…")
train_result = trainer.train(resume_from_checkpoint=last_ckpt)

trainer.model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
print(f"Adaptateur LoRA sauvegardé dans : {ADAPTER_DIR}")
print(f"Étapes   : {train_result.global_step}")
print(f"Loss train finale : {train_result.training_loss:.4f}")

# ── 11. Évaluation finale ─────────────────────────────────────────────────────
print("Chargement du modèle fine-tuné pour évaluation…")
base_model_ft = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model_ft      = PeftModel.from_pretrained(base_model_ft, ADAPTER_DIR)
model_ft      = model_ft.merge_and_unload().to(device)
model_ft.eval()
tokenizer_ft  = AutoTokenizer.from_pretrained(ADAPTER_DIR)

print("=== FINE-TUNÉ — test complet ===")
ft_test = compute_metrics_on_split(model_ft, tokenizer_ft, ds_raw["test"])
print(f"  BLEU   : {ft_test['bleu']}")
print(f"  chrF++ : {ft_test['chrf++']}")

with open(RESULTS_FILE, "r", encoding="utf-8") as f:
    results = json.load(f)
results["fine_tune"]["test"] = ft_test
with open(RESULTS_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

b, ft = results["baseline"]["test"], results["fine_tune"]["test"]
print(f"\n{'Métrique':<12} {'Baseline':>10} {'Fine-tuné':>10} {'Δ':>8}")
print("-" * 44)
print(f"{'BLEU':<12} {b['bleu']:>10} {ft['bleu']:>10} {ft['bleu']-b['bleu']:>+8.2f}")
print(f"{'chrF++':<12} {b['chrf++']:>10} {ft['chrf++']:>10} {ft['chrf++']-b['chrf++']:>+8.2f}")
print(f"\nRésultats complets : {RESULTS_FILE}")

# ── 12. Push de l'adaptateur sur HF Hub ──────────────────────────────────────
if HF_TOKEN:
    print(f"\nPush de l'adaptateur vers {HF_REPO_ID}…")
    from huggingface_hub import HfApi
    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id=HF_REPO_ID, repo_type="model", exist_ok=True, private=True)
    api.upload_folder(folder_path=ADAPTER_DIR, repo_id=HF_REPO_ID, repo_type="model")
    print("Push terminé.")
else:
    print("HF_TOKEN absent — adaptateur non poussé sur le Hub.")
