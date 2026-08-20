import json

NB = "kaggle_kernels/asr-ewe-mms-v2/asr_ewe_mms_continue.ipynb"
with open(NB, encoding="utf-8") as f:
    nb = json.load(f)
cells = nb["cells"]

def src(i): return "".join(cells[i].get("source", []))
def set_src(i, code): cells[i]["source"] = code

# ─── Cell [6] : Data loading (streaming WaxalNLP) ─────────────────────────────
set_src(6, r"""# ---- Authentification Hugging Face ----
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

def _get_secret(name):
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception:
        return os.environ.get(name)

HF_TOKEN_READ  = _get_secret("HF_TOKEN_READ")  or _get_secret("HF_TOKEN")
HF_TOKEN_WRITE = _get_secret("HF_TOKEN_WRITE") or HF_TOKEN_READ
if HF_TOKEN_READ:
    os.environ["HF_TOKEN"] = HF_TOKEN_READ

# --- 1) Jeu d'origine : snapshot_download bulk (dataset modeste, eprouve) ---
from huggingface_hub import snapshot_download

ORIG_LOCAL_DIR = snapshot_download(
    repo_id    = DATASET_ID,
    repo_type  = "dataset",
    token      = HF_TOKEN_READ or True,
    max_workers= 4,
    local_dir  = os.path.join(_SCRATCH, "ewe_asr_orig"),
)
orig = load_dataset("audiofolder", data_dir=ORIG_LOCAL_DIR, token=HF_TOKEN_READ or True)
if "transcription" in orig["train"].column_names:
    orig = orig.rename_column("transcription", "sentence")
orig = DatasetDict({
    s: orig[s].remove_columns([c for c in orig[s].column_names if c not in ["audio","sentence"]])
    for s in orig
}).cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
print("Origine :", {s: len(orig[s]) for s in orig})

# --- 2) WaxalNLP en STREAMING ---
# streaming=True : les shards parquet sont telecharges au fur et a mesure pendant
# l'entrainement, pas tous d'un coup (centaines de Go au total -> timeout).

def _load_waxal_stream(split):
    """Charge un split WaxalNLP (ewe_asr) en streaming."""
    ds = load_dataset(
        WAXAL_SOURCE, "ewe_asr",
        split=split,
        streaming=True,
        token=HF_TOKEN_READ or True,
    )
    ds = ds.rename_column("transcription", "sentence")
    try:
        extra = [c for c in ds.features if c not in {"audio", "sentence"}]
        if extra:
            ds = ds.remove_columns(extra)
    except Exception:
        pass
    return ds.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

def _materialize(stream_ds, max_n=None):
    """Convertit un IterableDataset -> Dataset classique (pour val/test, petits)."""
    from datasets import Dataset as _DS
    rows = []
    for i, row in enumerate(stream_ds):
        if max_n is not None and i >= max_n:
            break
        rows.append({"audio": row["audio"], "sentence": row["sentence"]})
    result = _DS.from_list(rows)
    return result.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

# Val et test WaxalNLP : petits (< 300 ex) -> materialiser maintenant
print("Chargement WaxalNLP val + test (materialisation)...")
wax_val  = _materialize(_load_waxal_stream("validation"))
wax_test = _materialize(_load_waxal_stream("test"))
print(f"  WaxalNLP : val={len(wax_val)}  test={len(wax_test)}")

# Train WaxalNLP : IterableDataset (streaming progressif)
wax_train_stream = _load_waxal_stream("train")
if MAX_WAXAL_SAMPLES:
    wax_train_stream = wax_train_stream.take(MAX_WAXAL_SAMPLES)

# --- 3) Fusion ---
if COMBINE_ORIGINAL:
    val_combined = concatenate_datasets([orig["validation"], wax_val])
else:
    val_combined = wax_val

ds = DatasetDict({
    "validation" : val_combined,
    "test"       : orig["test"],       # ORIGINE -> comparable au run precedent (WER 13.5 %)
    "test_waxal" : wax_test,           # WaxalNLP -> mesure sur le nouveau domaine
})

# Train : IterableDataset chaine (orig converti + waxal streaming)
if COMBINE_ORIGINAL:
    orig_train_iter = orig["train"].to_iterable_dataset(num_shards=16)
    ds_train_iter   = concatenate_datasets([orig_train_iter, wax_train_stream])
else:
    ds_train_iter = wax_train_stream

print(f"Val combinee : {len(ds['validation'])}  |  Test origine : {len(ds['test'])}")
print("Train : IterableDataset streaming (~15k orig + ~15k waxal)")
""")

# ─── Cell [7] : inspect example ───────────────────────────────────────────────
set_src(7, """# Inspecter un exemple depuis le jeu d'origine (acces direct par index)
ex = orig["train"][0]
print("Phrase        :", ex["sentence"])
print("Sampling rate :", ex["audio"]["sampling_rate"])
print("Duree (s)     :", round(len(ex["audio"]["array"]) / ex["audio"]["sampling_rate"], 2))
""")

# ─── Cell [9] : text normalization ────────────────────────────────────────────
set_src(9, r"""# Normalisation appliquee aux splits val/test (Datasets classiques).
# Le train streaming est normalise directement dans prepare_batch (section 6).
chars_to_remove_regex = r'[\,\?\.\!\-\;\:\"""„\u201f\u2018\u2019«»…()\[\]/]'

def remove_special_characters(batch):
    text = re.sub(chars_to_remove_regex, "", batch["sentence"])
    batch["sentence"] = text.lower().strip()
    return batch

ds = ds.map(remove_special_characters, desc="Nettoyage du texte (val/test)")
print("Exemple nettoye (val) :", ds["validation"][0]["sentence"])
""")

# ─── Cell [13] : OOV check ────────────────────────────────────────────────────
set_src(13, """# Jeu de caracteres connu (vocabulaire fige) : '|' represente l'espace
known = set(processor.tokenizer.get_vocab().keys())
for tok in ["[UNK]", "[PAD]", "<s>", "</s>"]:
    known.discard(tok)
if "|" in known:
    known.discard("|")
    known.add(" ")

# Caracteres presents dans val/test + orig train (Datasets classiques).
# Le train WaxalNLP (streaming) est ignore ici pour eviter un telechargement inutile.
seen = {}
for split in ds:  # validation, test, test_waxal
    for txt in ds[split]["sentence"]:
        for ch in txt:
            seen[ch] = seen.get(ch, 0) + 1
for txt in orig["train"]["sentence"]:
    for ch in txt:
        seen[ch] = seen.get(ch, 0) + 1

oov = sorted(c for c in seen if c not in known)
n_chars = sum(seen.values())
n_oov   = sum(seen[c] for c in oov)
print(f"Caracteres distincts vus : {len(seen)} | hors-vocabulaire : {len(oov)}")
print(f"Occurrences OOV : {n_oov} / {n_chars} ({100*n_oov/max(n_chars,1):.4f} %)")
if oov:
    print("Exemples OOV :", "".join(oov)[:60])
    print("-> ces caracteres seront encodes en [UNK] (negligeable si le % est tres faible).")
else:
    print("Couverture parfaite : aucun caractere hors-vocabulaire.")
print("(Train WaxalNLP streaming ignore dans ce check)")
""")

# ─── Cell [15] : prepare_batch / set_transform ────────────────────────────────
set_src(15, r"""# Deux cas selon le type de dataset :
# - Train (IterableDataset) : .map() lazy -- prepare + telecharge a la volee.
# - Val/Test (Dataset classique) : set_transform, aucune ecriture disque.

_MAX_LEN = int(MAX_AUDIO_SEC * SAMPLING_RATE)
_norm_re  = re.compile(chars_to_remove_regex)

def _normalize_text(text):
    return _norm_re.sub("", text or "").lower().strip()

def prepare_batch(batch):
    out = {"input_values": [], "labels": []}
    for audio, sentence in zip(batch["audio"], batch["sentence"]):
        sentence = _normalize_text(sentence)   # idempotent (no-op si deja normalise)
        iv = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        out["input_values"].append(iv[:_MAX_LEN])
        out["labels"].append(processor(text=sentence).input_ids)
    return out

# Train (streaming) : map lazy
ds_train_prep = ds_train_iter.map(
    prepare_batch,
    batched=True,
    remove_columns=["audio", "sentence"],
)

# Val / Test (Datasets classiques) : set_transform a la volee
ds_eval = ds   # DatasetDict : validation, test, test_waxal
for _split in ds_eval:
    ds_eval[_split].set_transform(prepare_batch)

print("ds_train_prep : IterableDataset (streaming + map)")
print(f"ds_eval       : {list(ds_eval.keys())} (set_transform)")
""")

# ─── Cell [23] : TrainingArguments + hub push ─────────────────────────────────
set_src(23, """# Avec un IterableDataset, num_train_epochs est ignore (pas de len()).
# max_steps remplace : ~30k ex (orig+waxal), batch effectif=8 -> ~3750 steps/epoque -> 7500 pour 2.
MAX_STEPS = 7_500

training_args = TrainingArguments(
    output_dir = OUTPUT_DIR,
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 2,        # batch effectif = 8
    per_device_eval_batch_size  = 4,
    learning_rate = LEARNING_RATE,
    warmup_steps  = 200,
    max_steps     = MAX_STEPS,              # remplace num_train_epochs (IterableDataset)
    gradient_checkpointing = True,
    fp16 = (device == "cuda"),
    eval_strategy = "steps",
    eval_steps    = 500,
    save_steps    = 500,
    logging_steps = 25,
    load_best_model_at_end = True,
    metric_for_best_model  = "wer",
    greater_is_better      = False,
    save_total_limit       = 2,             # garder 2 checkpoints locaux
    dataloader_num_workers = 2,
    disable_tqdm           = True,
    report_to              = "none",
    # --- Resilience Kaggle : push automatique de chaque checkpoint sur HF Hub ---
    push_to_hub      = True,
    hub_model_id     = NEW_HUB_REPO,
    hub_token        = HF_TOKEN_WRITE,
    hub_strategy     = "checkpoint",        # pousse a chaque save_steps=500
    hub_private_repo = True,
)

trainer = Trainer(
    model            = model,
    args             = training_args,
    train_dataset    = ds_train_prep,           # IterableDataset
    eval_dataset     = ds_eval["validation"],   # Dataset classique
    data_collator    = data_collator,
    compute_metrics  = compute_metrics,
    processing_class = processor,
)
print(f"Trainer MMS pret. max_steps={MAX_STEPS}")
""")

# ─── Cell [25] : checkpoint resume (local puis Hub) ───────────────────────────
set_src(25, """from huggingface_hub import HfApi, snapshot_download as _snap_dl

last_ckpt = None
output_path = Path(OUTPUT_DIR)

# 1. Chercher un checkpoint LOCAL (priorite maximale)
if output_path.is_dir():
    ckpts = sorted(
        [d for d in output_path.iterdir()
         if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[-1]),
    )
    if ckpts:
        last_ckpt = str(ckpts[-1])
        print(f"Reprise locale : {last_ckpt}")

# 2. Sinon, telecharger le dernier checkpoint depuis le HUB
if last_ckpt is None and HF_TOKEN_WRITE:
    try:
        api = HfApi()
        entries = list(api.list_repo_tree(
            NEW_HUB_REPO, repo_type="model",
            token=HF_TOKEN_WRITE, recursive=False,
        ))
        ckpt_dirs = sorted(
            {e.rfilename.split("/")[0] for e in entries
             if e.rfilename.startswith("checkpoint-")},
            key=lambda x: int(x.split("-")[-1]),
        )
        if ckpt_dirs:
            latest = ckpt_dirs[-1]
            print(f"Telechargement checkpoint Hub : {NEW_HUB_REPO}/{latest} ...")
            _snap_dl(
                repo_id        = NEW_HUB_REPO,
                repo_type      = "model",
                token          = HF_TOKEN_WRITE,
                allow_patterns = [f"{latest}/*"],
                local_dir      = OUTPUT_DIR,
            )
            last_ckpt = str(output_path / latest)
            print(f"Reprise Hub : {last_ckpt}")
        else:
            print("Aucun checkpoint sur le Hub -> entrainement from scratch.")
    except Exception as e:
        print(f"Hub inaccessible ({e}) -> entrainement from scratch.")

if last_ckpt is None:
    print("Aucun checkpoint -> entrainement from scratch.")

train_result = trainer.train(resume_from_checkpoint=last_ckpt)

# Sauvegarde finale locale
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
model.save_pretrained(OUTPUT_DIR)
print(f"\\nModele MMS sauvegarde : {OUTPUT_DIR}")
print(f"Loss train finale : {train_result.training_loss:.4f}")
""")

# ─── Cell [27] : final eval (ds_prep -> ds_eval) ─────────────────────────────
old27 = src(27)
new27 = (old27
    .replace('ds_prep["validation"]', 'ds_eval["validation"]')
    .replace('ds_prep["test"]',       'ds_eval["test"]')
    .replace('ds_prep["test_waxal"]', 'ds_eval["test_waxal"]'))
set_src(27, new27)

# ─── Cell [28] : transcription example (ds["test"] -> orig["test"]) ──────────
old28 = src(28)
new28 = old28.replace('ex = ds["test"][0]', 'ex = orig["test"][0]')
set_src(28, new28)

# ─── Save ─────────────────────────────────────────────────────────────────────
with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("Done — 8 cells patched.")
