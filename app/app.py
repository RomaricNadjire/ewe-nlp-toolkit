"""
Traducteur vocal éwé — démo locale (Streamlit), 100 % en mémoire.

Reconnaissance vocale (mms-ewe-asr-mixed) + Traduction (NLLB-600M + LoRA éwé
multi-directions) + Synthèse vocale (MMS-VITS). Les modèles sont chargés
IN-PROCESS et mis en cache pour un maximum de vitesse sur CPU multi-cœurs.

Lancement :  streamlit run app/app.py
"""
from __future__ import annotations

import io
import os
from pathlib import Path

# ─── Optimisation CPU : fixer les threads AVANT l'import de torch ───
_N_CPU = os.cpu_count() or 8
os.environ.setdefault("OMP_NUM_THREADS", str(_N_CPU))
os.environ.setdefault("MKL_NUM_THREADS", str(_N_CPU))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import streamlit as st

# .env du projet (HF_TOKEN_READ pour accéder aux dépôts privés)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except Exception:
    pass

HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HF_TOKEN_READ") or os.getenv("HF_TOKEN_WRITE")

# ─── Modèles ───
BASE_MT = "facebook/nllb-200-distilled-600M"
LORA_MT = os.getenv("HF_MODEL_LORA", "romaricnadjire/nllb-ewe-en-fr-multilingual-lora")
ASR_ID = os.getenv("HF_MODEL_ASR", "romaricnadjire/mms-ewe-asr-mixed")
TTS_IDS = {
    "ee": os.getenv("HF_MODEL_TTS_EE", "facebook/mms-tts-ewe"),
    "en": os.getenv("HF_MODEL_TTS_EN", "facebook/mms-tts-eng"),
    "fr": os.getenv("HF_MODEL_TTS_FR", "facebook/mms-tts-fra"),
}
LANGS = {"Éwé": "ee", "Anglais": "en", "Français": "fr"}
CODE = {"ee": "ewe_Latn", "en": "eng_Latn", "fr": "fra_Latn"}


# ─── Chargement paresseux + cache des modèles (une seule fois) ───
@st.cache_resource(show_spinner=False)
def _torch():
    import torch
    torch.set_num_threads(_N_CPU)
    return torch


@st.cache_resource(show_spinner=False)
def load_asr():
    from transformers import AutoProcessor, Wav2Vec2ForCTC
    processor = AutoProcessor.from_pretrained(ASR_ID, token=HF_TOKEN)
    model = Wav2Vec2ForCTC.from_pretrained(ASR_ID, token=HF_TOKEN).eval()
    return processor, model


@st.cache_resource(show_spinner=False)
def load_mt():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from peft import PeftModel
    tokenizer = AutoTokenizer.from_pretrained(BASE_MT)
    base = AutoModelForSeq2SeqLM.from_pretrained(BASE_MT, token=HF_TOKEN, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(base, LORA_MT, token=HF_TOKEN).eval()
    return tokenizer, model


@st.cache_resource(show_spinner=False)
def load_tts(lang: str):
    from transformers import AutoTokenizer, VitsModel
    tokenizer = AutoTokenizer.from_pretrained(TTS_IDS[lang], token=HF_TOKEN)
    model = VitsModel.from_pretrained(TTS_IDS[lang], token=HF_TOKEN).eval()
    return tokenizer, model


# ─── Inférence ───
# Formats décodables par soundfile / libsndfile (sans ffmpeg).
AUDIO_TYPES = ["wav", "flac", "ogg", "oga", "opus", "mp3", "aiff", "aif", "au", "caf", "w64"]


def _load_audio_16k(data: bytes) -> np.ndarray:
    """Décode des octets audio (WAV/FLAC/OGG/MP3/…) en mono float32 à 16 kHz."""
    import soundfile as sf
    audio, sr = sf.read(io.BytesIO(data), dtype="float32")
    if audio.ndim == 2:                       # stéréo -> mono
        audio = audio.mean(axis=1)
    if sr != 16000:                           # rééchantillonnage haute qualité (soxr)
        import soxr
        audio = soxr.resample(audio, sr, 16000)
    return np.ascontiguousarray(audio, dtype=np.float32)


def transcribe(data: bytes) -> str:
    torch = _torch()
    processor, model = load_asr()
    audio = _load_audio_16k(data)
    if audio.size < 1600 or float(np.abs(audio).mean()) < 1e-4:   # < 0,1 s ou quasi-silence
        raise ValueError(
            "Audio vide ou quasi silencieux. Vérifiez le micro (drivers PipeWire/SOF) "
            "ou importez un fichier audio.")
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    with torch.inference_mode():
        logits = model(inputs.input_values).logits
    pred_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()


def translate(text: str, src: str, tgt: str, num_beams: int) -> str:
    torch = _torch()
    tokenizer, model = load_mt()
    tokenizer.src_lang = CODE[src]
    inputs = tokenizer(text, return_tensors="pt")
    bos = tokenizer.convert_tokens_to_ids(CODE[tgt])
    with torch.inference_mode():
        out = model.generate(**inputs, forced_bos_token_id=bos,
                             max_new_tokens=128, num_beams=num_beams)
    return tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()


def synthesize(text: str, lang: str):
    torch = _torch()
    tokenizer, model = load_tts(lang)
    inputs = tokenizer(text, return_tensors="pt")
    with torch.inference_mode():
        waveform = model(**inputs).waveform
    return model.config.sampling_rate, waveform.squeeze().cpu().numpy()


# ══════════════════════════════ INTERFACE ══════════════════════════════
st.set_page_config(page_title="Kekeli · Traducteur vocal éwé",
                   page_icon="🗣️", layout="centered")

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 860px;}
      h1 {font-weight: 800; letter-spacing: -0.02em; margin-bottom: .1rem;}
      [data-testid="stTabs"] button p {font-size: 1rem; font-weight: 600;}
      .result-card {
        background:#F0FDFA; border:1px solid #99F6E4; border-radius:14px;
        padding:1rem 1.25rem; font-size:1.2rem; line-height:1.6; color:#0F172A;
      }
      .muted {color:#64748B; font-size:.9rem;}
      footer {visibility:hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── État par défaut ──
st.session_state.setdefault("src_lang", "Éwé")
st.session_state.setdefault("tgt_lang", "Français")
st.session_state.setdefault("mt_text", "")


def _swap_langs():
    st.session_state.src_lang, st.session_state.tgt_lang = (
        st.session_state.tgt_lang, st.session_state.src_lang)


# ── Barre latérale : réglages + statut système (Nielsen #1) ──
with st.sidebar:
    st.markdown("### ⚙️ Réglages")
    beams = st.slider(
        "Qualité de traduction", 1, 5, 2,
        help="Plus élevé = meilleure qualité, mais plus lent (num_beams). "
             "1 = le plus rapide.")
    st.markdown("---")
    st.markdown(f"🧵 **CPU** · {_N_CPU} threads")
    if HF_TOKEN:
        st.success("🔑 Token Hugging Face détecté", icon="✅")
    else:
        st.error("Token HF absent : ajoutez `HF_TOKEN_READ` dans `.env` "
                 "pour charger les modèles privés.", icon="⚠️")
    with st.expander("À propos des modèles"):
        st.markdown(
            "- **ASR** : `mms-ewe-asr-mixed`\n"
            "- **Traduction** : NLLB-600M + LoRA `nllb-ewe-en-fr-multilingual-lora`\n"
            "- **TTS** : `mms-tts-{ewe,eng,fra}`\n\n"
            "Meilleure qualité en **éwé ↔ anglais**. Les modèles se chargent "
            "à la 1re utilisation puis restent en mémoire.")

# ── En-tête ──
st.title("🗣️ Traducteur vocal éwé")
st.markdown(
    '<p class="muted">Reconnaissance vocale · Traduction éwé ↔ anglais ↔ français '
    '· Synthèse vocale — 100&nbsp;% en local sur votre machine.</p>',
    unsafe_allow_html=True)
st.write("")

tab_asr, tab_mt, tab_tts = st.tabs(
    ["🎙️  Transcription", "🌐  Traduction", "🔊  Synthèse vocale"])

# ─────────────── Transcription (ASR) ───────────────
with tab_asr:
    st.subheader("Parole en éwé → texte")
    source = st.radio("Source audio", ["🎙️ Micro", "📁 Fichier"],
                      horizontal=True, label_visibility="collapsed")
    data = None
    if source == "🎙️ Micro":
        rec = st.audio_input("Enregistrez votre voix")
        if rec is not None:
            data = rec.getvalue()
        else:
            st.caption("🎧 Parlez distinctement, 2 à 15 s. Micro capricieux ? "
                       "Basculez sur « Fichier ».")
    else:
        up = st.file_uploader("Importez un fichier audio", type=AUDIO_TYPES,
                              label_visibility="collapsed")
        if up is not None:
            st.audio(up)                       # écoute de contrôle
            data = up.getvalue()
        else:
            st.caption("Formats : WAV, FLAC, OGG, MP3, AIFF… "
                       "(m4a/webm : convertir en WAV d'abord).")

    if data is not None and st.button("Transcrire", type="primary", use_container_width=True):
        try:
            with st.spinner("Transcription en cours…"):
                st.session_state["asr_text"] = transcribe(data)
        except Exception as exc:               # état d'erreur
            st.error(f"Échec de la transcription : {exc}")

    if st.session_state.get("asr_text"):
        st.markdown("**Transcription**")
        st.markdown(f'<div class="result-card">{st.session_state["asr_text"]}</div>',
                    unsafe_allow_html=True)
        if st.button("➡️  Envoyer vers la traduction", use_container_width=True):
            st.session_state["mt_text"] = st.session_state["asr_text"]
            st.session_state["src_lang"] = "Éwé"
            st.toast("Texte copié dans l'onglet Traduction ✅")

# ─────────────── Traduction (NLLB + LoRA) ───────────────
with tab_mt:
    col_src, col_swap, col_tgt = st.columns([6, 1, 6])
    with col_src:
        st.selectbox("De", list(LANGS), key="src_lang")
    with col_swap:
        st.write("")
        st.write("")
        st.button("↔", help="Inverser les langues", on_click=_swap_langs,
                  use_container_width=True)
    with col_tgt:
        st.selectbox("Vers", list(LANGS), key="tgt_lang")

    text = st.text_area("Texte à traduire", key="mt_text", height=150,
                        placeholder="Saisissez du texte…")

    if st.button("Traduire", type="primary", use_container_width=True):
        src, tgt = LANGS[st.session_state.src_lang], LANGS[st.session_state.tgt_lang]
        if not text.strip():                       # état vide
            st.warning("Entrez d'abord du texte à traduire.")
        elif src == tgt:                           # prévention d'erreur
            st.info("Choisissez deux langues différentes.")
        else:
            try:
                with st.spinner("Traduction en cours…"):
                    st.session_state["mt_result"] = translate(text, src, tgt, beams)
            except Exception as exc:
                st.error(f"Échec de la traduction : {exc}")

    if st.session_state.get("mt_result"):
        st.markdown("**Traduction**")
        st.markdown(f'<div class="result-card">{st.session_state["mt_result"]}</div>',
                    unsafe_allow_html=True)

# ─────────────── Synthèse vocale (TTS) ───────────────
with tab_tts:
    st.subheader("Texte → voix")
    lang_name = st.selectbox("Langue", list(LANGS), key="tts_lang")
    tts_text = st.text_area("Texte à lire", height=130, key="tts_text",
                            placeholder="Saisissez du texte à vocaliser…")
    if st.button("Générer l'audio", type="primary", use_container_width=True):
        if not tts_text.strip():
            st.warning("Entrez du texte à vocaliser.")
        else:
            try:
                with st.spinner("Génération de la voix…"):
                    sr, wav = synthesize(tts_text, LANGS[lang_name])
                st.audio(wav, sample_rate=sr)
            except Exception as exc:
                st.error(f"Échec de la synthèse : {exc}")
