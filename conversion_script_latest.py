import os
import yaml
import torch
import argparse
import hashlib
import json
import shutil
import traceback
import random
from pathlib import Path
from collections import OrderedDict, defaultdict

import numpy as np
import onnx
import onnxruntime as ort
import librosa
import torch.nn.functional as F
import torch.nn as nn

from munch import Munch
from onnx.tools import update_model_dims
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

import boto3
from botocore.exceptions import ClientError

from common_code.styletts2.Modules.diffusion.sampler import *
from common_code.styletts2.Utils.PLBERT.util import load_plbert
from common_code.styletts2.Modules.hifigan_latest import *
from model_builder_helpers import *
from onnx_exporter_modules import (
    BertEncoderWrapper, PredLSTMWrapper, ProsodyPredictorONNXWrapper,
    IntermediateSteps1, DiffusionONNX, ProsodyPredictorMerged,
    StyleTTS2ProsodyBlock, AcousticMegaWrapper
)

from dotenv import load_dotenv

load_dotenv()

# ── Reproducibility ──────────────────────────────────────────────────────────
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.allow_tf32 = False

# ── Paths ─────────────────────────────────────────────────────────────────────
SPEAKER_CONFIGS_FILE = "speaker_configs.json"   # { "amelie": { "bucket": ..., "config": ..., "model": ..., "reference": ... } }
MODEL_STORE_DIR      = "raw_models/model_store"
REFERENCE_STORE_DIR  = "raw_models/reference_store"
STYLE_CACHE_DIR      = "style_cache"
TRITON_REPOS_DIR     = "triton_repos"
PLBERT_DIR           = os.path.join(
    "/mnt/vdb1/amarnath_dev/TTS_explore/triton_tts/voicingtts-triton-inference/models",
    "PLBERT", "multi"
)
TRITON_TEMPLATE_DIR  = "latest_triton_model_repo/triton_model_repoistory_with_diffusion"

# ── Mel transform (unchanged from original) ───────────────────────────────────
_mel_transform = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300
)
_mel_mean = -4.0
_mel_std  =  4.0


# ═════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═════════════════════════════════════════════════════════════════════════════

def load_speaker_language(path="speaker_languages.json"):
    with open(path) as f:
        return json.load(f)

def flatten_speaker_language(data):
    flat = {}
    for lang, speakers in data.items():
        for spk, model_name in speakers.items():
            flat[spk] = model_name
    return flat

def load_plbert_only(plbert_dir: str):
    return load_plbert(plbert_dir)


def recursive_munch(d):
    if isinstance(d, dict):
        return Munch({k: recursive_munch(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [recursive_munch(x) for x in d]
    return d


def _s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id     = os.getenv("AWS_ACCESS_KEY"),
        aws_secret_access_key = os.getenv("AWS_SECRET_KEY"),
    )


def _download_s3_file(bucket: str, s3_key: str, local_path: str) -> None:
    """Download a single S3 object to *local_path*, creating directories as needed."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    _s3_client().download_file(bucket, s3_key, local_path)
    print(f"  ↓  s3://{bucket}/{s3_key}  →  {local_path}")


def _s3_key_hash(bucket: str, s3_key: str) -> str:
    """Stable 12-char SHA-256 hash of '<bucket>/<s3_key>' – used as a filename stem."""
    raw = f"{bucket}/{s3_key}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def _file_hash(path: str, chunk: int = 1 << 20) -> str:
    """SHA-256 hash of a local file's contents (used for style cache keying)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()[:12]


def load_speaker_configs(path: str = SPEAKER_CONFIGS_FILE) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Speaker config file '{path}' not found.\n"
            "Create it with entries like:\n"
            '  { "amelie": { "bucket": "my-bucket", "config": "path/cfg.yml", '
            '"model": "path/model.pth", "reference": "path/ref.wav" } }'
        )
    with open(path) as f:
        return json.load(f)


def copy_triton_contents(src: str, dst: str) -> None:
    if not os.path.exists(src):
        print(f"[WARN] Triton template '{src}' does not exist – skipping copy.")
        return
    shutil.copytree(src, dst, dirs_exist_ok=True)
    print(f"  Triton template copied → {dst}")


# ═════════════════════════════════════════════════════════════════════════════
#  Smart download  (Requirement 4)
# ═════════════════════════════════════════════════════════════════════════════

def smart_download(s3_config: dict, config_local_dir: str) -> tuple:
    """
    Downloads only what is missing.

    Returns
    -------
    model_path     : str   – path inside MODEL_STORE_DIR
    reference_path : str   – path inside REFERENCE_STORE_DIR
    config_path    : str   – path inside config_local_dir
    model_hash     : str   – 12-char hash identifying the model file
    reference_hash : str   – 12-char hash identifying the reference file
    """
    bucket = s3_config.get("bucket", "verizon-audio-main")
    if not bucket:
        raise ValueError("'bucket' is missing from s3_config")

    for key in ("config", "model", "reference"):
        if not s3_config.get(key):
            raise ValueError(f"'{key}' is missing from s3_config")

    os.makedirs(MODEL_STORE_DIR,     exist_ok=True)
    os.makedirs(REFERENCE_STORE_DIR, exist_ok=True)
    os.makedirs(config_local_dir,    exist_ok=True)

    # ── Config (lightweight – store per-run folder) ──────────────────────────
    config_path = os.path.join(config_local_dir, "config_ft.yml")
    if not os.path.exists(config_path):
        print("[Download] Fetching config …")
        _download_s3_file(bucket, s3_config["config"], config_path)
    else:
        print("[Cache] Config already present – skipping download.")

    # ── Model (global deduplicated store) ────────────────────────────────────
    model_hash  = _s3_key_hash(bucket, s3_config["model"])
    model_path  = os.path.join(MODEL_STORE_DIR, f"{model_hash}.pth")
    if not os.path.exists(model_path):
        print(f"[Download] Fetching model (hash={model_hash}) …")
        _download_s3_file(bucket, s3_config["model"], model_path)
    else:
        print(f"[Cache] Model {model_hash}.pth already in model_store – skipping download.")

    # ── Reference audio (global deduplicated store) ───────────────────────────
    reference_hash = _s3_key_hash(bucket, s3_config["reference"])
    reference_path = os.path.join(REFERENCE_STORE_DIR, f"{reference_hash}.wav")
    if not os.path.exists(reference_path):
        print(f"[Download] Fetching reference (hash={reference_hash}) …")
        _download_s3_file(bucket, s3_config["reference"], reference_path)
    else:
        print(f"[Cache] Reference {reference_hash}.wav already in reference_store – skipping download.")

    return model_path, reference_path, config_path, model_hash, reference_hash


def smart_download_by_model(s3_config, speaker_name, model_id):
    bucket = s3_config.get("bucket", "verizon-audio-main")

    model_dir = os.path.join("raw_models", "models", model_id)
    ref_dir   = os.path.join("raw_models", "references", model_id)
    cfg_dir   = os.path.join("raw_models", "configs", model_id)

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(cfg_dir, exist_ok=True)

    # -------- MODEL (shared per model_id) --------
    model_path = os.path.join(model_dir, "model.pth")
    if not os.path.exists(model_path):
        _download_s3_file(bucket, s3_config["model"], model_path)

    # -------- CONFIG (shared) --------
    config_path = os.path.join(cfg_dir, "config.yml")
    if not os.path.exists(config_path):
        _download_s3_file(bucket, s3_config["config"], config_path)

    # -------- REFERENCE (per speaker) --------
    ref_path = os.path.join(ref_dir, f"{speaker_name}.wav")
    if not os.path.exists(ref_path):
        _download_s3_file(bucket, s3_config["reference"], ref_path)

    return model_path, ref_path, config_path

# ═════════════════════════════════════════════════════════════════════════════
#  ONNX export skip logic  (Requirement 6)
# ═════════════════════════════════════════════════════════════════════════════

# All sub-paths (relative to repo root) that must exist for a full export.
_ONNX_REQUIRED_PATHS = [
    "int_steps_1/1/model.onnx",
    "bert/1/model.onnx",
    "text_encoder/1/model.onnx",
    "bert_encoder/1/model.onnx",
    "prosody_text_encoder/1/model.onnx",
    "prosody_lstm/1/model.onnx",
    "prosody_dur_proj/1/model.onnx",
    "prosody_predictor_ftrain/1/model.onnx",
    "diffusion_steps/1/model.onnx",
    "prep_decoder/1/model.onnx",
    "decoder/1/model.onnx",
    "generator/1/model.onnx",
]


def all_onnx_exports_exist(repo_path: str) -> bool:
    missing = [p for p in _ONNX_REQUIRED_PATHS
               if not os.path.exists(os.path.join(repo_path, p))]
    if missing:
        print(f"[ONNX] {len(missing)} file(s) missing – full export required.")
        for m in missing:
            print(f"       • {m}")
        return False
    print("[ONNX] All required ONNX files already exist – skipping export.")
    return True


# ═════════════════════════════════════════════════════════════════════════════
#  Style generation with caching  (Requirements 7 & 8)
# ═════════════════════════════════════════════════════════════════════════════

def create_reference_style(
    model,
    reference_audio_path: str,
    dest_folder: str,
    speaker_name: str,
) -> torch.Tensor:
    """
    Generates (or reuses) a reference style tensor.

    Cache key  : SHA-256 hash of the reference WAV file contents.
    Cache path : style_cache/<ref_content_hash>.pt
    Output path: <dest_folder>/reference_style_<speaker_name>.pt  (human-readable)

    If the cached tensor already exists the model is never called.
    """
    os.makedirs(STYLE_CACHE_DIR, exist_ok=True)
    os.makedirs(dest_folder,     exist_ok=True)

    # ── Stable cache key based on file *contents* ─────────────────────────
    ref_content_hash = _file_hash(reference_audio_path)
    cache_path       = os.path.join(STYLE_CACHE_DIR, f"{ref_content_hash}.pt")
    output_path      = os.path.join(dest_folder, f"reference_style_{speaker_name}.pt")

    # ── Reuse cached tensor if available ─────────────────────────────────
    if os.path.exists(cache_path):
        print(f"[Cache] Style already computed (hash={ref_content_hash}) – reusing.")
        ref = torch.load(cache_path, map_location="cpu")
        # Always write the human-readable copy (cheap – just a copy)
        torch.save(ref.cpu(), output_path)
        print(f"  Style saved → {output_path}")
        return ref

    # ── Compute style ─────────────────────────────────────────────────────
    print(f"[Style] Computing reference style for speaker '{speaker_name}' …")
    try:
        device = next(model.style_encoder.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    wave, sr = librosa.load(reference_audio_path, sr=24000)
    audio, _ = librosa.effects.trim(wave, top_db=30)
    if sr != 24000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)

    wave_tensor = torch.from_numpy(audio).float()
    mel_tensor  = _mel_transform(wave_tensor.unsqueeze(0))
    mel_tensor  = (torch.log(1e-5 + mel_tensor) - _mel_mean) / _mel_std
    mel_tensor  = mel_tensor.to(device)

    with torch.no_grad():
        input_tensor = mel_tensor.unsqueeze(1)
        ref_s = model.style_encoder(input_tensor)
        ref_p = model.predictor_encoder(input_tensor)

    ref = torch.cat([ref_s, ref_p], dim=1)

    # ── Persist ───────────────────────────────────────────────────────────
    torch.save(ref.cpu(), cache_path)
    print(f"  Style cached  → {cache_path}")

    torch.save(ref.cpu(), output_path)
    print(f"  Style saved   → {output_path}  (shape={tuple(ref.shape)})")

    return ref


# ═════════════════════════════════════════════════════════════════════════════
#  ONNX export functions  (UNCHANGED from original)
# ═════════════════════════════════════════════════════════════════════════════

def export_bert(model, repo_path):
    bmodel = model
    origbert = bmodel.bert
    origbert.eval().to("cpu")

    model_path = os.path.join(repo_path, "bert", "1", "model.onnx")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    batch = 1
    seq_len = 57
    tokens_test = torch.randint(0, 178, (batch, seq_len), dtype=torch.long)
    attention_mask = torch.ones((batch, seq_len), dtype=torch.long)

    tokens_np = tokens_test.cpu().numpy().astype(np.int64)
    mask_np   = attention_mask.cpu().numpy().astype(np.int64)

    with torch.no_grad():
        torch_out = origbert(tokens_test, attention_mask)
    torch_out_np = torch_out.numpy()

    try:
        torch.onnx.export(
            origbert,
            (tokens_test, attention_mask),
            model_path,
            input_names=["TOKENS", "ATTENTION_MASK"],
            output_names=["BERT_DUR"],
            dynamic_axes={
                "TOKENS":         {0: "batch", 1: "seq_len"},
                "ATTENTION_MASK": {0: "batch", 1: "seq_len"},
                "BERT_DUR":       {0: "batch", 1: "seq_len"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
        print("PLBERT exported:", model_path)
    except Exception as e:
        print(traceback.print_exc())
        print(f"Failed to export bert: {e}")

    try:
        model_onnx = onnx.load(model_path)
        onnx.save(onnx.shape_inference.infer_shapes(model_onnx), model_path)

        session = ort.InferenceSession(model_path)
        print("ONNX input names:",  [i.name for i in session.get_inputs()])
        print("ONNX output names:", [o.name for o in session.get_outputs()])

        onnx_out = session.run(None, {"TOKENS": tokens_np, "ATTENTION_MASK": mask_np})[0]
        np.testing.assert_allclose(torch_out_np, onnx_out, rtol=1e-02, atol=1e-04)
        print("ONNX outputs match PyTorch for bert.\n")
    except Exception as e:
        print(traceback.print_exc())
        print(f"Failed to verify bert: {e}")


def export_text_encoder(model, repo_path, model_params):
    bmodel = model
    text_encoder = bmodel.text_encoder.eval().to("cpu")

    model_path = os.path.join(repo_path, "text_encoder", "1", "model.onnx")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    batch   = 1
    seq_len = 74
    tokens  = torch.randint(0, model_params.n_token, (batch, seq_len), dtype=torch.long)
    lengths = torch.tensor([seq_len], dtype=torch.long)
    mask    = text_encoder.length_to_mask(lengths)

    tokens_np  = tokens.numpy().astype(np.int64)
    lengths_np = lengths.numpy().astype(np.int64)
    mask_np    = mask.numpy().astype(np.bool_)

    with torch.no_grad():
        torch_out = text_encoder(tokens, lengths, mask)
    torch_out_np = torch_out.numpy()
    print("Torch output shape:", torch_out_np.shape)

    try:
        torch.onnx.export(
            text_encoder,
            (tokens, lengths, mask),
            model_path,
            input_names=["TOKENS", "INPUT_LENGTHS", "TEXT_MASK"],
            output_names=["T_EN"],
            dynamic_axes={
                "TOKENS":        {0: "batch", 1: "seq_len"},
                "INPUT_LENGTHS": {0: "batch"},
                "TEXT_MASK":     {0: "batch", 1: "seq_len"},
                "T_EN":          {0: "batch", 2: "seq_len"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
        print("text_encoder exported:", model_path)
    except Exception as e:
        print(traceback.print_exc())
        print(f"Failed to export text_encoder: {e}")

    try:
        model_onnx = onnx.load(model_path)
        onnx.save(onnx.shape_inference.infer_shapes(model_onnx), model_path)

        session = ort.InferenceSession(model_path)
        print("ONNX input names:",  [i.name for i in session.get_inputs()])
        print("ONNX output names:", [o.name for o in session.get_outputs()])

        onnx_out = session.run(
            None, {"TOKENS": tokens_np, "INPUT_LENGTHS": lengths_np, "TEXT_MASK": mask_np}
        )[0]
        np.testing.assert_allclose(torch_out_np, onnx_out, rtol=1e-03, atol=1e-05)
        print("ONNX outputs match PyTorch for text_encoder.\n")
    except Exception as e:
        print(traceback.print_exc())
        print(f"Failed to verify text_encoder: {e}")


def export_bert_encoder(model, repo_path):
    bmodel = model
    bert_encoder = bmodel.bert_encoder
    bert_encoder.eval().to("cpu")

    model_path = os.path.join(repo_path, "bert_encoder", "1", "model.onnx")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    batch      = 1
    seq_len    = 74
    hidden_dim = 768
    dummy_input = torch.randn(batch, seq_len, hidden_dim, dtype=torch.float32)
    dummy_input_np = dummy_input.cpu().numpy()

    wrapper = BertEncoderWrapper(bert_encoder).half()

    with torch.no_grad():
        torch_out = wrapper(dummy_input)
    torch_out_np = torch_out.numpy()
    print("Torch output shape:", torch_out_np.shape)

    try:
        torch.onnx.export(
            wrapper,
            dummy_input,
            model_path,
            export_params=True,
            input_names=["BERT_DUR"],
            output_names=["D_EN"],
            dynamic_axes={
                "BERT_DUR": {0: "batch", 1: "seq_len"},
                "D_EN":     {0: "batch", 2: "seq_len"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
        print("bert_encoder exported:", model_path)
    except Exception as e:
        print(traceback.print_exc())
        print(f"Failed to export bert_encoder: {e}")

    try:
        model_onnx = onnx.load(model_path)
        onnx.save(onnx.shape_inference.infer_shapes(model_onnx), model_path)

        session = ort.InferenceSession(model_path)
        print("ONNX input names:",  [i.name for i in session.get_inputs()])
        print("ONNX output names:", [o.name for o in session.get_outputs()])

        onnx_out = session.run(None, {"BERT_DUR": dummy_input_np})[0]
        print("ONNX output shape:", onnx_out.shape)
        np.testing.assert_allclose(torch_out_np, onnx_out, rtol=1e-02, atol=1e-02)
        print("ONNX outputs match PyTorch for bert_encoder.\n")
    except Exception as e:
        print(traceback.print_exc())
        print(f"Failed to verify bert_encoder: {e}")


def export_prosody_text_encoder(model, repo_path):
    bmodel = model
    predictor_text_encoder = bmodel.predictor.text_encoder.eval().to("cpu")

    model_path = os.path.join(repo_path, "prosody_text_encoder", "1", "model.onnx")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    batch      = 1
    seq_len    = 72
    hidden_dim = 512

    d_en         = torch.randn(batch, hidden_dim, seq_len, dtype=torch.float16)
    s            = torch.randn(batch, 128, dtype=torch.float32)
    input_lengths = torch.tensor([seq_len], dtype=torch.int64)
    text_mask    = torch.zeros(batch, seq_len, dtype=torch.bool)

    d_en_np         = d_en.cpu().numpy()
    s_np            = s.cpu().numpy()
    input_lengths_np = input_lengths.cpu().numpy()
    text_mask_np    = text_mask.cpu().numpy()

    with torch.no_grad():
        torch_out = predictor_text_encoder(d_en, s, text_mask)
    torch_out_np = torch_out.cpu().numpy()
    print("Torch output shape:", torch_out_np.shape)

    try:
        torch.onnx.export(
            predictor_text_encoder,
            (d_en, s, text_mask),
            model_path,
            input_names=["D_EN", "S_OUT", "TEXT_MASK_PRED"],
            output_names=["D_OUT"],
            dynamic_axes={
                "D_EN":           {0: "batch", 2: "seq_len"},
                "S_OUT":          {0: "batch"},
                "TEXT_MASK_PRED": {0: "batch", 1: "seq_len"},
                "D_OUT":          {0: "batch", 1: "seq_len"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
        print("prosody_text_encoder exported:", model_path)
    except Exception as e:
        print(traceback.print_exc())
        print(f"Failed to export prosody_text_encoder: {e}")

    try:
        model_onnx = onnx.load(model_path)
        onnx.save(onnx.shape_inference.infer_shapes(model_onnx), model_path)

        session = ort.InferenceSession(model_path)
        print("ONNX input names:",  [i.name for i in session.get_inputs()])
        print("ONNX output names:", [o.name for o in session.get_outputs()])

        onnx_out = session.run(
            None, {"D_EN": d_en_np, "S_OUT": s_np, "TEXT_MASK_PRED": text_mask_np}
        )[0]
        print("ONNX output shape:", onnx_out.shape)
        np.testing.assert_allclose(torch_out_np, onnx_out, rtol=1e-01, atol=1e-02)
        print("ONNX outputs match PyTorch for prosody_text_encoder.\n")
    except Exception as e:
        print(traceback.print_exc())
        print(f"Failed to verify prosody_text_encoder: {e}")


def export_prosody_lstm(model, repo_path):
    bmodel = model
    pred_lstm = bmodel.predictor.lstm.eval().to("cpu").float()

    model_path = os.path.join(repo_path, "prosody_lstm", "1", "model.onnx")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    batch   = 1
    seq_len = 74
    d       = torch.randn(batch, seq_len, 640, dtype=torch.float32)
    d_np    = d.cpu().numpy()

    wrapped_model = PredLSTMWrapper(pred_lstm).to("cpu").eval()

    with torch.no_grad():
        torch_out = wrapped_model(d)
    torch_out_np = torch_out.cpu().numpy()

    try:
        torch.onnx.export(
            wrapped_model,
            d,
            model_path,
            input_names=["D_OUT"],
            output_names=["X_OUT"],
            dynamic_axes={
                "D_OUT": {0: "batch", 1: "seq_len"},
                "X_OUT": {0: "batch", 1: "seq_len"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
        print("prosody_lstm exported:", model_path)
    except Exception as e:
        print(traceback.print_exc())
        print(f"Failed to export prosody_lstm: {e}")

    try:
        model_onnx = onnx.load(model_path)
        onnx.save(onnx.shape_inference.infer_shapes(model_onnx), model_path)

        session = ort.InferenceSession(model_path)
        print("ONNX input names:",  [i.name for i in session.get_inputs()])
        print("ONNX output names:", [o.name for o in session.get_outputs()])

        onnx_out = session.run(["X_OUT"], {"D_OUT": d_np})[0]
        print("ONNX output shape:", onnx_out.shape)
        np.testing.assert_allclose(torch_out_np, onnx_out, rtol=1e-03, atol=1e-05)
        print("ONNX outputs match PyTorch for prosody_lstm.\n")
    except Exception as e:
        print(traceback.print_exc())
        print(f"Failed to verify prosody_lstm: {e}")


def export_prosody_dur_proj(model, repo_path):
    bmodel = model
    pred_dur_proj = bmodel.predictor.duration_proj.eval().to("cpu").half()

    model_path = os.path.join(repo_path, "prosody_dur_proj", "1", "model.onnx")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    batch   = 1
    seq_len = 74
    x       = torch.randn(batch, seq_len, 512, dtype=torch.float16)
    x_np    = x.cpu().numpy()

    with torch.no_grad():
        torch_out = pred_dur_proj(x)
    torch_out_np = torch_out.cpu().numpy()

    try:
        torch.onnx.export(
            pred_dur_proj,
            x,
            model_path,
            input_names=["X_OUT"],
            output_names=["DURATION"],
            dynamic_axes={
                "X_OUT":    {0: "batch", 1: "seq_len"},
                "DURATION": {0: "batch", 1: "seq_len"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
        print("prosody_dur_proj exported:", model_path)
    except Exception as e:
        print(traceback.print_exc())
        print(f"Failed to export prosody_dur_proj: {e}")

    try:
        model_onnx = onnx.load(model_path)
        onnx.save(onnx.shape_inference.infer_shapes(model_onnx), model_path)

        session = ort.InferenceSession(model_path)
        print("ONNX input names:",  [i.name for i in session.get_inputs()])
        print("ONNX output names:", [o.name for o in session.get_outputs()])

        onnx_out = session.run(["DURATION"], {"X_OUT": x_np})[0]
        print("ONNX output shape:", onnx_out.shape)
        np.testing.assert_allclose(torch_out_np, onnx_out, rtol=1e-02, atol=1e-04)
        print("ONNX outputs match PyTorch for prosody_dur_proj.\n")
    except Exception as e:
        print(traceback.print_exc())
        print(f"Failed to verify prosody_dur_proj: {e}")


def export_merged_prosody(model, repo_path):
    lstm      = model.predictor.lstm.eval().to("cpu").float()
    dur_proj  = model.predictor.duration_proj.eval().to("cpu").float()
    merged    = ProsodyPredictorMerged(lstm, dur_proj)

    model_path = os.path.join(repo_path, "prosody_merged", "1", "model.onnx")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    dummy = torch.randn(1, 74, 640, dtype=torch.float32)
    torch.onnx.export(
        merged, dummy, model_path,
        input_names=["D_IN"], output_names=["DURATION"],
        dynamic_axes={"D_IN": {0: "batch", 1: "seq_len"}, "DURATION": {0: "batch", 1: "seq_len"}},
        opset_version=17, do_constant_folding=True,
    )
    print(f"Merged Prosody model exported to: {model_path}")


def export_prosody_predictor_ftrain(model, repo_path):
    bmodel = model
    prosody_model = bmodel.predictor.eval().to("cpu")

    model_path = os.path.join(repo_path, "prosody_predictor_ftrain", "1", "model.onnx")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    batch      = 1
    seq_len    = 302
    hidden_dim = 640
    style_dim  = 128

    en = torch.randn(batch, hidden_dim, seq_len, dtype=torch.float32)
    s  = torch.randn(batch, style_dim,          dtype=torch.float32)

    wrapper = ProsodyPredictorONNXWrapper(prosody_model).eval()

    with torch.no_grad():
        F0_torch, N_torch = wrapper(en, s)

    F0_np = F0_torch.cpu().numpy()
    N_np  = N_torch.cpu().numpy()

    try:
        torch.onnx.export(
            wrapper, (en, s), model_path,
            input_names=["EN", "S_OUT"],
            output_names=["F0_PRED", "N_PRED"],
            dynamic_axes={
                "EN":     {0: "batch", 2: "seq_len"},
                "S_OUT":  {0: "batch"},
                "F0_PRED":{0: "batch", 1: "seq_len"},
                "N_PRED": {0: "batch", 1: "seq_len"},
            },
            opset_version=17, do_constant_folding=True,
        )
        print("prosody_predictor_ftrain exported:", model_path)
    except Exception as e:
        print(traceback.print_exc())
        print(f"Failed to export prosody_predictor_ftrain: {e}")

    try:
        model_onnx = onnx.load(model_path)
        onnx.save(onnx.shape_inference.infer_shapes(model_onnx), model_path)

        session = ort.InferenceSession(model_path)
        print("ONNX input names:",  [i.name for i in session.get_inputs()])
        print("ONNX output names:", [o.name for o in session.get_outputs()])

        F0_onnx, N_onnx = session.run(["F0_PRED", "N_PRED"], {"EN": en.numpy(), "S_OUT": s.numpy()})
        print("ONNX output shapes:", F0_onnx.shape, N_onnx.shape)
        np.testing.assert_allclose(F0_np, F0_onnx, rtol=1e-03, atol=1e-05)
        np.testing.assert_allclose(N_np,  N_onnx,  rtol=1e-03, atol=1e-05)
        print("ONNX outputs match PyTorch for prosody_predictor_ftrain.\n")
    except Exception as e:
        print(traceback.print_exc())
        print(f"Failed to verify prosody_predictor_ftrain: {e}")


def export_intsteps1(repo_path):
    device = "cpu"
    blend  = IntermediateSteps1().to(device)

    model_path = os.path.join(repo_path, "int_steps_1", "1", "model.onnx")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    alpha   = torch.tensor([0.3], dtype=torch.float32, device=device)
    beta    = torch.tensor([0.7], dtype=torch.float32, device=device)
    s_pred  = torch.randn(1, 256, device=device)
    ref_s   = torch.randn(1, 256, device=device)
    t_en    = torch.rand(0, 512, 137, device=device)

    with torch.no_grad():
        ref, s_out, text_mask_pred = blend(alpha, beta, s_pred, ref_s, t_en)

    ref_np           = ref.cpu().numpy()
    s_out_np         = s_out.cpu().numpy()
    text_mask_pred_np = text_mask_pred.cpu().numpy()

    try:
        torch.onnx.export(
            blend,
            (alpha, beta, s_pred, ref_s, t_en),
            model_path,
            input_names=["ALPHA", "BETA", "S_PRED", "REF_S", "T_EN"],
            output_names=["REF", "S_OUT", "TEXT_MASK_PRED"],
            dynamic_axes={
                "ALPHA":          {0: "batch"},
                "BETA":           {0: "batch"},
                "S_PRED":         {0: "batch"},
                "REF_S":          {0: "batch"},
                "T_EN":           {0: "batch", 2: "seq_len"},
                "REF":            {0: "batch"},
                "S_OUT":          {0: "batch"},
                "TEXT_MASK_PRED": {0: "batch", 1: "seq_len"},
            },
            opset_version=17,
        )
        print("int_steps_1 exported:", model_path)
    except Exception as e:
        print(traceback.print_exc())
        print(f"Failed to export int_steps_1: {e}")

    try:
        model_onnx = onnx.load(model_path)
        onnx.save(onnx.shape_inference.infer_shapes(model_onnx), model_path)

        session = ort.InferenceSession(model_path)
        print("ONNX input names:",  [i.name for i in session.get_inputs()])
        print("ONNX output names:", [o.name for o in session.get_outputs()])

        ref_onnx, s_out_onnx, tmask_onnx = session.run(None, {
            "ALPHA": alpha.numpy(), "BETA": beta.numpy(),
            "S_PRED": s_pred.numpy(), "REF_S": ref_s.numpy(), "T_EN": t_en.numpy(),
        })
        np.testing.assert_allclose(ref_np,            ref_onnx,   rtol=1e-03, atol=1e-05)
        np.testing.assert_allclose(s_out_np,          s_out_onnx, rtol=1e-03, atol=1e-05)
        np.testing.assert_allclose(text_mask_pred_np, tmask_onnx, rtol=1e-03, atol=1e-05)
        print("ONNX outputs match PyTorch for int_steps_1.\n")
    except Exception as e:
        print(traceback.print_exc())
        print(f"Failed to verify int_steps_1: {e}")


def export_diffusion_steps(model, repo_path):
    bmodel    = model
    diffusion = bmodel.diffusion
    diffusion.eval().to("cpu")

    diffusion = DiffusionONNX(
        unet=diffusion.diffusion.net,
        sigma_data=diffusion.diffusion.sigma_data,
        sigma_min=1e-4,
        sigma_max=3.0,
        rho=9.0,
        clamp=False,
    ).eval().to("cpu")

    for p in diffusion.parameters():
        p.requires_grad_(False)

    model_path = os.path.join(repo_path, "diffusion_steps", "1", "model.onnx")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    batch      = 1
    seq_len    = 74
    hidden_dim = 768

    bert_dur  = torch.randn(batch, seq_len, hidden_dim, dtype=torch.float32)
    ref_s     = torch.randn(batch, 256,                 dtype=torch.float32)
    num_steps = torch.tensor([3], dtype=torch.int32)

    try:
        torch.onnx.export(
            diffusion,
            (num_steps, bert_dur, ref_s),
            model_path,
            export_params=True,
            opset_version=17,
            input_names=["NUM_STEPS", "BERT_DUR", "REF_S"],
            output_names=["S_PRED"],
            dynamic_axes={
                "BERT_DUR": {0: "batch_size", 1: "seq_len"},
                "REF_S":    {0: "batch_size"},
            },
        )
        print("diffusion_steps exported:", model_path)
    except Exception as e:
        print(traceback.print_exc())
        print(f"Failed to export diffusion_steps: {e}")

    try:
        model_onnx  = onnx.load(model_path)
        onnx.save(onnx.shape_inference.infer_shapes(model_onnx), model_path)

        model_onnx  = onnx.load(model_path)
        inferred    = SymbolicShapeInference.infer_shapes(model_onnx, auto_merge=True)
        onnx.save(inferred, model_path)

        session = ort.InferenceSession(model_path)
        print("ONNX input names:",  [i.name for i in session.get_inputs()])
        print("ONNX output names:", [o.name for o in session.get_outputs()])

        onnx_out = session.run(
            ["S_PRED"],
            {"NUM_STEPS": num_steps.numpy(), "BERT_DUR": bert_dur.numpy(), "REF_S": ref_s.numpy()},
        )[0]
        print("ONNX output shape:", onnx_out.shape)
    except Exception as e:
        print(traceback.print_exc())
        print(f"Failed to verify diffusion_steps: {e}")


def export_prep_decoder_dynamic(model, repo_path):
    bmodel      = model
    prep_decoder = bmodel.prep_decoder.eval().cuda()

    model_path = os.path.join(repo_path, "prep_decoder", "1", "model.onnx")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    batch   = 2
    seq_len = 40
    F0_pred = torch.randn(batch, seq_len * 2, dtype=torch.float32).cuda()
    N_pred  = torch.randn(batch, seq_len * 2, dtype=torch.float32).cuda()

    try:
        torch.onnx.export(
            prep_decoder,
            (F0_pred, N_pred),
            model_path,
            input_names=["F0_PRED", "N_PRED"],
            output_names=["F0_OUT", "N_OUT"],
            dynamic_axes={
                "F0_PRED": {0: "batch", 1: "seq_len"},
                "N_PRED":  {0: "batch", 1: "seq_len"},
                "F0_OUT":  {0: "batch", 2: "seq_len"},
                "N_OUT":   {0: "batch", 2: "seq_len"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
        print("prep_decoder exported:", model_path)
    except Exception as e:
        print(traceback.print_exc())
        print(f"Failed to export prep_decoder: {e}")

    try:
        model_onnx = onnx.load(model_path)
        onnx.save(onnx.shape_inference.infer_shapes(model_onnx), model_path)

        session = ort.InferenceSession(model_path)
        print("ONNX input names:",  [i.name for i in session.get_inputs()])
        print("ONNX output names:", [o.name for o in session.get_outputs()])

        onnx_out = session.run(
            None, {"F0_PRED": F0_pred.cpu().numpy(), "N_PRED": N_pred.cpu().numpy()}
        )[0]
        print("ONNX output shape:", onnx_out.shape)
    except Exception as e:
        print(traceback.print_exc())
        print(f"Failed to verify prep_decoder: {e}")


def export_decoder_dynamic(model, repo_path):
    bmodel  = model
    decoder = bmodel.decoder.eval().cuda()

    model_path = os.path.join(repo_path, "decoder", "1", "model.onnx")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    batch   = 2
    seq_len = 40

    asr     = torch.randn(batch, 512, seq_len, dtype=torch.float32).cuda()
    F0_pred = torch.randn(batch, 1,   seq_len, dtype=torch.float32).cuda()
    N_pred  = torch.randn(batch, 1,   seq_len, dtype=torch.float32).cuda()
    ref     = torch.randn(batch, 128,          dtype=torch.float32).cuda()

    print(asr.shape, F0_pred.shape, N_pred.shape, ref.shape)

    try:
        torch.onnx.export(
            decoder,
            (asr, F0_pred, N_pred, ref),
            model_path,
            input_names=["ASR", "F0_OUT", "N_OUT", "REF"],
            output_names=["X_OUT"],
            dynamic_axes={
                "ASR":   {0: "batch", 2: "seq_len"},
                "F0_OUT":{0: "batch", 2: "seq_len"},
                "N_OUT": {0: "batch", 2: "seq_len"},
                "REF":   {0: "batch"},
                "X_OUT": {0: "batch", 2: "seq_len"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
        print("decoder exported:", model_path)
    except Exception as e:
        print(traceback.print_exc())
        print(f"Failed to export decoder: {e}")

    try:
        model_onnx = onnx.load(model_path)
        onnx.save(onnx.shape_inference.infer_shapes(model_onnx), model_path)

        session = ort.InferenceSession(model_path)
        print("ONNX input names:",  [i.name for i in session.get_inputs()])
        print("ONNX output names:", [o.name for o in session.get_outputs()])

        onnx_out = session.run(None, {
            "ASR":    asr.cpu().numpy(),
            "F0_OUT": F0_pred.cpu().numpy(),
            "N_OUT":  N_pred.cpu().numpy(),
            "REF":    ref.cpu().numpy(),
        })[0]
        print("ONNX output shape:", onnx_out.shape)
    except Exception as e:
        print(traceback.print_exc())
        print(f"Failed to verify decoder: {e}")


def export_generator_dynamic(model, repo_path):
    bmodel    = model
    generator = bmodel.generator.eval().cuda()

    model_path = os.path.join(repo_path, "generator", "1", "model.onnx")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    batch   = 2
    seq_len = 200

    x       = torch.randn(batch, 512, seq_len, dtype=torch.float32).cuda()
    F0_pred = torch.randn(batch, seq_len,       dtype=torch.float32).cuda()
    ref     = torch.randn(batch, 128,           dtype=torch.float32).cuda()

    try:
        torch.onnx.export(
            generator,
            (x, ref, F0_pred),
            model_path,
            input_names=["X_OUT", "REF", "F0_PRED"],
            output_names=["AUDIO_OUT"],
            dynamic_axes={
                "X_OUT":    {0: "batch", 2: "seq_len"},
                "F0_PRED":  {0: "batch", 1: "seq_len"},
                "REF":      {0: "batch"},
                "AUDIO_OUT":{0: "batch", 2: "seq_len"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
        print("generator exported:", model_path)
    except Exception as e:
        print(traceback.print_exc())
        print(f"Failed to export generator: {e}")

    try:
        model_onnx = onnx.load(model_path)
        onnx.save(onnx.shape_inference.infer_shapes(model_onnx), model_path)

        session = ort.InferenceSession(model_path)
        print("ONNX input names:",  [i.name for i in session.get_inputs()])
        print("ONNX output names:", [o.name for o in session.get_outputs()])

        onnx_out = session.run(None, {
            "X_OUT":   x.cpu().numpy(),
            "F0_PRED": F0_pred.cpu().numpy(),
            "REF":     ref.cpu().numpy(),
        })[0]
        print("ONNX output shape:", onnx_out.shape)
    except Exception as e:
        print(traceback.print_exc())
        print(f"Failed to verify generator: {e}")


def get_speakers_with_same_model(all_configs, target_speaker):
    target_model = all_configs[target_speaker]["model"]

    same_model_speakers = {
        spk: cfg for spk, cfg in all_configs.items()
        if cfg["model"] == target_model
    }

    return target_model, same_model_speakers



def build_registry_from_local(
    base_dir="raw_models",
    output_path="model_registry.json"
):
    models_dir = os.path.join(base_dir, "models")
    refs_dir   = os.path.join(base_dir, "references")
    cfg_dir    = os.path.join(base_dir, "configs")

    registry = {}

    if not os.path.exists(models_dir):
        print("No models directory found ❌")
        return

    for model_id in os.listdir(models_dir):
        model_path = os.path.join(models_dir, model_id, "model.pth")
        config_path = os.path.join(cfg_dir, model_id, "config.yml")
        ref_folder = os.path.join(refs_dir, model_id)

        if not os.path.exists(model_path):
            print(f"[SKIP] No model for {model_id}")
            continue

        # ── Collect speakers ─────────────────────
        speakers = {}

        if os.path.exists(ref_folder):
            for file in os.listdir(ref_folder):
                if file.endswith(".wav"):
                    speaker_name = os.path.splitext(file)[0]
                    speakers[speaker_name] = os.path.join(ref_folder, file)

        registry[model_id] = {
            "model_path": model_path,
            "config_path": config_path if os.path.exists(config_path) else None,
            "speakers": speakers
        }

    # ── Save ────────────────────────────────────
    with open(output_path, "w") as f:
        json.dump(registry, f, indent=2)

    print(f"\n✅ Registry built from local files → {output_path}")


# ═════════════════════════════════════════════════════════════════════════════
#  Main entry-point
# ═════════════════════════════════════════════════════════════════════════════

def run_full_export_pipeline(bmodel, OUTPUT_REPO, model_params):
    """Run all ONNX export functions in order."""
    export_intsteps1(OUTPUT_REPO)
    export_bert(bmodel, OUTPUT_REPO)
    export_text_encoder(bmodel, OUTPUT_REPO, model_params)
    export_bert_encoder(bmodel, OUTPUT_REPO)
    export_prosody_text_encoder(bmodel, OUTPUT_REPO)
    export_prosody_lstm(bmodel, OUTPUT_REPO)
    export_prosody_dur_proj(bmodel, OUTPUT_REPO)
    export_prosody_predictor_ftrain(bmodel, OUTPUT_REPO)
    export_diffusion_steps(bmodel, OUTPUT_REPO)
    export_prep_decoder_dynamic(bmodel, OUTPUT_REPO)
    export_decoder_dynamic(bmodel, OUTPUT_REPO)
    export_generator_dynamic(bmodel, OUTPUT_REPO)



def main():
    parser = argparse.ArgumentParser(description="TTS Model Builder & ONNX Exporter")
    parser.add_argument("--speaker", required=True)
    parser.add_argument("--diffusion_module", required=False, default=True)

    args = parser.parse_args()
    speaker_name = args.speaker
    diffusion_module = args.diffusion_module

    print(f"\n{'='*60}")
    print(f"  Speaker : {speaker_name}")
    print(f"{'='*60}\n")

    # update_model_registry(SPEAKER_CONFIGS_FILE)

    # ── 1. Load configs ─────────────────────────────
    all_speaker_configs = load_speaker_configs(SPEAKER_CONFIGS_FILE)

    if speaker_name not in all_speaker_configs:
        raise KeyError(f"{speaker_name} not found")

    # ── 2. Get all speakers with SAME MODEL ────────
    # model_id, speakers_group = get_speakers_with_same_model(
    #     all_speaker_configs,
    #     speaker_name
    # )

    speaker_lang_raw = load_speaker_language("speaker_languages.json")
    speaker_to_model = flatten_speaker_language(speaker_lang_raw)

    if speaker_name not in speaker_to_model:
        raise KeyError(f"{speaker_name} not found in speaker_languages.json")

    # 🔥 THIS is your new model_id
    model_id = speaker_to_model[speaker_name]


    speakers_group = {
        spk: cfg for spk, cfg in all_speaker_configs.items()
        if speaker_to_model.get(spk) == model_id
    }


    print(f"Model ID (clean name): {model_id}")

    print(f"\nTarget Model: {model_id}")
    print(f"Speakers: {list(speakers_group.keys())}")

    # ── 3. Download all (model shared, refs per speaker) ─────
    model_path = None
    config_path = None
    speaker_ref_map = {}

    for spk, cfg in speakers_group.items():
        print(f"\nDownloading: {spk}")

        m_path, r_path, c_path = smart_download_by_model(
            cfg, spk, model_id
        )

        model_path = m_path        # same for all
        config_path = c_path       # same for all
        speaker_ref_map[spk] = r_path

    # ── 4. Triton repo per MODEL ───────────────
    OUTPUT_REPO = os.path.join(TRITON_REPOS_DIR, model_id, "triton_model_repository")
    reference_dest_folder = os.path.join(TRITON_REPOS_DIR, model_id)

    os.makedirs(OUTPUT_REPO, exist_ok=True)

    # ── 5. Skip full pipeline if already done ───
    onnx_ready = all_onnx_exports_exist(OUTPUT_REPO)

    if diffusion_module:
        copy_triton_contents(TRITON_TEMPLATE_DIR, OUTPUT_REPO)

    # ── 6. Build model ONCE ─────────────────────
    plbert = load_plbert_only(PLBERT_DIR)

    config = yaml.safe_load(open(config_path))
    model_params = recursive_munch(config["model_params"])

    bmodel = build_model(model_params, plbert)

    device = "cuda:0"
    _ = [bmodel[k].to(device).eval() for k in bmodel]

    state = torch.load(model_path, map_location="cpu")["net"]

    # --- load weights (same as your code) ---
    prep_sd = OrderedDict()
    for k, v in state["decoder"].items():
        if k.startswith("module.F0_conv.") or k.startswith("module.N_conv."):
            prep_sd[k[7:]] = v
    bmodel["prep_decoder"].load_state_dict(prep_sd, strict=True)

    decoder_sd = OrderedDict()
    for k, v in state["decoder"].items():
        if not (k.startswith("module.F0_conv.") or k.startswith("module.N_conv.")):
            decoder_sd[k[7:]] = v

    bmodel["decoder"].load_state_dict(decoder_sd, strict=False)
    bmodel["generator"].load_state_dict(decoder_sd, strict=False)

    for key in bmodel:
        if key in ("prep_decoder", "decoder"):
            continue
        if key in state:
            try:
                bmodel[key].load_state_dict(state[key])
            except:
                new_sd = OrderedDict()
                for k, v in state[key].items():
                    name = k[7:] if k.startswith("module.") else k
                    new_sd[name] = v
                bmodel[key].load_state_dict(new_sd, strict=False)

    print("Model loaded ✅")

    # ── 7. ONNX export (ONLY ONCE per model) ─────
    if not onnx_ready:
        print("[ONNX] Exporting...")
        run_full_export_pipeline(bmodel, OUTPUT_REPO, model_params)
    else:
        print("[ONNX] Already exists ✅")

    # ── 8. Generate styles for ALL speakers ─────
    for spk, ref_path in speaker_ref_map.items():

        style_path = os.path.join(
            reference_dest_folder,
            f"reference_style_{spk}.pt"
        )

        if os.path.exists(style_path):
            print(f"[Style] {spk} already exists ✅")
            continue

        print(f"[Style] Generating for {spk}")

        create_reference_style(
            bmodel,
            ref_path,
            reference_dest_folder,
            spk
        )

    build_registry_from_local(base_dir="raw_models", output_path="model_registry.json")

    # ── 9. Summary ─────────────────────────────
    print(f"\n{'='*60}")
    print(f"  MODEL ID     : {model_id}")
    print(f"  Speakers     : {len(speaker_ref_map)}")
    print(f"  Triton repo  : {OUTPUT_REPO}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main() 