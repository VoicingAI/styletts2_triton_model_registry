"""What to export, and in what order.

Each :class:`ExportSpec` in :data:`EXPORTERS` names one Triton model and
supplies only what differs from every other export: the example inputs, the
graph's input/output names, which axes stay dynamic, and how closely the result
has to match PyTorch. The mechanism that consumes a spec lives in
:mod:`styletts2_export.onnx.harness`.

Order matters: :func:`run_exports` walks ``EXPORTERS`` in sequence, which is
the order the Triton ensemble consumes them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import torch

from .harness import ExportOutcome, ExportSpec, export_module, onnx_path_for
from .wrappers import (
    BertEncoderWrapper,
    DiffusionONNX,
    IntermediateSteps1,
    PredLSTMWrapper,
    ProsodyPredictorONNXWrapper,
)

__all__ = ["EXPORTERS", "run_exports"]

# ─────────────────────────────────────────────────────────────────────────────
#  Example-input builders
#
#  Each returns (module_to_export, example_inputs). Shapes here only seed the
#  trace — every axis listed in the spec's `dynamic_axes` stays symbolic in the
#  emitted graph. They are picked to be small but non-degenerate: a batch or
#  sequence length of 0 or 1 lets the tracer constant-fold shape arithmetic
#  that must stay dynamic.
# ─────────────────────────────────────────────────────────────────────────────

_SEQ = 74          # representative phoneme-sequence length
_FRAMES = 200      # representative mel-frame count
_STYLE = 128       # per-half style dimension (acoustic | prosodic)
_HIDDEN = 512
_BERT_HIDDEN = 768
_BATCH = 2         # >1 so the tracer cannot fold the batch axis away


def _build_int_steps_1(model, ctx):
    style = ctx["style"]
    blend = IntermediateSteps1()
    alpha = torch.full((_BATCH,), style.alpha, dtype=torch.float32)
    beta = torch.full((_BATCH,), style.beta, dtype=torch.float32)
    s_pred = torch.randn(_BATCH, _STYLE * 2)
    ref_s = torch.randn(_BATCH, _STYLE * 2)
    t_en = torch.randn(_BATCH, _HIDDEN, _SEQ)
    return blend, (alpha, beta, s_pred, ref_s, t_en)


def _build_bert(model, ctx):
    n_token = int(ctx["model_params"].n_token)
    tokens = torch.randint(0, n_token, (1, _SEQ), dtype=torch.long)
    attention_mask = torch.ones((1, _SEQ), dtype=torch.long)
    return model["bert"], (tokens, attention_mask)


def _build_text_encoder(model, ctx):
    encoder = model["text_encoder"]
    n_token = int(ctx["model_params"].n_token)
    tokens = torch.randint(0, n_token, (1, _SEQ), dtype=torch.long)
    lengths = torch.tensor([_SEQ], dtype=torch.long)
    mask = encoder.length_to_mask(lengths)
    return encoder, (tokens, lengths, mask)


def _build_bert_encoder(model, ctx):
    # Weights run in fp16 (see the spec's dtype); the wrapper casts the fp32
    # input itself, so the graph's BERT_DUR input stays fp32 as Triton declares.
    wrapper = BertEncoderWrapper(model["bert_encoder"])
    hidden = torch.randn(_BATCH, _SEQ, _BERT_HIDDEN, dtype=torch.float32)
    return wrapper, (hidden,)


def _build_prosody_text_encoder(model, ctx):
    # D_EN arrives fp16 from bert_encoder. DurationEncoder keeps fp32 weights;
    # the torch.cat with the fp32 style vector promotes it back on the first op.
    encoder = model["predictor"].text_encoder
    d_en = torch.randn(_BATCH, _HIDDEN, _SEQ, dtype=torch.float16)
    s = torch.randn(_BATCH, _STYLE, dtype=torch.float32)
    text_mask = torch.zeros(_BATCH, _SEQ, dtype=torch.bool)
    return encoder, (d_en, s, text_mask)


def _build_prosody_lstm(model, ctx):
    wrapper = PredLSTMWrapper(model["predictor"].lstm)
    d_out = torch.randn(_BATCH, _SEQ, _HIDDEN + _STYLE, dtype=torch.float32)
    return wrapper, (d_out,)


def _build_prosody_dur_proj(model, ctx):
    # fp16 in, fp16 out -- prosody_lstm emits fp16 X_OUT.
    x = torch.randn(_BATCH, _SEQ, _HIDDEN, dtype=torch.float16)
    return model["predictor"].duration_proj, (x,)


def _build_prosody_predictor(model, ctx):
    wrapper = ProsodyPredictorONNXWrapper(model["predictor"])
    en = torch.randn(_BATCH, _HIDDEN + _STYLE, _FRAMES, dtype=torch.float32)
    s = torch.randn(_BATCH, _STYLE, dtype=torch.float32)
    return wrapper, (en, s)


def _build_diffusion(model, ctx):
    diffusion = model["diffusion"]
    wrapper = DiffusionONNX(
        unet=diffusion.diffusion.net,
        sigma_data=diffusion.diffusion.sigma_data,
        sigma_min=1e-4,
        sigma_max=3.0,
        rho=9.0,
        clamp=False,
        num_steps=ctx["diffusion_num_steps"],
    )
    for parameter in wrapper.parameters():
        parameter.requires_grad_(False)

    bert_dur = torch.randn(1, _SEQ, _BERT_HIDDEN, dtype=torch.float32)
    ref_s = torch.randn(1, _STYLE * 2, dtype=torch.float32)
    return wrapper, (bert_dur, ref_s)


def _build_prep_decoder(model, ctx):
    f0 = torch.randn(_BATCH, _FRAMES, dtype=torch.float32)
    n = torch.randn(_BATCH, _FRAMES, dtype=torch.float32)
    return model["prep_decoder"], (f0, n)


def _build_decoder(model, ctx):
    asr = torch.randn(_BATCH, _HIDDEN, _FRAMES, dtype=torch.float32)
    f0 = torch.randn(_BATCH, 1, _FRAMES, dtype=torch.float32)
    n = torch.randn(_BATCH, 1, _FRAMES, dtype=torch.float32)
    ref = torch.randn(_BATCH, _STYLE, dtype=torch.float32)
    return model["decoder"], (asr, f0, n, ref)


def _build_generator(model, ctx):
    # F0_PRED is per-output-frame here, so its length matches X_OUT's: the
    # decoder's upsampling block already doubled the rate that prep_decoder
    # halved, putting both back at the prosody predictor's frame rate.
    x = torch.randn(_BATCH, _HIDDEN, _FRAMES, dtype=torch.float32)
    ref = torch.randn(_BATCH, _STYLE, dtype=torch.float32)
    f0 = torch.randn(_BATCH, _FRAMES, dtype=torch.float32)
    return model["generator"], (x, ref, f0)


# ─────────────────────────────────────────────────────────────────────────────
#  Registry — order here is the order exports run in.
#
#  Tolerances are per-module because the graphs differ in how much numerical
#  drift they legitimately accumulate: LSTM stacks and the 4-stage upsampling
#  decoder diverge far more than a single Linear.
# ─────────────────────────────────────────────────────────────────────────────

EXPORTERS: Tuple[ExportSpec, ...] = (
    ExportSpec(
        name="int_steps_1",
        build=_build_int_steps_1,
        input_names=("ALPHA", "BETA", "S_PRED", "REF_S", "T_EN"),
        output_names=("REF", "S_OUT", "TEXT_MASK_PRED"),
        dynamic_axes={
            "ALPHA": {0: "batch"},
            "BETA": {0: "batch"},
            "S_PRED": {0: "batch"},
            "REF_S": {0: "batch"},
            "T_EN": {0: "batch", 2: "seq_len"},
            "REF": {0: "batch"},
            "S_OUT": {0: "batch"},
            "TEXT_MASK_PRED": {0: "batch", 1: "seq_len"},
        },
        rtol=1e-3, atol=1e-5,
        do_constant_folding=False,
        notes="Blends diffusion-predicted style with the reference style.",
    ),
    ExportSpec(
        name="bert",
        build=_build_bert,
        input_names=("TOKENS", "ATTENTION_MASK"),
        output_names=("BERT_DUR",),
        dynamic_axes={
            "TOKENS": {0: "batch", 1: "seq_len"},
            "ATTENTION_MASK": {0: "batch", 1: "seq_len"},
            "BERT_DUR": {0: "batch", 1: "seq_len"},
        },
        rtol=1e-2, atol=1e-4,
        notes="PLBERT (ALBERT) phoneme encoder.",
    ),
    ExportSpec(
        name="text_encoder",
        build=_build_text_encoder,
        input_names=("TOKENS", "INPUT_LENGTHS", "TEXT_MASK"),
        output_names=("T_EN",),
        dynamic_axes={
            "TOKENS": {0: "batch", 1: "seq_len"},
            "INPUT_LENGTHS": {0: "batch"},
            "TEXT_MASK": {0: "batch", 1: "seq_len"},
            "T_EN": {0: "batch", 2: "seq_len"},
        },
        rtol=1e-3, atol=1e-5,
    ),
    ExportSpec(
        name="bert_encoder",
        build=_build_bert_encoder,
        input_names=("BERT_DUR",),
        output_names=("D_EN",),
        dynamic_axes={
            "BERT_DUR": {0: "batch", 1: "seq_len"},
            "D_EN": {0: "batch", 2: "seq_len"},
        },
        dtype=torch.float16,
        rtol=1e-2, atol=1e-2,
        notes="Linear 768->512 in fp16, plus transpose to [B, C, T].",
    ),
    ExportSpec(
        name="prosody_text_encoder",
        build=_build_prosody_text_encoder,
        input_names=("D_EN", "S_OUT", "TEXT_MASK_PRED"),
        output_names=("D_OUT",),
        dynamic_axes={
            "D_EN": {0: "batch", 2: "seq_len"},
            "S_OUT": {0: "batch"},
            "TEXT_MASK_PRED": {0: "batch", 1: "seq_len"},
            "D_OUT": {0: "batch", 1: "seq_len"},
        },
        rtol=1e-1, atol=1e-2,
        notes="DurationEncoder: stacked BiLSTM + AdaLayerNorm.",
    ),
    ExportSpec(
        name="prosody_lstm",
        build=_build_prosody_lstm,
        input_names=("D_OUT",),
        output_names=("X_OUT",),
        dynamic_axes={
            "D_OUT": {0: "batch", 1: "seq_len"},
            "X_OUT": {0: "batch", 1: "seq_len"},
        },
        rtol=1e-3, atol=1e-5,
    ),
    ExportSpec(
        name="prosody_dur_proj",
        build=_build_prosody_dur_proj,
        input_names=("X_OUT",),
        output_names=("DURATION",),
        dynamic_axes={
            "X_OUT": {0: "batch", 1: "seq_len"},
            "DURATION": {0: "batch", 1: "seq_len"},
        },
        dtype=torch.float16,
        rtol=1e-2, atol=1e-4,
    ),
    ExportSpec(
        name="prosody_predictor_ftrain",
        build=_build_prosody_predictor,
        input_names=("EN", "S_OUT"),
        output_names=("F0_PRED", "N_PRED"),
        dynamic_axes={
            "EN": {0: "batch", 2: "seq_len"},
            "S_OUT": {0: "batch"},
            "F0_PRED": {0: "batch", 1: "seq_len"},
            "N_PRED": {0: "batch", 1: "seq_len"},
        },
        rtol=1e-3, atol=1e-5,
        notes="F0 and energy prediction (ProsodyPredictor.F0Ntrain).",
    ),
    ExportSpec(
        name="diffusion_steps",
        build=_build_diffusion,
        input_names=("BERT_DUR", "REF_S"),
        output_names=("S_PRED",),
        dynamic_axes={
            "BERT_DUR": {0: "batch", 1: "seq_len"},
            "REF_S": {0: "batch"},
            "S_PRED": {0: "batch"},
        },
        verify="shape",
        symbolic_shape_inference=True,
        notes=(
            "Style diffusion sampler. Stochastic (RandomNormal), so only shapes "
            "are checked. The step count is unrolled into the graph at trace "
            "time -- set export.diffusion_num_steps and re-export to change it."
        ),
    ),
    ExportSpec(
        name="prep_decoder",
        device="auto",
        build=_build_prep_decoder,
        input_names=("F0_PRED", "N_PRED"),
        output_names=("F0_OUT", "N_OUT"),
        dynamic_axes={
            "F0_PRED": {0: "batch", 1: "seq_len"},
            "N_PRED": {0: "batch", 1: "seq_len"},
            "F0_OUT": {0: "batch", 2: "seq_len"},
            "N_OUT": {0: "batch", 2: "seq_len"},
        },
        rtol=1e-3, atol=1e-5,
        notes="The strided F0/N convolutions split out of the decoder.",
    ),
    ExportSpec(
        name="decoder",
        device="auto",
        build=_build_decoder,
        input_names=("ASR", "F0_OUT", "N_OUT", "REF"),
        output_names=("X_OUT",),
        dynamic_axes={
            "ASR": {0: "batch", 2: "seq_len"},
            "F0_OUT": {0: "batch", 2: "seq_len"},
            "N_OUT": {0: "batch", 2: "seq_len"},
            "REF": {0: "batch"},
            "X_OUT": {0: "batch", 2: "seq_len"},
        },
        rtol=2e-2, atol=5e-3,
        notes=(
            "AdaIN residual stack up to (not including) the vocoder. Tolerance "
            "is set by how X_OUT is consumed, not by fp32: it feeds the "
            "generator as an fp16 TensorRT engine, where the representable "
            "step is ~1e-3 relative. Four 1024-channel conv blocks accumulate "
            "fp32 ordering differences of that order between torch and ORT, so "
            "a tighter bound would be measuring below the noise floor of the "
            "precision this tensor is actually computed in."
        ),
    ),
    ExportSpec(
        name="generator",
        device="auto",
        build=_build_generator,
        input_names=("X_OUT", "REF", "F0_PRED"),
        output_names=("AUDIO_OUT",),
        dynamic_axes={
            "X_OUT": {0: "batch", 2: "seq_len"},
            "REF": {0: "batch"},
            "F0_PRED": {0: "batch", 1: "seq_len"},
            "AUDIO_OUT": {0: "batch", 2: "seq_len"},
        },
        verify="shape",
        notes=(
            "HiFi-GAN vocoder. SineGen draws a random initial phase and adds "
            "Gaussian noise, so this graph is stochastic and its values cannot "
            "be compared against PyTorch."
        ),
    ),
)


def run_exports(
    model: Mapping[str, torch.nn.Module],
    repo_dir: Path,
    *,
    module_names: Iterable[str],
    opset: int,
    overwrite: bool,
    verify: bool,
    strict: bool,
    context: Optional[Dict[str, Any]] = None,
) -> list:
    """Export the named modules in dependency order. Returns the outcomes."""
    by_name = {spec.name: spec for spec in EXPORTERS}
    unknown = [n for n in module_names if n not in by_name]
    if unknown:
        raise KeyError(f"no exporter registered for {unknown}")

    wanted = set(module_names)
    outcomes = []
    for spec in EXPORTERS:                       # EXPORTERS defines the order
        if spec.name not in wanted:
            continue

        path = onnx_path_for(repo_dir, spec.name)
        if path.is_file() and not overwrite:
            outcomes.append(ExportOutcome(name=spec.name, path=path, skipped=True))
            print(outcomes[-1].summary())
            continue

        print(f"  export  {spec.name} ...")
        outcome = export_module(
            spec, model, repo_dir, opset=opset, verify=verify, context=context
        )
        outcomes.append(outcome)
        print(outcome.summary())

        if strict and not outcome.ok:
            raise RuntimeError(
                f"export of '{spec.name}' failed: {outcome.error}\n"
                "Set `export.strict: false` in the config to continue past failures."
            )

    return outcomes
