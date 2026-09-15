"""How to export and verify one module.

Every StyleTTS2 sub-model is exported the same way: build example inputs, run
the module under ``torch.no_grad``, export, re-run the ``.onnx`` under
onnxruntime and compare. That loop lives here, once. *What* to export, and in
what order, lives in :mod:`styletts2_export.onnx.pipeline`.

Two things this file is deliberate about:

* ``dynamo=False``. torch >= 2.9 defaults ``torch.onnx.export`` to the dynamo
  exporter, which handles ``dynamic_axes`` and the LSTM/``pack_padded_sequence``
  patterns in this model very differently. These graphs were written for, and
  validated against, the TorchScript exporter; pinning it keeps a torch upgrade
  from silently changing the emitted graph.
* Dtype and device changes are scoped. ``nn.Module.to()`` and ``.half()``
  mutate in place, so the original script left ``bert_encoder`` permanently
  fp16 and made each export depend on the ones before it. :func:`module_as`
  snapshots and restores, so exports are order-independent and repeatable.
"""

from __future__ import annotations

import itertools
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

__all__ = [
    "ExportOutcome",
    "ExportSpec",
    "export_module",
    "module_as",
    "onnx_path_for",
]


# ─────────────────────────────────────────────────────────────────────────────
#  Scoped dtype / device changes
# ─────────────────────────────────────────────────────────────────────────────

@contextmanager
def module_as(module: torch.nn.Module, *, device=None, dtype=None):
    """Temporarily move/cast *module*, restoring its exact tensors on exit.

    A dtype change is lossy (fp32 -> fp16 -> fp32 does not round-trip), so when
    ``dtype`` is given the original tensor data is cloned and copied back.
    A device-only change is exact, so nothing is cloned.
    """
    tensors = list(itertools.chain(module.parameters(), module.buffers()))
    saved = [
        (t, t.device, t.dtype, t.detach().clone() if dtype is not None else None)
        for t in tensors
    ]
    try:
        module.to(device=device, dtype=dtype)
        yield module
    finally:
        for tensor, original_device, original_dtype, original_data in saved:
            if original_data is not None:
                tensor.data = original_data.to(device=original_device)
            else:
                tensor.data = tensor.data.to(device=original_device, dtype=original_dtype)


# ─────────────────────────────────────────────────────────────────────────────
#  Result + spec types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExportOutcome:
    name: str
    path: Path
    exported: bool = False
    verified: Optional[bool] = None       # None when verification was skipped
    max_abs_error: Optional[float] = None
    generalised: Optional[int] = None     # second shape it was re-checked at
    skipped: bool = False
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.skipped or (self.exported and self.verified is not False)

    def summary(self) -> str:
        if self.skipped:
            return f"  skip    {self.name:<26} (already present)"
        if self.error:
            return f"  FAIL    {self.name:<26} {self.error}"
        if self.verified is None:
            return f"  ok      {self.name:<26} (not verified)"
        gen = "" if self.generalised is None else f"  · dynamic @{self.generalised}"
        if self.max_abs_error is None:
            return f"  ok      {self.name:<26} shape + finiteness (stochastic){gen}"
        return f"  ok      {self.name:<26} max|err|={self.max_abs_error:.2e}{gen}"


@dataclass
class ExportSpec:
    """Everything that differs between one module's export and another's."""

    name: str
    input_names: Sequence[str]
    output_names: Sequence[str]
    dynamic_axes: Mapping[str, Mapping[int, str]]
    # (model, ctx) -> (module_to_export, example_inputs)
    build: Callable[..., Tuple[torch.nn.Module, Tuple[torch.Tensor, ...]]]
    rtol: float = 1e-3
    atol: float = 1e-5
    # "cpu" forces CPU tracing; "auto" uses the run's `export.device`. The
    # heavy convolutional graphs trace far faster on GPU, the rest do not care.
    device: str = "cpu"
    dtype: Optional[torch.dtype] = None
    # How far verification can go, which depends on whether the graph is
    # deterministic:
    #   "full"  -- run under onnxruntime and compare values against PyTorch
    #   "shape" -- run it, check output shapes and that nothing is NaN/inf
    #              (all a graph containing a Random* node allows)
    #   "none"  -- export only
    verify: str = "full"
    symbolic_shape_inference: bool = False
    do_constant_folding: bool = True
    notes: str = ""


def onnx_path_for(repo_dir: Path, name: str) -> Path:
    """Triton lays models out as ``<repo>/<model name>/<version>/model.onnx``."""
    return Path(repo_dir) / name / "1" / "model.onnx"


# ─────────────────────────────────────────────────────────────────────────────
#  The export loop
# ─────────────────────────────────────────────────────────────────────────────

def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def export_module(
    spec: ExportSpec,
    model: Mapping[str, torch.nn.Module],
    repo_dir: Path,
    *,
    opset: int,
    verify: bool = True,
    context: Optional[Dict[str, Any]] = None,
) -> ExportOutcome:
    """Export one module and verify the result against PyTorch."""
    path = onnx_path_for(repo_dir, spec.name)
    outcome = ExportOutcome(name=spec.name, path=path)
    context = context or {}

    try:
        module, example_inputs = spec.build(model, context)
    except Exception as exc:
        outcome.error = f"building example inputs failed: {exc}"
        traceback.print_exc()
        return outcome

    path.parent.mkdir(parents=True, exist_ok=True)
    requested = context.get("device", "cpu") if spec.device == "auto" else spec.device
    if requested != "cpu" and not torch.cuda.is_available():
        print(f"          (no CUDA available -- tracing {spec.name} on CPU)")
        requested = "cpu"
    device = torch.device(requested)

    with module_as(module, device=device, dtype=spec.dtype) as prepared:
        prepared.eval()
        inputs = tuple(
            t.to(device) if isinstance(t, torch.Tensor) else t for t in example_inputs
        )

        try:
            with torch.no_grad():
                reference = prepared(*inputs)
        except Exception as exc:
            outcome.error = f"PyTorch forward pass failed: {exc}"
            traceback.print_exc()
            return outcome

        try:
            with torch.no_grad():
                torch.onnx.export(
                    prepared,
                    inputs,
                    str(path),
                    export_params=True,
                    input_names=list(spec.input_names),
                    output_names=list(spec.output_names),
                    dynamic_axes={k: dict(v) for k, v in spec.dynamic_axes.items()},
                    opset_version=opset,
                    do_constant_folding=spec.do_constant_folding,
                    dynamo=False,
                )
            outcome.exported = True
        except Exception as exc:
            outcome.error = f"torch.onnx.export failed: {exc}"
            traceback.print_exc()
            return outcome

    try:
        _postprocess(path, symbolic=spec.symbolic_shape_inference)
    except Exception as exc:
        outcome.error = f"shape inference failed: {exc}"
        traceback.print_exc()
        return outcome

    if not verify or spec.verify == "none":
        outcome.verified = None
        return outcome

    try:
        outcome.max_abs_error = _verify(
            path, spec, inputs, reference, rtol=spec.rtol, atol=spec.atol
        )
        outcome.verified = True
    except Exception as exc:
        outcome.verified = False
        outcome.error = f"onnxruntime verification failed: {exc}"
        return outcome

    # Verifying only at the traced shape cannot catch a graph that baked a
    # dimension in -- the classic failure for LSTM and pack_padded_sequence
    # exports, which torch warns about but does not prevent. Re-run at a
    # different size on every dynamic axis and compare against PyTorch there.
    try:
        outcome.generalised = _verify_second_shape(
            path, spec, module, example_inputs, device,
            rtol=spec.rtol, atol=spec.atol,
        )
    except Exception as exc:
        outcome.verified = False
        outcome.error = f"failed at a second shape (graph is not dynamic): {exc}"

    return outcome


def _rescale_dynamic_axes(
    spec: ExportSpec, inputs: Sequence[torch.Tensor]
) -> Optional[Tuple[torch.Tensor, ...]]:
    """Build a second set of inputs with every dynamic axis at a new size.

    Axes sharing a name in ``dynamic_axes`` (``seq_len`` across several inputs)
    must stay consistent, so one new size is chosen per axis name. Returns None
    when the spec has no resizable axis.
    """
    sizes: Dict[str, int] = {}
    for name, axes in spec.dynamic_axes.items():
        if name not in spec.input_names:
            continue
        tensor = inputs[list(spec.input_names).index(name)]
        for axis, label in axes.items():
            if label in sizes or axis >= tensor.ndim:
                continue
            current = tensor.shape[axis]
            if label == "batch" or current <= 1:
                continue          # batch stays 1; a size-1 axis has nowhere to go
            sizes[label] = max(2, current // 2 + 3)   # not a multiple of the original

    if not sizes:
        return None

    rebuilt = []
    for name, tensor in zip(spec.input_names, inputs):
        shape = list(tensor.shape)
        changed = False
        for axis, label in spec.dynamic_axes.get(name, {}).items():
            if label in sizes and axis < len(shape):
                shape[axis] = sizes[label]
                changed = True
        if not changed:
            rebuilt.append(tensor)
        elif tensor.dtype.is_floating_point:
            rebuilt.append(torch.randn(*shape, dtype=tensor.dtype, device=tensor.device))
        elif tensor.dtype == torch.bool:
            rebuilt.append(torch.zeros(*shape, dtype=tensor.dtype, device=tensor.device))
        else:
            high = int(tensor.max().item()) + 1 if tensor.numel() else 2
            rebuilt.append(torch.randint(0, max(high, 2), shape,
                                         dtype=tensor.dtype, device=tensor.device))
    return tuple(rebuilt)


def _verify_second_shape(
    path: Path,
    spec: ExportSpec,
    module: torch.nn.Module,
    example_inputs: Sequence[torch.Tensor],
    device: torch.device,
    *,
    rtol: float,
    atol: float,
) -> Optional[int]:
    """Re-run the graph at a different dynamic size. Returns that size, or None."""
    base = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in example_inputs)
    resized = _rescale_dynamic_axes(spec, base)
    if resized is None:
        return None

    # Length-carrying inputs must agree with the tensors they describe.
    for i, name in enumerate(spec.input_names):
        if name in ("INPUT_LENGTHS",):
            seq = next(
                (resized[j].shape[a]
                 for j, n in enumerate(spec.input_names)
                 for a, label in spec.dynamic_axes.get(n, {}).items()
                 if label == "seq_len" and a < resized[j].ndim),
                None,
            )
            if seq is not None:
                resized = resized[:i] + (torch.tensor([seq], dtype=resized[i].dtype,
                                                      device=device),) + resized[i + 1:]

    with module_as(module, device=device, dtype=spec.dtype) as prepared:
        prepared.eval()
        with torch.no_grad():
            reference = prepared(*resized)

    _verify(path, spec, resized, reference, rtol=rtol, atol=atol)
    longest = max(t.shape[-1] for t in resized if isinstance(t, torch.Tensor) and t.ndim)
    return int(longest)


def _postprocess(path: Path, *, symbolic: bool) -> None:
    """Run shape inference in place so Triton/TRT see concrete tensor shapes."""
    import onnx

    model = onnx.load(str(path))
    onnx.save(onnx.shape_inference.infer_shapes(model), str(path))

    if symbolic:
        from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

        model = onnx.load(str(path))
        onnx.save(SymbolicShapeInference.infer_shapes(model, auto_merge=True), str(path))


def _verify(
    path: Path,
    spec: ExportSpec,
    inputs: Sequence[torch.Tensor],
    reference: Any,
    *,
    rtol: float,
    atol: float,
) -> float:
    """Run the exported graph and compare against *reference*. Returns max |err|."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    graph_inputs = [i.name for i in session.get_inputs()]
    if set(graph_inputs) != set(spec.input_names):
        raise AssertionError(
            f"graph inputs {graph_inputs} do not match spec input_names "
            f"{list(spec.input_names)}"
        )

    feed = {name: _to_numpy(t) for name, t in zip(spec.input_names, inputs)}
    produced = session.run(list(spec.output_names), feed)

    expected = reference if isinstance(reference, (tuple, list)) else (reference,)
    if len(expected) != len(produced):
        raise AssertionError(
            f"expected {len(expected)} output(s), graph produced {len(produced)}"
        )

    worst = 0.0
    for name, torch_out, onnx_out in zip(spec.output_names, expected, produced):
        torch_np = _to_numpy(torch_out)
        if torch_np.shape != onnx_out.shape:
            raise AssertionError(
                f"output '{name}' shape mismatch: torch {torch_np.shape} "
                f"vs onnx {onnx_out.shape}"
            )
        if not np.all(np.isfinite(onnx_out)):
            bad = int(np.count_nonzero(~np.isfinite(onnx_out)))
            raise AssertionError(
                f"output '{name}' contains {bad} non-finite value(s)"
            )
        if spec.verify != "full":
            # A graph with a Random* node produces different values on every
            # run, so shapes and finiteness are all that can be asserted.
            continue
        a = torch_np.astype(np.float64)
        b = onnx_out.astype(np.float64)
        worst = max(worst, float(np.max(np.abs(a - b))) if a.size else 0.0)
        np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, err_msg=f"output '{name}'")
    return worst if spec.verify == "full" else None
