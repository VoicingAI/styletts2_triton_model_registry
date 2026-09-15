"""ONNX export: the traceable wrappers, the harness, and the model registry.

    wrappers.py   nn.Modules reshaped so the exporter can trace them
    harness.py    how to export and verify ONE module
    pipeline.py   WHAT to export, in what order, and the loop over it
"""

from __future__ import annotations

__all__ = [
    "EXPORTERS",
    "ExportOutcome",
    "ExportSpec",
    "export_module",
    "module_as",
    "onnx_path_for",
    "run_exports",
]

from .harness import (
    ExportOutcome,
    ExportSpec,
    export_module,
    module_as,
    onnx_path_for,
)
from .pipeline import EXPORTERS, run_exports
