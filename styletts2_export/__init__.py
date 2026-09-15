"""Export a StyleTTS2 checkpoint to the ONNX graphs a Triton ensemble needs.

One YAML file describes one export run. The stages, in the order
:func:`~styletts2_export.cli.run` executes them:

    config      parse and validate the YAML          config.py
    assets      fetch checkpoint / PLBERT / audio    assets.py
    checkpoint  build the model, load the weights    checkpoint.py + models.py
    onnx        export and verify twelve graphs      onnx/
    styles      one style vector per speaker         styles.py
    triton      assemble the model repository        triton.py

Nothing here builds TensorRT engines: a `.plan` is tied to the GPU and TensorRT
version that produced it, so it has to be built on the serving machine. See
`tools/build_trt_engines.py` and `docs/RUNBOOK.md`.
"""

from __future__ import annotations

__version__ = "0.2.0"

__all__ = [
    "ConfigError",
    "ExportConfig",
    "load_config",
    "ALL_MODULES",
]

from .config import ALL_MODULES, ConfigError, ExportConfig, load_config
