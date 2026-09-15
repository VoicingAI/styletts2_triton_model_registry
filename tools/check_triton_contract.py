#!/usr/bin/env python3
"""Check an exported Triton repo against its config.pbtxt files.

Triton will not tell you that a model's config disagrees with its graph until
load time, and an ensemble wires fifteen of them together — so one wrong dtype
surfaces as an opaque failure much later. This walks the repo and reports, per
model, any input/output that the config declares but the ONNX graph does not
(or vice versa), plus dtype and rank mismatches, and any ensemble tensor that is
produced by nobody or consumed by nobody.

    python tools/check_triton_contract.py triton_repos/<model_id>/triton_model_repository

Exits non-zero if anything is wrong, so it works as a CI gate.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# config.pbtxt TYPE_* -> ONNX TensorProto elem_type
_TRITON_TO_ONNX = {
    "TYPE_BOOL": 9, "TYPE_UINT8": 2, "TYPE_UINT16": 4, "TYPE_UINT32": 12,
    "TYPE_UINT64": 13, "TYPE_INT8": 3, "TYPE_INT16": 5, "TYPE_INT32": 6,
    "TYPE_INT64": 7, "TYPE_FP16": 10, "TYPE_FP32": 1, "TYPE_FP64": 11,
    "TYPE_STRING": 8,
}
_ONNX_TO_NAME = {v: k for k, v in _TRITON_TO_ONNX.items()}

_BLOCK = re.compile(r"^(input|output)\s*\[(.*?)^\]", re.S | re.M)
_ENTRY = re.compile(r"\{([^{}]*)\}", re.S)
_FIELD = re.compile(r'(\w+)\s*:\s*("?[\w\-.]+"?|\[[^\]]*\])')


def parse_pbtxt(path: Path) -> dict:
    """Minimal config.pbtxt reader: name, backend/platform, inputs, outputs.

    Deliberately not a protobuf parser — it only needs the four fields above,
    and depending on tritonclient just to lint a repo is not worth it.
    """
    text = path.read_text()
    text = re.sub(r"#.*$", "", text, flags=re.M)

    def scalar(key: str) -> Optional[str]:
        match = re.search(rf'^\s*{key}\s*:\s*"?([\w\-./]+)"?', text, re.M)
        return match.group(1) if match else None

    result = {
        "name": scalar("name"),
        "backend": scalar("backend") or scalar("platform"),
        "input": [],
        "output": [],
    }
    for kind, body in _BLOCK.findall(text):
        for entry in _ENTRY.findall(body):
            fields = dict(_FIELD.findall(entry))
            if "name" not in fields:
                continue
            dims_raw = fields.get("dims", "[]").strip("[]")
            dims = [int(d) for d in re.findall(r"-?\d+", dims_raw)]
            result[kind].append({
                "name": fields["name"].strip('"'),
                "data_type": fields.get("data_type", "").strip('"'),
                "dims": dims,
            })
    return result


def onnx_io(path: Path) -> Tuple[List[dict], List[dict]]:
    import onnx

    model = onnx.load(str(path))
    initialisers = {i.name for i in model.graph.initializer}

    def describe(values):
        out = []
        for value in values:
            if value.name in initialisers:
                continue           # a weight, not a runtime input
            shape = value.type.tensor_type.shape
            out.append({
                "name": value.name,
                "elem_type": value.type.tensor_type.elem_type,
                "rank": len(shape.dim),
            })
        return out

    return describe(model.graph.input), describe(model.graph.output)


def check_model(model_dir: Path) -> List[str]:
    problems: List[str] = []
    config_path = model_dir / "config.pbtxt"
    if not config_path.is_file():
        return [f"{model_dir.name}: no config.pbtxt"]

    config = parse_pbtxt(config_path)
    name = config["name"] or model_dir.name
    if config["name"] and config["name"] != model_dir.name:
        problems.append(
            f"{model_dir.name}: config name is '{config['name']}' but the "
            f"directory is '{model_dir.name}' — Triton requires they match"
        )

    graphs = sorted(model_dir.glob("*/model.onnx"))
    if not graphs:
        backend = (config["backend"] or "").lower()
        if backend in ("onnxruntime", "tensorrt"):
            problems.append(
                f"{name}: backend '{backend}' but no model.onnx was exported"
            )
        return problems

    graph_inputs, graph_outputs = onnx_io(graphs[-1])

    for kind, declared, actual in (
        ("input", config["input"], graph_inputs),
        ("output", config["output"], graph_outputs),
    ):
        declared_by_name = {d["name"]: d for d in declared}
        actual_by_name = {a["name"]: a for a in actual}

        for missing in sorted(set(declared_by_name) - set(actual_by_name)):
            problems.append(
                f"{name}: config declares {kind} '{missing}' but the graph has no "
                f"such {kind} (graph has: {sorted(actual_by_name) or 'none'})"
            )
        for extra in sorted(set(actual_by_name) - set(declared_by_name)):
            problems.append(
                f"{name}: graph has {kind} '{extra}' that config.pbtxt does not declare"
            )

        for shared in sorted(set(declared_by_name) & set(actual_by_name)):
            want = _TRITON_TO_ONNX.get(declared_by_name[shared]["data_type"])
            got = actual_by_name[shared]["elem_type"]
            if want is not None and got and want != got:
                problems.append(
                    f"{name}: {kind} '{shared}' is {declared_by_name[shared]['data_type']} "
                    f"in config but {_ONNX_TO_NAME.get(got, got)} in the graph"
                )
            declared_rank = len(declared_by_name[shared]["dims"])
            actual_rank = actual_by_name[shared]["rank"]
            if declared_rank and actual_rank and declared_rank != actual_rank:
                problems.append(
                    f"{name}: {kind} '{shared}' has rank {declared_rank} in config "
                    f"but rank {actual_rank} in the graph"
                )
    return problems


def check_ensemble(repo: Path) -> List[str]:
    problems: List[str] = []
    for config_path in sorted(repo.glob("*/config.pbtxt")):
        text = re.sub(r"#.*$", "", config_path.read_text(), flags=re.M)
        if "ensemble_scheduling" not in text:
            continue

        name = config_path.parent.name
        produced = set(re.findall(r'output_map\s*\{[^}]*value\s*:\s*"([^"]+)"', text))
        consumed = set(re.findall(r'input_map\s*\{[^}]*value\s*:\s*"([^"]+)"', text))

        header = text.split("ensemble_scheduling")[0]
        ensemble_in = set()
        ensemble_out = set()
        for kind, body in _BLOCK.findall(header):
            for entry in _ENTRY.findall(body):
                fields = dict(_FIELD.findall(entry))
                if "name" in fields:
                    target = ensemble_in if kind == "input" else ensemble_out
                    target.add(fields["name"].strip('"'))

        for orphan in sorted(consumed - produced - ensemble_in):
            problems.append(
                f"{name}: step consumes '{orphan}' but nothing produces it"
            )
        for unused in sorted(ensemble_in - consumed):
            problems.append(
                f"{name}: ensemble input '{unused}' is accepted from clients but "
                f"never routed to any step — it is silently ignored"
            )
        for dead in sorted(produced - consumed - ensemble_out):
            problems.append(f"{name}: '{dead}' is produced but never consumed")

        for step_model in sorted(set(re.findall(r'model_name\s*:\s*"([^"]+)"', text))):
            if not (repo / step_model / "config.pbtxt").is_file():
                problems.append(
                    f"{name}: references model '{step_model}', which is not in this repo"
                )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("repo", help="path to a Triton model repository")
    args = parser.parse_args()

    repo = Path(args.repo)
    if not repo.is_dir():
        print(f"not a directory: {repo}", file=sys.stderr)
        return 2

    problems: List[str] = []
    models = sorted(p for p in repo.iterdir() if p.is_dir())
    for model_dir in models:
        problems.extend(check_model(model_dir))
    problems.extend(check_ensemble(repo))

    print(f"checked {len(models)} model(s) in {repo}\n")
    if not problems:
        print("no contract mismatches found ✓")
        return 0

    print(f"{len(problems)} problem(s):\n")
    for problem in problems:
        print(f"  • {problem}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
