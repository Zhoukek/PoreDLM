"""Strict JSON experiment configuration and sequential, auditable orchestration.

Planning uses only JSON metadata; feature arrays, labels and checkpoints are not
opened. Execution preserves the freeze-all-predictions-before-scoring boundary.
This is a known-boundary diagnostic, not a raw-signal basecaller.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import platform
from pathlib import Path
import re
import sys
import warnings


TRAIN_DEFAULTS = {
    "epochs": 30, "patience": 5, "batch_reads": 8, "learning_rate": 1e-3,
    "weight_decay": 1e-4, "class_balance": "none", "seed": 260916,
    "device": "cpu", "threads": 4,
}
PREDICT_DEFAULTS = {"device": "cpu", "batch_events": 2048, "threads": 4}
SCORE_DEFAULTS = {"bootstrap_reps": 500, "seed": 260916, "min_reads": 5}
MODEL_DEFAULTS = [
    {"name": f"context_w{width}", "head": "context", "window_size": width, "projection_dim": 128}
    for width in (1, 3, 5)
]
LIMITATION = (
    "Known-boundary, alignment-filtered event classification; not signal-only inference. "
    "PoreKmer does not estimate or add label offsets. Imported NanoSignalAlign phase "
    "calibration is retained as phase_calibrated, not independently validated. "
    "Legacy unverified-register results remain exploratory, regardless of numerical performance."
)


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"Non-finite JSON constant is not allowed: {value}")


def _read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=_pairs, parse_constant=_constant)


def _object(value, name, allowed):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(f"Unknown {name} keys: {', '.join(sorted(unknown))}")
    return value


def _integer(value, name, minimum=1, maximum=None):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return value


def _number(value, name, positive):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric")
    if (positive and value <= 0) or (not positive and value < 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return value


def _path(value, name, root):
    if not isinstance(value, str) or not value.strip() or "\0" in value:
        raise ValueError(f"{name} must be a nonempty path string")
    path = Path(value).expanduser()
    return str((root / path).resolve())


def _device(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"cpu|cuda(?::[0-9]+)?|mps", value):
        raise ValueError(f"{name} must be cpu, cuda, cuda:<index>, or mps")


def _normalize_config(raw, path):
    raw = _object(raw, "config", (
        "schema_version", "source_manifest", "aligned_manifest", "manifest", "output_dir",
        "allow_unverified_register", "allow_calibration_reads", "max_reads_per_split",
        "models", "train", "predict", "score",
    ))
    if _integer(raw.get("schema_version"), "schema_version") != 1:
        raise ValueError("schema_version must be 1")
    input_keys = [key for key in ("source_manifest", "aligned_manifest", "manifest") if key in raw]
    if len(input_keys) != 1:
        raise ValueError("Exactly one of source_manifest, aligned_manifest or manifest is required")
    input_key = input_keys[0]
    allow = raw.get("allow_unverified_register", False)
    if not isinstance(allow, bool):
        raise ValueError("allow_unverified_register must be boolean")
    allow_calibration = raw.get("allow_calibration_reads", False)
    if not isinstance(allow_calibration, bool):
        raise ValueError("allow_calibration_reads must be boolean")
    if "allow_calibration_reads" in raw and input_key != "aligned_manifest":
        raise ValueError("allow_calibration_reads applies only to aligned_manifest preparation")
    limit = raw.get("max_reads_per_split")
    if limit is not None:
        _integer(limit, "max_reads_per_split")
        if input_key == "manifest":
            raise ValueError("max_reads_per_split applies only to source_manifest or aligned_manifest preparation")
    models = raw.get("models", MODEL_DEFAULTS)
    if not isinstance(models, list) or len(models) < 2:
        raise ValueError("models must contain at least two candidates; use train for a single model")
    normalized_models, names = [], set()
    for index, item in enumerate(models):
        model = _object(item, f"models[{index}]", ("name", "head", "window_size", "projection_dim"))
        name = model.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name):
            raise ValueError("Model names must be 1-64 safe letters, digits, underscores or hyphens")
        if name.casefold() in names:
            raise ValueError(f"Duplicate model name: {name}")
        names.add(name.casefold())
        head, window = model.get("head", "context"), model.get("window_size", 1)
        if head not in ("context", "cosine", "linear"):
            raise ValueError("Model head must be context, cosine or linear")
        if _integer(window, "window_size") not in (1, 3, 5):
            raise ValueError("window_size must be 1, 3 or 5")
        if head != "context" and window != 1:
            raise ValueError("cosine and linear require window_size=1")
        projection = _integer(model.get("projection_dim", 128), "projection_dim", 2)
        normalized_models.append({"name": name, "head": head, "window_size": window,
                                  "projection_dim": projection})
    sections = {}
    for name, defaults in (("train", TRAIN_DEFAULTS), ("predict", PREDICT_DEFAULTS), ("score", SCORE_DEFAULTS)):
        sections[name] = {**defaults, **_object(raw.get(name, {}), name, defaults)}
    training, prediction, scoring = sections["train"], sections["predict"], sections["score"]
    for key in ("epochs", "patience", "batch_reads", "threads"):
        _integer(training[key], f"train.{key}")
    for key in ("threads", "batch_events"):
        _integer(prediction[key], f"predict.{key}")
    for name, section in (("train", training), ("score", scoring)):
        _integer(section["seed"], f"{name}.seed", 0, 2**32 - 1)
    _integer(scoring["bootstrap_reps"], "score.bootstrap_reps", 0)
    _integer(scoring["min_reads"], "score.min_reads")
    _number(training["learning_rate"], "train.learning_rate", True)
    _number(training["weight_decay"], "train.weight_decay", False)
    if training["class_balance"] not in ("none", "inverse_sqrt"):
        raise ValueError("train.class_balance must be none or inverse_sqrt")
    _device(training["device"], "train.device")
    _device(prediction["device"], "predict.device")
    normalized = {
        "schema_version": 1, input_key: _path(raw[input_key], input_key, path.parent),
        "output_dir": _path(raw.get("output_dir"), "output_dir", path.parent),
        "allow_unverified_register": allow, "max_reads_per_split": limit,
        "models": normalized_models, **sections,
    }
    if input_key == "aligned_manifest":
        normalized["allow_calibration_reads"] = allow_calibration
    return normalized


def load_config(config_path) -> dict:
    """Validate strict JSON and return defaults plus absolute input/output paths.

    All candidates share one training budget, seed and corpus. At least two
    models are required because a run includes family freezing and comparison.
    File existence and register gating are surfaced separately by planning.
    """
    path = Path(config_path).resolve()
    return _normalize_config(_read_json(path), path)


def _make_plan(config, config_path):
    output = Path(config["output_dir"])
    aligned = "aligned_manifest" in config
    preparing = "source_manifest" in config or aligned
    input_key = "aligned_manifest" if aligned else "source_manifest" if preparing else "manifest"
    input_path = Path(config[input_key])
    manifest = output / "corpus" / "manifest.json" if preparing else input_path
    blockers, notes = [], [LIMITATION]
    if not input_path.is_file():
        blockers.append(f"Input manifest does not exist or is not a file: {input_path}")
    if output.exists() or output.is_symlink():
        blockers.append(f"Output directory already exists; runs never overwrite: {output}")
    register_status = "phase_calibrated" if aligned else "unverified" if preparing else "unknown"
    if aligned and input_path.is_file():
        metadata = _read_json(input_path)
        if not isinstance(metadata, dict) or metadata.get("kind") != "nanosignalalign_hidden":
            blockers.append("aligned_manifest must describe a nanosignalalign_hidden source")
    if not preparing and input_path.is_file():
        metadata = _read_json(input_path)
        if not isinstance(metadata, dict):
            blockers.append("Prepared manifest must be a JSON object")
        else:
            register = metadata.get("register", {})
            register_status = register.get("status", "unknown") if isinstance(register, dict) else "unknown"
            if register_status == "phase_calibrated":
                from .data import current_metadata, validate_phase_register
                try:
                    if type(metadata.get("schema_version")) is not int or metadata["schema_version"] != 2:
                        raise ValueError("phase_calibrated prepared registers require schema 2")
                    validate_phase_register(register)
                    if "current" not in metadata:
                        raise ValueError("schema 2 requires explicit current metadata")
                    current_metadata(metadata)
                except ValueError as exc:
                    blockers.append(str(exc))
            elif register_status != "unverified":
                blockers.append("Prepared register must be unverified or phase_calibrated; independent verification is not inferred")
    if register_status != "phase_calibrated" and not config["allow_unverified_register"]:
        blockers.append("Unverified register: set allow_unverified_register=true only for explicitly exploratory work")
    if register_status == "unverified" and config["allow_unverified_register"]:
        notes.append("Unverified registration explicitly acknowledged; results are exploratory")
    if register_status == "phase_calibrated":
        notes.append("Upstream phase calibration is preserved with zero additional offset; no independent physical-register validation is claimed")
    if aligned and config["allow_calibration_reads"]:
        notes.append("Upstream calibration reads explicitly included; phase-selection independence is not claimed for those reads")
    if config["max_reads_per_split"] is not None:
        notes.append("Read-limited smoke/debug subset; not a representative scientific evaluation")
    stages = [{
        "id": "prepare_aligned" if aligned else "prepare" if preparing else "reuse_corpus",
        "operation": "prepare_aligned" if aligned else "prepare" if preparing else "audit_manifest",
        "input": str(input_path), "output": str(manifest),
    }]
    artifacts = {}
    for model in config["models"]:
        name = model["name"]
        base = output / "models" / name
        artifacts[name] = {
            "training_dir": str(base),
            "checkpoint": str(base / "checkpoint.pt"),
            "prediction_dir": str(output / "predictions" / name),
            "predictions": str(output / "predictions" / name / "predictions.json"),
            "score_dir": str(output / "scores" / name),
            "scores": str(output / "scores" / name / "scores.json"),
        }
        stages.append({"id": f"train:{name}", "operation": "train", "model": name,
                       "input": str(manifest), "output": artifacts[name]["checkpoint"]})
    for name, paths in artifacts.items():
        stages.append({"id": f"predict:{name}", "operation": "predict", "model": name,
                       "input": paths["checkpoint"], "output": paths["predictions"]})
    family = output / "family.json"
    stages.append({"id": "freeze_family", "operation": "freeze_family",
                   "inputs": [a["predictions"] for a in artifacts.values()], "output": str(family)})
    for name, paths in artifacts.items():
        stages.append({"id": f"score:{name}", "operation": "score", "model": name,
                       "input": paths["predictions"], "family_lock": str(family), "output": paths["scores"]})
    comparison = output / "comparison.json"
    stages.append({"id": "compare", "operation": "compare",
                   "inputs": [a["scores"] for a in artifacts.values()], "output": str(comparison)})
    return {
        "schema_version": 1, "kind": "porekmer_experiment_plan", "config_path": str(config_path),
        "config": config, "runnable": not blockers, "blocking_issues": blockers,
        "register_status": register_status, "manifest": str(manifest),
        "models": artifacts, "family_lock": str(family), "comparison": str(comparison),
        "run_status": str(output / "run_status.json"), "report": str(output / "run_report.json"),
        "stages": stages, "notes": notes,
        "validation_scope": "JSON configuration and path/register metadata only; no array or model files opened",
    }


def plan_experiment(config_path) -> dict:
    """Return a JSON-safe dry run without creating files or importing torch."""
    path = Path(config_path).resolve()
    return _make_plan(load_config(path), path)


def _timestamp():
    return datetime.now(timezone.utc).isoformat()


def _write_json(path, value):
    """Replace only this run's own JSON status, avoiding half-written status files."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def run_experiment(config_path) -> dict:
    """Execute a fresh run; record stage failures and re-raise original errors.

    Invalid configuration, an existing output directory, or an unacknowledged
    register is rejected before any output is created. A failed run is preserved
    for diagnosis and is never automatically resumed or overwritten.
    """
    path = Path(config_path).resolve()
    source_bytes = path.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    config = _normalize_config(json.loads(source_bytes, object_pairs_hook=_pairs,
                                         parse_constant=_constant), path)
    plan = _make_plan(config, path)
    plan["config_sha256"] = source_sha256
    output = Path(config["output_dir"])
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite run: {output}")
    if plan["blocking_issues"]:
        raise ValueError("; ".join(plan["blocking_issues"]))
    if plan["register_status"] == "unverified" and config["allow_unverified_register"]:
        warnings.warn("PoreKmer run uses an UNVERIFIED register; results are exploratory. " + LIMITATION,
                      UserWarning, stacklevel=2)
    output.mkdir(parents=True, exist_ok=False)
    status = {
        "schema_version": 1, "kind": "porekmer_run_status", "status": "running",
        "started_at": _timestamp(), "current_stage": "initialize",
        "config_sha256": source_sha256,
        "stages": [{**stage, "status": "pending"} for stage in plan["stages"]],
    }
    active_stage = None
    try:
        with (output / "config.input.json").open("xb") as handle:
            handle.write(source_bytes)
        _write_json(output / "config.normalized.json", config)
        _write_json(output / "plan.json", plan)
        _write_json(output / "run_status.json", status)
        from . import data, workflow

        manifest = Path(plan["manifest"])
        candidates = {model["name"]: model for model in config["models"]}
        for stage in status["stages"]:
            active_stage = stage
            status["current_stage"] = stage["id"]
            stage.update(status="running", started_at=_timestamp())
            _write_json(output / "run_status.json", status)
            operation, name = stage["operation"], stage.get("model")
            paths = plan["models"].get(name, {})
            if operation == "prepare":
                data.prepare_corpus(Path(config["source_manifest"]), manifest.parent,
                                    allow_unverified_register=config["allow_unverified_register"],
                                    max_reads_per_split=config["max_reads_per_split"])
            elif operation == "prepare_aligned":
                from .aligned import prepare_aligned_corpus
                prepare_aligned_corpus(Path(config["aligned_manifest"]), manifest.parent,
                                       max_reads_per_split=config["max_reads_per_split"],
                                       allow_calibration_reads=config["allow_calibration_reads"])
            elif operation == "audit_manifest":
                data.load_manifest(manifest)
            elif operation == "train":
                model = {key: value for key, value in candidates[name].items() if key != "name"}
                workflow.train(manifest, Path(paths["training_dir"]), **model, **config["train"],
                               allow_unverified_register=config["allow_unverified_register"])
            elif operation == "predict":
                workflow.predict(manifest, Path(paths["checkpoint"]), Path(paths["prediction_dir"]),
                                 split="prediction", **config["predict"])
            elif operation == "freeze_family":
                workflow.freeze_family([Path(a["predictions"]) for a in plan["models"].values()],
                                       Path(plan["family_lock"]))
            elif operation == "score":
                workflow.score(manifest, Path(paths["predictions"]), Path(paths["score_dir"]),
                               family_lock=Path(plan["family_lock"]), **config["score"])
            elif operation == "compare":
                workflow.compare([Path(a["scores"]) for a in plan["models"].values()],
                                 Path(plan["comparison"]))
            stage.update(status="complete", finished_at=_timestamp())
            _write_json(output / "run_status.json", status)
        active_stage = None
        status["current_stage"] = "finalize"
        report = {
            "schema_version": 1, "kind": "porekmer_run_report", "status": "complete",
            "config_path": str(path), "config_sha256": source_sha256,
            "config_snapshot": str(output / "config.input.json"),
            "config": config, "manifest": str(manifest), "models": plan["models"],
            "family_lock": plan["family_lock"], "comparison": plan["comparison"],
            "register_status": plan["register_status"], "notes": plan["notes"],
            "started_at": status["started_at"], "finished_at": _timestamp(),
        }
        _write_json(output / "run_report.json", report)
        status.update(status="complete", current_stage=None, finished_at=report["finished_at"])
        _write_json(output / "run_status.json", status)
        return report
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        status.update(status="failed", error=error, finished_at=_timestamp())
        if active_stage is not None:
            active_stage.update(status="failed", error=error, finished_at=status["finished_at"])
        try:
            _write_json(output / "run_status.json", status)
        except Exception as status_error:
            warnings.warn(f"Could not persist run failure status: {status_error}", RuntimeWarning)
        raise


def environment_report() -> dict:
    """Report importability and device availability; never install dependencies."""
    from . import __version__

    report = {"toolkit": "PoreKmer", "version": __version__, "python": platform.python_version(),
              "executable": sys.executable, "platform": platform.platform(), "dependencies": {},
              "cuda": {"available": False}, "limitation": LIMITATION}
    for name in ("numpy", "torch"):
        try:
            module = importlib.import_module(name)
            report["dependencies"][name] = {"available": True, "version": str(module.__version__)}
            if name == "torch":
                available = bool(module.cuda.is_available())
                report["cuda"] = {"available": available, "runtime_version": module.version.cuda,
                                  "device_count": module.cuda.device_count() if available else 0}
        except Exception as exc:
            report["dependencies"][name] = {"available": False, "error": str(exc), "error_type": type(exc).__name__}
    report["ready"] = all(item["available"] for item in report["dependencies"].values())
    return report
