"""PoreKmer training and frozen-prediction evaluation.

Prediction opens only feature sidecars. Boundary/eligibility information still
comes from annotated move/alignment data: this is NOT signal-only inference.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np

from .data import current_metadata, file_sha256, load_features, load_manifest, load_truth, make_windows
from .metrics import TABLE_FIELDS, _validated_rows, classification_metrics, compare_tables, table_rows, write_table
from . import __version__


def _json(path, value, *, exclusive=False):
    with Path(path).open("x" if exclusive else "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _new_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def _rows(manifest, split):
    return sorted(
        (r for r in manifest["samples"] if r["split"] == split),
        key=lambda r: (r["read_id"], r["record_id"]),
    )


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _registration_gate(manifest, acknowledged):
    status = manifest.get("register", {}).get("status")
    if status == "phase_calibrated":
        return
    if status != "unverified" or not acknowledged:
        raise ValueError(
            "Register is unverified. Resolve label calibration before confirmatory work; "
            "use --allow-unverified-register only for explicitly exploratory runs."
        )


def _register_interpretation(manifest):
    if manifest["register"]["status"] == "phase_calibrated":
        return "Upstream phase calibration retained with zero additional offset; physical registration is not independently validated"
    return "An unverified register makes results exploratory regardless of numerical performance"


def _write_current_table(path, rows, current):
    """Keep legacy Stone tables stable while naming other current scales accurately."""
    if current["normalization"] == "stone_unclamped":
        write_table(path, rows)
        return
    rows = _validated_rows(rows)
    fields = tuple("current_median" if field == "stone_median" else field for field in TABLE_FIELDS)
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            result = {field: row[field] for field in TABLE_FIELDS if field != "stone_median"}
            result["current_median"] = "NA" if row["stone_median"] is None else format(float(row["stone_median"]), ".17g")
            writer.writerow(result)


def _configure_torch(device, threads):
    # cuBLAS requires this before creating a CUDA context when deterministic
    # matrix operations are enabled. Never silently override a user's setting.
    if str(device).startswith("cuda"):
        configured = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if configured not in (":4096:8", ":16:8"):
            raise ValueError("Deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8")
    import torch
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but not available; use --device cpu")


def _read_batch(root, rows):
    windows, labels, read_indices = [], [], []
    for index, row in enumerate(rows):
        features = load_features(root, row)
        truth = load_truth(root, row)
        values = make_windows(features)
        if truth.shape != (len(values),):
            raise ValueError("Feature/truth center count mismatch")
        windows.append(values)
        labels.append(truth)
        read_indices.append(np.full(len(truth), index, dtype=np.int64))
    return np.concatenate(windows), np.concatenate(labels), np.concatenate(read_indices)


def _topk(logits, count=5):
    import torch
    return torch.argsort(logits, dim=-1, descending=True, stable=True)[:, :count]


def train(
    manifest_path, output_dir, *, window_size=1, head="context", projection_dim=128,
    epochs=30, patience=5, batch_reads=8, learning_rate=1e-3, weight_decay=1e-4,
    class_balance="none", seed=260916, device="cpu", threads=4,
    allow_unverified_register=False,
):
    """Fit a read-balanced event probe using ONLY train and validation sidecars."""
    import torch
    from .models import EventContextClassifier, read_balanced_cross_entropy

    for key, value in (("epochs", epochs), ("patience", patience),
                       ("batch_reads", batch_reads), ("threads", threads)):
        _positive(value, key)
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if class_balance not in ("none", "inverse_sqrt"):
        raise ValueError("class_balance must be none or inverse_sqrt")
    manifest_path = Path(manifest_path).resolve()
    root = manifest_path.parent
    manifest = load_manifest(manifest_path)
    _registration_gate(manifest, allow_unverified_register)
    training, validation = _rows(manifest, "train"), _rows(manifest, "validation")
    if not training or not validation:
        raise ValueError("Nonempty train and validation splits are required")
    _configure_torch(device, threads)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = EventContextClassifier(
        hidden_dim=int(manifest["hidden_dim"]), num_classes=int(manifest["num_classes"]),
        window_size=window_size, head=head, projection_dim=projection_dim,
    ).to(device)
    output = _new_directory(output_dir)
    config = {
        "manifest_path": str(manifest_path), "manifest_sha256": file_sha256(manifest_path),
        "toolkit_version": __version__, "smoke_limit": manifest.get("smoke_limit"),
        "architecture": model.architecture_config(),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "epochs": epochs, "patience": patience, "batch_reads": batch_reads,
        "learning_rate": learning_rate, "weight_decay": weight_decay,
        "class_balance": class_balance, "seed": seed, "device": str(device), "threads": threads,
        "training_reads": len(training), "validation_reads": len(validation),
        "unit": "center_event", "loss": "event_mean_within_read_then_mean_across_reads",
        "selection_metric": "validation_read_macro_cross_entropy",
        "register": manifest["register"],
        "current": current_metadata(manifest),
        "information_boundary": manifest["information_boundary"],
        "allow_unverified_register": bool(allow_unverified_register),
        "torch_version": str(torch.__version__), "numpy_version": str(np.__version__),
        "deterministic_algorithms": True,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    _json(output / "config.json", config)
    counts = np.zeros(manifest["num_classes"], dtype=np.int64)
    for row in training:
        counts += np.bincount(load_truth(root, row), minlength=len(counts))
    class_weights = None
    if class_balance == "inverse_sqrt":
        weights = 1.0 / np.sqrt(np.maximum(counts, 1))
        weights /= weights.mean()
        class_weights = torch.as_tensor(weights, dtype=torch.float32, device=device)
    _json(output / "training_class_support.json", counts.tolist())
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generator = np.random.default_rng(seed)
    history, best, stale = [], float("inf"), 0
    started = time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train()
        train_total = 0.0
        shuffled = [training[i] for i in generator.permutation(len(training))]
        for first in range(0, len(shuffled), batch_reads):
            rows = shuffled[first:first + batch_reads]
            x, y, r = _read_batch(root, rows)
            logits = model(torch.as_tensor(x, dtype=torch.float32, device=device))
            loss = read_balanced_cross_entropy(
                logits, torch.as_tensor(y, dtype=torch.long, device=device),
                torch.as_tensor(r, dtype=torch.long, device=device), class_weights=class_weights,
            )
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            train_total += float(loss.detach()) * len(rows)
        model.eval()
        validation_total, validation_top1 = 0.0, 0.0
        with torch.inference_mode():
            for first in range(0, len(validation), batch_reads):
                rows = validation[first:first + batch_reads]
                x, y, r = _read_batch(root, rows)
                logits = model(torch.as_tensor(x, dtype=torch.float32, device=device))
                targets = torch.as_tensor(y, dtype=torch.long, device=device)
                loss = read_balanced_cross_entropy(
                    logits, targets, torch.as_tensor(r, dtype=torch.long, device=device),
                )
                validation_total += float(loss) * len(rows)
                hit = (logits.argmax(-1) == targets).cpu().numpy()
                validation_top1 += sum(float(hit[r == i].mean()) for i in range(len(rows)))
        value = validation_total / len(validation)
        if not math.isfinite(value):
            raise ValueError("Nonfinite validation loss")
        entry = {
            "epoch": epoch, "train_read_macro_ce": train_total / len(training),
            "validation_read_macro_ce": value,
            "validation_read_macro_top1": validation_top1 / len(validation),
            "elapsed_seconds": time.monotonic() - started,
        }
        history.append(entry)
        _json(output / "history.json", history)
        print(json.dumps(entry), flush=True)
        if value < best:
            best, stale = value, 0
            torch.save({
                "architecture": model.architecture_config(),
                "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "manifest_sha256": config["manifest_sha256"],
                "epoch": epoch, "validation_read_macro_ce": best,
                "register": manifest["register"],
                "current": current_metadata(manifest),
            }, output / "checkpoint.pt")
        else:
            stale += 1
            if stale >= patience:
                break
    report = {
        "status": "complete", "best_validation_read_macro_ce": best,
        "epochs_run": len(history), "checkpoint_sha256": file_sha256(output / "checkpoint.pt"),
        "elapsed_seconds": time.monotonic() - started,
        "interpretation": "Boundary-conditioned probe; not a signal-only decoder. " + _register_interpretation(manifest),
        **config,
    }
    # Export the BEST checkpoint's classification weights, never the final
    # early-stopping epoch's parameters. These are not empirical centroids.
    best_checkpoint = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)
    bank_key = "classifier.hidden_bank" if head != "linear" else "classifier.weight"
    bank = best_checkpoint["model_state"][bank_key].float()
    if head != "linear":
        bank = torch.nn.functional.normalize(bank, dim=-1)
    with (output / "class_vectors.npy").open("xb") as handle:
        np.save(handle, bank.numpy(), allow_pickle=False)
    report["class_vectors"] = {
        "path": "class_vectors.npy", "shape": list(bank.shape),
        "sha256": file_sha256(output / "class_vectors.npy"),
        "meaning": "normalized learned class directions" if head != "linear" else "affine classifier weights; bias remains in checkpoint",
        "not_empirical_centroids": True,
    }
    _json(output / "training_report.json", report)
    return report


def predict(manifest_path, checkpoint_path, output_dir, *, split="prediction",
            device="cpu", batch_events=2048, threads=4):
    """Freeze predictions without opening any truth sidecar."""
    import torch
    from .models import EventContextClassifier

    _positive(batch_events, "batch_events")
    _positive(threads, "threads")
    if split not in ("prediction", "test"):
        raise ValueError("Prediction split must be prediction or test")
    manifest_path = Path(manifest_path).resolve()
    manifest = load_manifest(manifest_path)
    rows = _rows(manifest, split)
    if not rows:
        raise ValueError(f"Empty split: {split}")
    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["manifest_sha256"] != file_sha256(manifest_path):
        raise ValueError("Checkpoint was trained against a different event corpus")
    for name in ("hidden_dim", "num_classes"):
        if checkpoint["architecture"][name] != manifest[name]:
            raise ValueError(f"Checkpoint/corpus {name} mismatch")
    _configure_torch(device, threads)
    model = EventContextClassifier(**checkpoint["architecture"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    output = _new_directory(output_dir)
    (output / "records").mkdir()
    result_rows, cohort = [], hashlib.sha256()
    with torch.inference_mode():
        for row in rows:
            features = load_features(manifest_path.parent, row)
            windows = make_windows(features)
            predictions, probabilities, margins = [], [], []
            for first in range(0, len(windows), batch_events):
                logits = model(torch.as_tensor(
                    windows[first:first + batch_events], dtype=torch.float32, device=device,
                ))
                if not torch.isfinite(logits).all():
                    raise ValueError("Nonfinite prediction logits")
                order = _topk(logits, min(5, manifest["num_classes"]))
                predictions.append(order.cpu().numpy())
                probabilities.append(logits.softmax(-1).gather(1, order).cpu().numpy())
                ranked = logits.gather(1, order)
                margins.append((ranked[:, 0] - ranked[:, 1]).cpu().numpy())
            record_name = hashlib.sha256(row["record_id"].encode()).hexdigest()[:24]
            relative = f"records/{record_name}.npz"
            path = output / relative
            if path.exists():
                raise ValueError("Duplicate prediction record identity")
            with path.open("xb") as handle:
                np.savez_compressed(
                    handle, center_ids=features["center_ids"], currents=features["currents"],
                    topk=np.concatenate(predictions), probabilities=np.concatenate(probabilities),
                    margins=np.concatenate(margins),
                )
            identity = json.dumps([row["read_id"], row["record_id"],
                                   features["center_ids"].tolist()], separators=(",", ":"))
            cohort.update(identity.encode())
            result_rows.append({
                "read_id": row["read_id"], "record_id": row["record_id"],
                "path": relative, "sha256": file_sha256(path), "n_centers": len(windows),
            })
    result = {
        "schema_version": 1, "kind": "kmer_event_context_predictions", "status": "complete",
        "source_manifest_sha256": file_sha256(manifest_path),
        "toolkit_version": __version__, "smoke_limit": manifest.get("smoke_limit"),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "architecture": checkpoint["architecture"], "split": split,
        "cohort_sha256": cohort.hexdigest(), "register": manifest["register"],
        "current": current_metadata(manifest),
        "information_boundary": manifest["information_boundary"],
        "truth_sidecars_opened": False,
        "deterministic_algorithms": True,
        "selection_limit": "Centers and event boundaries were previously annotation-filtered",
        "samples": result_rows,
    }
    _json(output / "predictions.json", result)
    return result


def _safe_path(root, relative):
    root = Path(root).resolve()
    relative = Path(relative)
    result = (root / relative).resolve()
    if relative.is_absolute() or not result.is_relative_to(root):
        raise ValueError("Artifact path escapes its root")
    return result


def freeze_family(prediction_paths, output_path):
    """Seal all candidate predictions before opening held-out truth for scoring.

    This is a reproducibility seal, not proof that no human previously inspected
    labels. Each subsequent score also verifies the member files against source.
    """
    if len(prediction_paths) < 2:
        raise ValueError("A prediction family needs at least two members")
    members, seen = [], set()
    for prediction_path in prediction_paths:
        prediction_path = Path(prediction_path).resolve()
        digest = file_sha256(prediction_path)
        if digest in seen:
            raise ValueError("Duplicate prediction family member")
        seen.add(digest)
        with prediction_path.open(encoding="utf-8") as handle:
            report = json.load(handle)
        if report.get("kind") != "kmer_event_context_predictions" or report.get("status") != "complete":
            raise ValueError("Only completed predictions can enter a family")
        for row in report["samples"]:
            path = _safe_path(prediction_path.parent, row["path"])
            if file_sha256(path) != row["sha256"]:
                raise ValueError("Cannot seal a modified prediction file")
        member = {"path": str(prediction_path), "sha256": digest,
                  "source_manifest_sha256": report["source_manifest_sha256"],
                  "cohort_sha256": report["cohort_sha256"], "architecture": report["architecture"],
                  "current": current_metadata(report)}
        if members and any(member[key] != members[0][key] for key in ("source_manifest_sha256", "cohort_sha256", "current")):
            raise ValueError("Family members must use the same corpus, center events and current normalization/units")
        members.append(member)
    result = {"schema_version": 1, "kind": "kmer_event_context_family", "members": members,
              "source_manifest_sha256": members[0]["source_manifest_sha256"],
              "cohort_sha256": members[0]["cohort_sha256"],
              "current": members[0]["current"],
              "limitation": "Seals supplied artifacts; cannot prove labels were never viewed previously"}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    _json(output_path, result, exclusive=True)
    return result


def score(manifest_path, predictions_path, output_dir, *, bootstrap_reps=500,
          seed=260916, min_reads=5, family_lock=None):
    """Score frozen predictions, then compare with the independent reference table."""
    _positive(min_reads, "min_reads")
    if not isinstance(bootstrap_reps, int) or bootstrap_reps < 0:
        raise ValueError("bootstrap_reps must be nonnegative")
    manifest_path, predictions_path = Path(manifest_path).resolve(), Path(predictions_path).resolve()
    manifest = load_manifest(manifest_path)
    with predictions_path.open(encoding="utf-8") as handle:
        prediction = json.load(handle)
    if prediction.get("kind") != "kmer_event_context_predictions" or prediction.get("status") != "complete":
        raise ValueError("Not a complete frozen prediction artifact")
    if prediction["source_manifest_sha256"] != file_sha256(manifest_path):
        raise ValueError("Prediction/corpus manifest mismatch")
    current_info = current_metadata(manifest)
    if current_metadata(prediction) != current_info:
        raise ValueError("Prediction/corpus current normalization or units mismatch")
    family_sha = None
    if family_lock is not None:
        with Path(family_lock).open(encoding="utf-8") as handle:
            family = json.load(handle)
        if family.get("kind") != "kmer_event_context_family":
            raise ValueError("Invalid prediction family lock")
        if current_metadata(family) != current_info:
            raise ValueError("Family/corpus current normalization or units mismatch")
        if (family["source_manifest_sha256"] != prediction["source_manifest_sha256"]
                or family["cohort_sha256"] != prediction["cohort_sha256"]
                or file_sha256(predictions_path) not in {m["sha256"] for m in family["members"]}):
            raise ValueError("Prediction is not a member of this frozen family")
        # Verify the whole sealed family still exists unchanged before scoring
        # even the first member. No truth sidecar is needed for this check.
        for member in family["members"]:
            member_path = Path(member["path"])
            if file_sha256(member_path) != member["sha256"]:
                raise ValueError("A sealed family prediction manifest was modified")
            with member_path.open(encoding="utf-8") as handle:
                member_prediction = json.load(handle)
            for row in member_prediction["samples"]:
                if file_sha256(_safe_path(member_path.parent, row["path"])) != row["sha256"]:
                    raise ValueError("A sealed family prediction file was modified")
        family_sha = file_sha256(family_lock)
    if prediction.get("split") not in ("prediction", "test"):
        raise ValueError("Cannot score training/validation predictions as held-out results")
    expected = {r["record_id"]: r for r in _rows(manifest, prediction["split"])}
    records = prediction["samples"]
    if len(records) != len(expected) or {r["record_id"] for r in records} != set(expected):
        raise ValueError("Frozen prediction cohort differs from the selected split")
    refs = _rows(manifest, "reference")
    if not refs:
        raise ValueError("Independent reference split is required for table evaluation")
    ys, ks, currents, read_ids, sources = [], [], [], [], []
    cohort = hashlib.sha256()
    for row in sorted(records, key=lambda r: (r["read_id"], r["record_id"])):
        source = expected[row["record_id"]]
        if row["read_id"] != source["read_id"]:
            raise ValueError("Prediction read identity mismatch")
        path = _safe_path(predictions_path.parent, row["path"])
        if file_sha256(path) != row["sha256"]:
            raise ValueError("Frozen prediction file checksum mismatch")
        features = load_features(manifest_path.parent, source)
        with np.load(path, allow_pickle=False) as arrays:
            ids, current, topk = arrays["center_ids"], arrays["currents"], arrays["topk"]
        if not np.array_equal(ids, features["center_ids"]) or not np.array_equal(current, features["currents"]):
            raise ValueError("Prediction center IDs/currents do not match central-event features")
        if topk.shape != (len(ids), min(5, manifest["num_classes"])):
            raise ValueError("Prediction top-k shape mismatch")
        if (topk.dtype.kind not in "iu" or np.any(topk < 0)
                or np.any(topk >= manifest["num_classes"])
                or np.any(np.diff(np.sort(topk, axis=1), axis=1) == 0)):
            raise ValueError("Prediction top-k IDs must be valid distinct integer classes")
        identity = json.dumps([row["read_id"], row["record_id"], ids.tolist()], separators=(",", ":"))
        cohort.update(identity.encode())
        sources.append(source)
        ks.append(topk)
        currents.append(current)
        read_ids.extend([row["read_id"]] * len(ids))
    if cohort.hexdigest() != prediction["cohort_sha256"]:
        raise ValueError("Frozen prediction cohort fingerprint mismatch")
    # Only now, after verifying EVERY frozen prediction, open held-out truth.
    for source in sources:
        ys.append(load_truth(manifest_path.parent, source))
    labels, topk, current = np.concatenate(ys), np.concatenate(ks), np.concatenate(currents)
    read_ids = np.asarray(read_ids)
    classification = classification_metrics(
        labels, topk, read_ids, num_classes=manifest["num_classes"],
        bootstrap_reps=bootstrap_reps, seed=seed,
    )
    predicted_table = table_rows(topk[:, 0], current, read_ids, num_classes=manifest["num_classes"])
    ref_labels, ref_current, ref_ids = [], [], []
    for row in refs:
        feature = load_features(manifest_path.parent, row)
        truth = load_truth(manifest_path.parent, row)
        if len(truth) != len(feature["centers"]):
            raise ValueError("Reference feature/truth length mismatch")
        ref_labels.append(truth)
        ref_current.append(feature["currents"])
        ref_ids.extend([row["read_id"]] * len(truth))
    reference_table = table_rows(np.concatenate(ref_labels), np.concatenate(ref_current),
                                np.asarray(ref_ids), num_classes=manifest["num_classes"])
    table = compare_tables(predicted_table, reference_table, min_reads=min_reads)
    output = _new_directory(output_dir)
    table_scale = "stone" if current_info["normalization"] == "stone_unclamped" else "current"
    predicted_name = f"predicted_{table_scale}_5mer.tsv"
    reference_name = f"reference_{table_scale}_5mer.tsv"
    _write_current_table(output / predicted_name, predicted_table, current_info)
    _write_current_table(output / reference_name, reference_table, current_info)
    report = {
        "schema_version": 1, "kind": "kmer_event_context_scores", "status": "complete",
        "toolkit_version": __version__, "smoke_limit": manifest.get("smoke_limit"),
        "source_manifest_sha256": file_sha256(manifest_path),
        "predictions_sha256": file_sha256(predictions_path),
        "cohort_sha256": prediction["cohort_sha256"], "architecture": prediction["architecture"],
        "classification": classification, "table": table, "register": manifest["register"],
        "current": current_info,
        "bootstrap_reps": bootstrap_reps, "seed": seed, "min_reads": min_reads,
        "family_lock_sha256": family_sha,
        "family_status": "sealed" if family_sha else "unsealed_exploratory_individual_model",
        "tables": {
            "predicted": {"path": predicted_name, "sha256": file_sha256(output / predicted_name)},
            "reference": {"path": reference_name, "sha256": file_sha256(output / reference_name)},
        },
        "information_boundary": manifest["information_boundary"],
        "interpretation": [
            "Event-level known-boundary diagnostic; not directly comparable to old frame Top1",
            "No forced class filling; missing table entries are NA",
            "Reference table uses disjoint physical reads and the same max-window eligibility rule",
            "Table correlation does not establish exact k-mer identity or correct physical register",
            _register_interpretation(manifest),
            "Current table MAE and RMSE use the declared current units and normalization",
        ],
    }
    _json(output / "scores.json", report)
    return report


def compare(score_paths, output_path):
    """Describe models on the EXACT same held-out center-event cohort."""
    if len(score_paths) < 2:
        raise ValueError("At least two score files are required")
    reports = []
    for path in score_paths:
        with Path(path).open(encoding="utf-8") as handle:
            report = json.load(handle)
        if report.get("kind") != "kmer_event_context_scores" or report.get("status") != "complete":
            raise ValueError("Not a complete event-context score report")
        reports.append(report)
    current_info = current_metadata(reports[0])
    if any(current_metadata(report) != current_info for report in reports[1:]):
        raise ValueError("Incomparable score reports: current normalization or units differ")
    for key in ("source_manifest_sha256", "cohort_sha256", "min_reads", "register", "family_lock_sha256"):
        if any(report.get(key) != reports[0].get(key) for report in reports[1:]):
            raise ValueError(f"Incomparable score reports: {key} differs")
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    from .comparison import shared_table_panel
    panel = shared_table_panel(score_paths, reports)
    result = {
        "kind": "kmer_event_context_comparison", "source_manifest_sha256": reports[0]["source_manifest_sha256"],
        "toolkit_version": __version__, "smoke_limit": reports[0].get("smoke_limit"),
        "cohort_sha256": reports[0]["cohort_sha256"], "register": reports[0]["register"],
        "current": current_info,
        "comparison_type": "descriptive_same_cohort_not_a_significance_test",
        "table_comparison_note": "Per-model table metrics may use different common-class panels; compare coverage alongside error, not correlation alone",
        "shared_table_panel": panel,
        "models": [{"source": str(Path(path).resolve()), "source_sha256": file_sha256(path),
                    "architecture": report["architecture"],
                    "classification": {k: v for k, v in report["classification"].items() if k != "per_class"},
                    "table": report["table"]} for path, report in zip(score_paths, reports)],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _json(output_path, result, exclusive=True)
    return result
