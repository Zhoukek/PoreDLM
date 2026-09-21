"""Command-line entry point; heavy training dependencies are imported lazily."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from . import __version__


def build_parser():
    parser = argparse.ArgumentParser(
        prog="porekmer",
        description="PoreKmer: frozen-feature event-context k-mer classification toolkit",
        epilog="Known-boundary, alignment-filtered experiments only; not a signal-only basecaller.",
    )
    parser.add_argument("--version", action="version", version=f"PoreKmer {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Inspect Python, dependencies and CUDA without installing anything")
    experiment = sub.add_parser("run", help="Run a configured, sealed multi-model experiment end to end")
    experiment.add_argument("--config", type=Path, required=True)
    experiment.add_argument("--dry-run", action="store_true", help="Validate configuration and show plan without creating outputs")
    prepare = sub.add_parser("prepare", help="Convert dense hidden cache to event feature/truth sidecars")
    prepare.add_argument("--source-manifest", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--max-reads-per-split", type=int, default=None,
                         help="Deterministic small-data smoke subset, not a final evaluation")
    prepare.add_argument("--allow-unverified-register", action="store_true",
                         help="Explicitly accept legacy label registration for exploratory work only")
    aligned = sub.add_parser("prepare-aligned", help="Pool hidden features using NanoSignalAlign reference spans")
    aligned.add_argument("--source-manifest", type=Path, required=True)
    aligned.add_argument("--output-dir", type=Path, required=True)
    aligned.add_argument("--max-reads-per-split", type=int, default=None,
                         help="Deterministic small-data smoke subset, not a final evaluation")
    aligned.add_argument("--allow-calibration-reads", action="store_true",
                         help="Include upstream calibration reads and retain their provenance")
    audit = sub.add_parser("audit", help="Validate prepared split identities and inspect corpus provenance")
    audit.add_argument("--manifest", type=Path, required=True)
    audit.add_argument("--verify-files", action="store_true", help="Also verify every feature/truth hash and shape")
    training = sub.add_parser("train", help="Fit a probe; only train/validation sidecars are opened")
    training.add_argument("--manifest", type=Path, required=True)
    training.add_argument("--output-dir", type=Path, required=True)
    training.add_argument("--window-size", type=int, choices=(1, 3, 5), default=1)
    training.add_argument("--head", choices=("context", "cosine", "linear"), default="context")
    training.add_argument("--projection-dim", type=int, default=128)
    training.add_argument("--epochs", type=int, default=30)
    training.add_argument("--patience", type=int, default=5)
    training.add_argument("--batch-reads", type=int, default=8)
    training.add_argument("--learning-rate", type=float, default=1e-3)
    training.add_argument("--weight-decay", type=float, default=1e-4)
    training.add_argument("--class-balance", choices=("none", "inverse_sqrt"), default="none")
    training.add_argument("--seed", type=int, default=260916)
    training.add_argument("--device", default="cpu")
    training.add_argument("--threads", type=int, default=4)
    training.add_argument("--allow-unverified-register", action="store_true")
    prediction = sub.add_parser("predict", help="Freeze held-out predictions without opening truth sidecars")
    prediction.add_argument("--manifest", type=Path, required=True)
    prediction.add_argument("--checkpoint", type=Path, required=True)
    prediction.add_argument("--output-dir", type=Path, required=True)
    prediction.add_argument("--split", choices=("prediction", "test"), default="prediction")
    prediction.add_argument("--device", default="cpu")
    prediction.add_argument("--batch-events", type=int, default=2048)
    prediction.add_argument("--threads", type=int, default=4)
    family = sub.add_parser("freeze-family", help="Seal all candidate predictions before any held-out scoring")
    family.add_argument("--predictions", nargs="+", type=Path, required=True)
    family.add_argument("--output", type=Path, required=True)
    scoring = sub.add_parser("score", help="Score frozen event predictions and independent-reference current tables")
    scoring.add_argument("--manifest", type=Path, required=True)
    scoring.add_argument("--predictions", type=Path, required=True)
    scoring.add_argument("--output-dir", type=Path, required=True)
    scoring.add_argument("--bootstrap-reps", type=int, default=500)
    scoring.add_argument("--seed", type=int, default=260916)
    scoring.add_argument("--min-reads", type=int, default=5)
    scoring.add_argument("--family-lock", type=Path, default=None)
    comparison = sub.add_parser("compare", help="Compare score reports only when center-event cohorts match")
    comparison.add_argument("--scores", nargs="+", type=Path, required=True)
    comparison.add_argument("--output", type=Path, required=True)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            from .pipeline import environment_report
            result = environment_report()
        elif args.command == "run":
            from .pipeline import plan_experiment, run_experiment
            result = plan_experiment(args.config) if args.dry_run else run_experiment(args.config)
        elif args.command == "prepare":
            from .data import prepare_corpus
            result = prepare_corpus(args.source_manifest, args.output_dir,
                                    allow_unverified_register=args.allow_unverified_register,
                                    max_reads_per_split=args.max_reads_per_split)
        elif args.command == "prepare-aligned":
            from .aligned import prepare_aligned_corpus
            result = prepare_aligned_corpus(args.source_manifest, args.output_dir,
                                            max_reads_per_split=args.max_reads_per_split,
                                            allow_calibration_reads=args.allow_calibration_reads)
        elif args.command == "audit":
            from .data import current_metadata, load_features, load_manifest, load_truth, make_windows
            result = load_manifest(args.manifest)
            if args.verify_files:
                for row in result["samples"]:
                    features = load_features(args.manifest.resolve().parent, row)
                    labels = load_truth(args.manifest.resolve().parent, row)
                    if len(make_windows(features)) != len(labels):
                        raise ValueError("Feature/truth center count mismatch")
            result = {
                "status": "pass", "reads_by_split": dict(Counter(r["split"] for r in result["samples"])),
                "events_by_split": dict(Counter({split: sum(r["n_centers"] for r in result["samples"]
                                                         if r["split"] == split)
                                                for split in {r["split"] for r in result["samples"]}})),
                "register": result["register"], "information_boundary": result["information_boundary"],
                "current": current_metadata(result),
                "all_files_verified": args.verify_files,
                "note": "Data-contract audit only; not proof of correct physical registration",
            }
        else:
            from . import workflow
            if args.command == "train":
                options = vars(args).copy()
                options.pop("command")
                manifest, output = options.pop("manifest"), options.pop("output_dir")
                result = workflow.train(manifest, output, **options)
            elif args.command == "predict":
                result = workflow.predict(args.manifest, args.checkpoint, args.output_dir,
                                          split=args.split, device=args.device,
                                          batch_events=args.batch_events, threads=args.threads)
            elif args.command == "score":
                result = workflow.score(args.manifest, args.predictions, args.output_dir,
                                        bootstrap_reps=args.bootstrap_reps, seed=args.seed, min_reads=args.min_reads,
                                        family_lock=args.family_lock)
            elif args.command == "freeze-family":
                result = workflow.freeze_family(args.predictions, args.output)
            else:
                result = workflow.compare(args.scores, args.output)
        # Keep terminal summaries compact; full reports remain in output artifacts.
        brief = {key: value for key, value in result.items() if key not in ("samples", "classification", "models")}
        if "classification" in result:
            brief["classification"] = {key: value for key, value in result["classification"].items()
                                       if key != "per_class"}
        print(json.dumps(brief, ensure_ascii=False, indent=2, allow_nan=False))
        return result
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"error: {exc}\n")


def entrypoint():
    """Console-script entry: never pass a result dict to sys.exit()."""
    main()
    return 0


if __name__ == "__main__":
    raise SystemExit(entrypoint())
