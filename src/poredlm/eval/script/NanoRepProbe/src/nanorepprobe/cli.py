import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description='NanoRepProbe: frozen-feature cross-domain linear evaluation')
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('run', help='fit source-train probes, select with source-validation, score held-out tests')
    run.add_argument('--config', required=True, type=Path)
    run.add_argument('--output', required=True, type=Path)
    run.add_argument('--resume', action='store_true')
    run.add_argument('--threads', type=int, default=4)
    audit = commands.add_parser('audit', help='validate manifest grouping, feature shape and input hashes')
    audit.add_argument('--config', required=True, type=Path)
    audit.add_argument('--output', type=Path)
    validate = commands.add_parser('validate', help='verify frozen outputs and recompute metrics from predictions')
    validate.add_argument('output', type=Path)
    report = commands.add_parser('report', help='regenerate plots/reports from saved metrics and renew output snapshot')
    report.add_argument('output', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == 'run':
            from .runner import run
            run(args.config, args.output, args.resume, args.threads)
        elif args.command == 'audit':
            from .data import load_config, load_domains, audit_domains
            result = audit_domains(load_domains(load_config(args.config)))
            if args.output:
                from .runner import dump
                dump(args.output, result)
            else:
                print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == 'validate':
            from .runner import validate_run
            print(json.dumps(validate_run(args.output), ensure_ascii=False, indent=2))
        elif args.command == 'report':
            from .report import render_report
            from .runner import write_snapshot
            print(json.dumps(render_report(args.output), ensure_ascii=False, indent=2))
            write_snapshot(args.output)
    except (ValueError, OSError, KeyError) as exc:
        print(f'NanoRepProbe error: {exc}', file=sys.stderr)
        return 2
    return 0
