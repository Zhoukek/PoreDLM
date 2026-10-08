"""
Bonito Basecaller
"""

import os
import sys
import csv
import toml
import numpy as np
from itertools import islice as take
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from os.path import dirname, realpath

from bonito.reader import read_chunks, Reader
from bonito.io import biofmt
from bonito.cli.download import Downloader, models, __models_dir__
from bonito.multiprocessing import process_cancel
from bonito.util import column_to_set, set_config_defaults


def output_directory(path=None):
    if path:
        return path
    return '.' if sys.stdout.isatty() else dirname(realpath('/dev/fd/1'))


def find_references_path(directory):
    for filename in ("references.npy", "reference.npy"):
        path = os.path.join(directory, filename)
        if os.path.exists(path):
            return path
    return os.path.join(directory, "references.npy")


def resolve_model_directory(model_directory):
    if model_directory in models and not (__models_dir__ / model_directory).exists():
        sys.stderr.write("> downloading model\n")
        Downloader(__models_dir__).download(model_directory)
    if not os.path.isdir(model_directory) and os.path.isdir(os.path.join(__models_dir__, model_directory)):
        return os.path.join(__models_dir__, model_directory)
    return model_directory


def load_basecaller_config(model_directory, chunksize=None, batchsize=None, overlap=None, quantize=None):
    model_directory = resolve_model_directory(model_directory)
    config_path = os.path.join(model_directory, "config.toml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(config_path)
    config = toml.load(config_path)
    return set_config_defaults(config, chunksize, batchsize, overlap, quantize)


def load_summary_read_ids(summary_path):
    read_id_to_idx = {}
    with open(summary_path, newline='') as fh:
        reader = csv.DictReader(fh, delimiter='\t')
        if reader.fieldnames is None or 'read_id' not in reader.fieldnames:
            raise ValueError(f"{summary_path} does not contain a read_id column")
        # DictReader consumes the header, so idx=0 maps to the first data row
        # and therefore references.npy[0].
        for idx, row in enumerate(reader):
            read_id = row['read_id']
            if read_id not in read_id_to_idx:
                read_id_to_idx[read_id] = idx
    return read_id_to_idx


def save_filtered_chunks(reads, read_id_to_idx, reference_rows, out_dir):
    chunks = []
    references = []
    seen = set()
    total = 0
    matched = 0

    for read in reads:
        total += 1
        ref_idx = read_id_to_idx.get(read.read_id)
        if ref_idx is None or read.read_id in seen:
            continue
        if ref_idx >= len(reference_rows):
            raise IndexError(
                f"read_id {read.read_id} maps to row {ref_idx}, but references.npy "
                f"has only {len(reference_rows)} rows"
            )
        seen.add(read.read_id)
        matched += 1
        chunks.append(read.signal)
        references.append(reference_rows[ref_idx])

    if not chunks:
        sys.stderr.write(
            f"> no chunks matched acc95_summary.tsv read_id column "
            f"(scanned {total} chunks)\n"
        )
        return

    chunks = np.asarray(chunks, dtype=np.float16)
    references = np.asarray(references, dtype=reference_rows.dtype)
    reference_lengths = np.count_nonzero(references, axis=1).astype(np.uint16)

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "chunks.npy"), chunks)
    np.save(os.path.join(out_dir, "references.npy"), references)
    np.save(os.path.join(out_dir, "reference_lengths.npy"), reference_lengths)

    sys.stderr.write(f"> scanned chunks: {total}\n")
    sys.stderr.write(f"> matched chunks: {matched}\n")
    sys.stderr.write(f"> written mongo training data to {out_dir}\n")
    sys.stderr.write("  - chunks.npy with shape (%s)\n" % ','.join(map(str, chunks.shape)))
    sys.stderr.write("  - references.npy with shape (%s)\n" % ','.join(map(str, references.shape)))
    sys.stderr.write(
        "  - reference_lengths.npy with shape (%s)\n"
        % ','.join(map(str, reference_lengths.shape))
    )


def main(args):
    try:
        reader = Reader(args.reads_directory, args.recursive)
        sys.stderr.write("> reading %s\n" % reader.fmt)
    except FileNotFoundError:
        sys.stderr.write("> error: no suitable files found in %s\n" % args.reads_directory)
        exit(1)

    fmt = biofmt(aligned=args.reference is not None)

    if args.reference and args.reference.endswith(".mmi") and fmt.name == "cram":
        sys.stderr.write("> error: reference cannot be a .mmi when outputting cram\n")
        exit(1)
    elif args.reference and fmt.name == "fastq":
        sys.stderr.write(f"> warning: did you really want {fmt.aligned} {fmt.name}?\n")
    else:
        sys.stderr.write(f"> outputting {fmt.aligned} {fmt.name}\n")

    try:
        model_config = load_basecaller_config(
            args.model_directory,
            chunksize=args.chunksize,
            overlap=args.overlap,
            batchsize=args.batchsize,
            quantize=args.quantize,
        )
    except FileNotFoundError:
        sys.stderr.write(f"> error: failed to load {args.model_directory}\n")
        sys.stderr.write(f"> available models:\n")
        for model in sorted(models): sys.stderr.write(f" - {model}\n")
        exit(1)

    if args.verbose:
        sys.stderr.write(f"> model basecaller params: {model_config['basecaller']}\n")

    if args.reference:
        sys.stderr.write("> using precomputed summary/references; not loading reference index\n")
    else:
        sys.stderr.write("> warning: no reference provided; using precomputed summary/references only\n")

    if args.save_ctc and not args.reference:
        sys.stderr.write("> a reference is needed to output ctc training data\n")
        exit(1)

    if fmt.name != 'fastq':
        groups, num_reads = reader.get_read_groups(
            args.reads_directory, args.model_directory,
            n_proc=args.read_workers, recursive=args.recursive,
            read_ids=column_to_set(args.read_ids), skip=args.skip,
            cancel=process_cancel()
        )
    else:
        groups = []
        num_reads = None

    # 这里是是做预处理的
    reads = reader.get_reads(
        args.reads_directory, n_proc=args.read_workers, recursive=args.recursive,
        read_ids=column_to_set(args.read_ids), skip=args.skip,
        do_trim=not args.no_trim,
        scaling_strategy=model_config.get("scaling"),
        norm_params=(model_config.get("standardisation")
                     if (model_config.get("scaling") and
                         model_config.get("scaling").get("strategy") == "pa")
                     else model_config.get("normalisation")
                     ),
        cancel=process_cancel()
    )

    if args.verbose:
        sys.stderr.write(f"> read scaling: {model_config.get('scaling')}\n")
    
    if args.max_reads:
        reads = take(reads, args.max_reads)
        if num_reads is not None:
            num_reads = min(num_reads, args.max_reads)

    if args.save_ctc:
        reads = (
            chunk for read in reads
            for chunk in read_chunks(
                read,
                chunksize=model_config["basecaller"]["chunksize"],
                overlap=model_config["basecaller"]["overlap"]
            )
        )

    summary_dir = args.summary_dir or args.reads_directory
    summary_path = os.path.join(summary_dir, "acc95_summary.tsv")
    references_path = find_references_path(summary_dir)
    if not os.path.exists(summary_path):
        sys.stderr.write(f"> error: missing summary file: {summary_path}\n")
        exit(1)
    if not os.path.exists(references_path):
        sys.stderr.write(f"> error: missing references file: {references_path}\n")
        exit(1)

    read_id_to_idx = load_summary_read_ids(summary_path)
    references = np.load(references_path, mmap_mode='r')
    sys.stderr.write(f"> loaded {len(read_id_to_idx)} read_ids from {summary_path}\n")
    sys.stderr.write(f"> loaded {references_path} with shape {references.shape}\n")
    save_filtered_chunks(reads, read_id_to_idx, references, output_directory(args.output_dir))




def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("model_directory")
    parser.add_argument("reads_directory")
    parser.add_argument("--reference")
    parser.add_argument("--read-ids")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=25, type=int)
    parser.add_argument("--weights", default=0, type=int)
    parser.add_argument("--skip", action="store_true", default=False)
    parser.add_argument("--no-trim", action="store_true", default=False)
    parser.add_argument("--save-ctc", action="store_true", default=True)
    parser.add_argument("--revcomp", action="store_true", default=False)
    parser.add_argument("--rna", action="store_true", default=False)
    parser.add_argument("--recursive", action="store_true", default=False)
    quant_parser = parser.add_mutually_exclusive_group(required=False)
    quant_parser.add_argument("--quantize", dest="quantize", action="store_true")
    quant_parser.add_argument("--no-quantize", dest="quantize", action="store_false")
    parser.set_defaults(quantize=None)
    parser.add_argument("--overlap", default=None, type=int)
    parser.add_argument("--chunksize", default=None, type=int)
    parser.add_argument("--batchsize", default=None, type=int)
    parser.add_argument("--max-reads", default=0, type=int)
    parser.add_argument("--min-qscore", default=0, type=int)
    parser.add_argument("--min-accuracy-save-ctc", default=0.99, type=float)
    parser.add_argument("--alignment-threads", default=8, type=int)
    parser.add_argument("--mm2-preset", default='lr:hq', type=str)
    parser.add_argument("--read-workers", default=8, type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--summary-dir")
    parser.add_argument('-v', '--verbose', action='count', default=0)
    return parser
