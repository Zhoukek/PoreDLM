"""Build local wheel + source archive without publishing or downloading.

Use a Python environment that already has setuptools, wheel, NumPy and PyTorch.
The default runs the standalone tests before invoking the PEP 517 backend.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("dist"))
    parser.add_argument("--skip-tests", action="store_true",
                        help="Skip unit tests explicitly; does not mark the release verified")
    args = parser.parse_args(argv)
    project = Path(__file__).resolve().parents[1]
    output = args.out.resolve() if args.out.is_absolute() else (project / args.out).resolve()
    if output.exists():
        parser.error(f"Refusing to overwrite an existing distribution directory: {output}")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(project / "src")
    if not args.skip_tests:
        subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                       cwd=project, env=environment, check=True)
    output.mkdir(parents=True, exist_ok=False)
    backend = (
        "from setuptools.build_meta import build_sdist, build_wheel; "
        f"build_sdist({str(output)!r}); build_wheel({str(output)!r})"
    )
    subprocess.run([sys.executable, "-c", backend], cwd=project, env=environment, check=True)
    artifacts = []
    for path in sorted(output.iterdir()):
        if path.name.endswith((".whl", ".tar.gz")):
            artifacts.append({"name": path.name, "bytes": path.stat().st_size,
                              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    if len(artifacts) != 2:
        raise RuntimeError("Expected exactly one wheel and one source distribution")
    report = {"project": "PoreKmer", "python": sys.version,
              "standalone_tests_executed": not args.skip_tests,
              "published": False, "artifacts": artifacts}
    with (output / "build_manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
