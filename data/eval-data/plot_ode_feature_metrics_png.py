#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import struct
import zlib
from pathlib import Path
from statistics import mean
from typing import Any


BG = (248, 250, 252)
WHITE = (255, 255, 255)
INK = (25, 35, 45)
MUTED = (92, 105, 118)
GRID = (218, 225, 232)
CONTEXT = (76, 120, 168)
ODE = (245, 133, 24)
GOOD = (47, 158, 68)
BAD = (217, 72, 15)


FONT = {
    " ": ["000", "000", "000", "000", "000", "000", "000"],
    "-": ["000", "000", "000", "111", "000", "000", "000"],
    ".": ["000", "000", "000", "000", "000", "110", "110"],
    "/": ["001", "001", "010", "010", "100", "100", "000"],
    ":": ["000", "110", "110", "000", "110", "110", "000"],
    "_": ["000", "000", "000", "000", "000", "000", "111"],
    "0": ["111", "101", "101", "101", "101", "101", "111"],
    "1": ["010", "110", "010", "010", "010", "010", "111"],
    "2": ["111", "001", "001", "111", "100", "100", "111"],
    "3": ["111", "001", "001", "111", "001", "001", "111"],
    "4": ["101", "101", "101", "111", "001", "001", "001"],
    "5": ["111", "100", "100", "111", "001", "001", "111"],
    "6": ["111", "100", "100", "111", "101", "101", "111"],
    "7": ["111", "001", "001", "010", "010", "100", "100"],
    "8": ["111", "101", "101", "111", "101", "101", "111"],
    "9": ["111", "101", "101", "111", "001", "001", "111"],
    "A": ["010", "101", "101", "111", "101", "101", "101"],
    "B": ["110", "101", "101", "110", "101", "101", "110"],
    "C": ["111", "100", "100", "100", "100", "100", "111"],
    "D": ["110", "101", "101", "101", "101", "101", "110"],
    "E": ["111", "100", "100", "110", "100", "100", "111"],
    "F": ["111", "100", "100", "110", "100", "100", "100"],
    "G": ["111", "100", "100", "101", "101", "101", "111"],
    "H": ["101", "101", "101", "111", "101", "101", "101"],
    "I": ["111", "010", "010", "010", "010", "010", "111"],
    "J": ["001", "001", "001", "001", "101", "101", "111"],
    "K": ["101", "101", "110", "100", "110", "101", "101"],
    "L": ["100", "100", "100", "100", "100", "100", "111"],
    "M": ["101", "111", "111", "101", "101", "101", "101"],
    "N": ["101", "111", "111", "111", "101", "101", "101"],
    "O": ["111", "101", "101", "101", "101", "101", "111"],
    "P": ["111", "101", "101", "111", "100", "100", "100"],
    "Q": ["111", "101", "101", "101", "111", "001", "001"],
    "R": ["110", "101", "101", "110", "110", "101", "101"],
    "S": ["111", "100", "100", "111", "001", "001", "111"],
    "T": ["111", "010", "010", "010", "010", "010", "010"],
    "U": ["101", "101", "101", "101", "101", "101", "111"],
    "V": ["101", "101", "101", "101", "101", "101", "010"],
    "W": ["101", "101", "101", "101", "111", "111", "101"],
    "X": ["101", "101", "101", "010", "101", "101", "101"],
    "Y": ["101", "101", "101", "010", "010", "010", "010"],
    "Z": ["111", "001", "001", "010", "100", "100", "111"],
}


def safe_float(value: Any) -> float:
    try:
        if value in ("", None):
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def finite(values: list[float]) -> list[float]:
    return [v for v in values if math.isfinite(v)]


def ratio(a: float, b: float) -> float:
    if not math.isfinite(a) or not math.isfinite(b) or abs(b) < 1e-12:
        return float("nan")
    return a / b


def fmt(value: float, digits: int = 3) -> str:
    return "NA" if not math.isfinite(value) else f"{value:.{digits}g}"


class Canvas:
    def __init__(self, width: int, height: int, bg: tuple[int, int, int] = BG):
        self.width = width
        self.height = height
        self.pixels = bytearray(bg * width * height)

    def set(self, x: int, y: int, color: tuple[int, int, int]) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            i = (y * self.width + x) * 3
            self.pixels[i : i + 3] = bytes(color)

    def rect(self, x: int, y: int, w: int, h: int, color: tuple[int, int, int]) -> None:
        x0, x1 = max(0, x), min(self.width, x + w)
        y0, y1 = max(0, y), min(self.height, y + h)
        row = bytes(color) * max(0, x1 - x0)
        for yy in range(y0, y1):
            i = (yy * self.width + x0) * 3
            self.pixels[i : i + len(row)] = row

    def line(self, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            self.set(x0, y0, color)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def circle(self, cx: int, cy: int, r: int, color: tuple[int, int, int]) -> None:
        for y in range(cy - r, cy + r + 1):
            for x in range(cx - r, cx + r + 1):
                if (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
                    self.set(x, y, color)

    def text(self, x: int, y: int, text: str, color: tuple[int, int, int] = INK, scale: int = 2) -> None:
        cursor = x
        for char in text.upper():
            glyph = FONT.get(char, FONT.get(" "))
            for gy, row in enumerate(glyph):
                for gx, bit in enumerate(row):
                    if bit == "1":
                        self.rect(cursor + gx * scale, y + gy * scale, scale, scale, color)
            cursor += (len(glyph[0]) + 1) * scale

    def save_png(self, path: Path) -> None:
        raw = bytearray()
        stride = self.width * 3
        for y in range(self.height):
            raw.append(0)
            raw.extend(self.pixels[y * stride : (y + 1) * stride])
        def chunk(kind: bytes, data: bytes) -> bytes:
            return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        png = b"\x89PNG\r\n\x1a\n"
        png += chunk(b"IHDR", struct.pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0))
        png += chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        png += chunk(b"IEND", b"")
        path.write_bytes(png)


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def weighted_mean(rows: list[dict[str, Any]], field: str) -> float:
    total = 0.0
    weight_sum = 0.0
    for row in rows:
        value = safe_float(row.get(field))
        weight = safe_float(row.get("n_reads"))
        if math.isfinite(value) and math.isfinite(weight) and weight > 0:
            total += value * weight
            weight_sum += weight
    return total / weight_sum if weight_sum > 0 else float("nan")


def draw_summary(rows: list[dict[str, Any]], out: Path) -> dict[str, Any]:
    metrics = [
        ("RMS", "centroid_rms_l2"),
        ("PAIR L2", "pairwise_mean_l2"),
        ("COS", "pairwise_mean_cosine"),
    ]
    context = [weighted_mean(rows, f"context_{field}") for _, field in metrics]
    ode = [weighted_mean(rows, f"ode_{field}") for _, field in metrics]
    ratios = [ratio(o, c) for c, o in zip(context, ode)]
    c = Canvas(1200, 720)
    c.rect(35, 35, 1130, 650, WHITE)
    c.text(70, 70, "CONTEXT VS ODE CLASS-INTERNAL METRICS", scale=3)
    c.text(70, 112, "LOWER ODE/CONTEXT MEANS TIGHTER WITHIN-SITE READS", MUTED, scale=2)
    left, top, plot_w, plot_h = 120, 185, 930, 390
    max_val = max(finite(context + ode) or [1.0]) * 1.18
    for i in range(6):
        value = max_val * i / 5
        y = int(top + plot_h - value / max_val * plot_h)
        c.line(left, y, left + plot_w, y, GRID)
        c.text(48, y - 8, fmt(value, 2), MUTED, scale=2)
    c.line(left, top, left, top + plot_h, INK)
    c.line(left, top + plot_h, left + plot_w, top + plot_h, INK)
    group_w = plot_w // len(metrics)
    bar_w = 72
    for i, ((label, _), cv, ov, rv) in enumerate(zip(metrics, context, ode, ratios)):
        cx = left + group_w * i + group_w // 2
        for value, color, offset in [(cv, CONTEXT, -48), (ov, ODE, 48)]:
            h = 0 if not math.isfinite(value) else int(value / max_val * plot_h)
            c.rect(cx + offset - bar_w // 2, top + plot_h - h, bar_w, h, color)
        c.text(cx - 60, top + plot_h + 28, label, INK, scale=3)
        c.text(cx - 60, top + plot_h + 68, "RATIO " + fmt(rv), GOOD if rv < 1 else BAD, scale=2)
    c.rect(840, 82, 28, 20, CONTEXT)
    c.text(880, 82, "CONTEXT", MUTED, scale=2)
    c.rect(840, 114, 28, 20, ODE)
    c.text(880, 114, "ODE", MUTED, scale=2)
    c.save_png(out)
    return {
        field: {"context": cv, "ode": ov, "ratio": rv}
        for (_, field), cv, ov, rv in zip(metrics, context, ode, ratios)
    }


def draw_ratio_plot(rows: list[dict[str, Any]], field: str, title: str, out: Path) -> None:
    rows = sorted(rows, key=lambda row: safe_float(row.get(field)))
    h = max(520, 115 + len(rows) * 62)
    c = Canvas(1350, h)
    c.rect(35, 35, 1280, h - 70, WHITE)
    c.text(70, 72, title, scale=3)
    c.text(70, 116, "EACH DOT IS ONE SITE. 1.0 MEANS NO CHANGE AFTER ODE.", MUTED, scale=2)
    left, right, top, bottom = 430, 90, 180, 80
    plot_w, plot_h = 1350 - left - right, h - top - bottom
    values = [safe_float(row.get(field)) for row in rows]
    vals = finite(values)
    xmin = min(0.0, min(vals) * 0.94) if vals else 0.0
    xmax = max(1.08, max(vals) * 1.08) if vals else 1.2
    def x_pos(v: float) -> int:
        return int(left + (v - xmin) / max(xmax - xmin, 1e-12) * plot_w)
    for tick in [0.5, 0.75, 1.0, 1.25]:
        if xmin <= tick <= xmax:
            x = x_pos(tick)
            c.line(x, top, x, top + plot_h, INK if tick == 1.0 else GRID)
            c.text(x - 22, top + plot_h + 20, fmt(tick, 2), MUTED, scale=2)
    c.line(left, top + plot_h, left + plot_w, top + plot_h, INK)
    base = x_pos(1.0)
    site_map = []
    for i, row in enumerate(rows):
        y = int(top + (i + 0.5) * plot_h / max(len(rows), 1))
        site = str(row.get("site", f"SITE_{i + 1}"))
        site_label = f"S{i + 1}"
        site_map.append({"site_index": site_label, "site": site, "ratio": values[i]})
        c.text(72, y - 9, site_label, INK, scale=2)
        c.text(145, y - 9, site[:34], MUTED, scale=2)
        if math.isfinite(values[i]):
            x = x_pos(values[i])
            c.line(base, y, x, y, GRID)
            c.circle(x, y, 9, GOOD if values[i] < 1.0 else BAD)
            c.text(x + 18, y - 8, fmt(values[i]), INK, scale=2)
    c.save_png(out)
    out.with_suffix(".sites.csv").write_text(
        "site_index,site,ratio\n"
        + "\n".join(f"{r['site_index']},{r['site']},{r['ratio']}" for r in site_map)
        + "\n",
        encoding="utf-8",
    )


def discover(root: Path) -> list[Path]:
    return sorted(root.glob("**/per_site_compactness_metrics.csv"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate PNG feature-space metric charts from ODE/context CSVs.")
    parser.add_argument("--root", default="/Users/kexuanzhou/project/PoreDLM/data/eval-data")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    out_dir = Path(args.output_dir).resolve() if args.output_dir else root / "feature-space-metrics-png"
    out_dir.mkdir(parents=True, exist_ok=True)
    csvs = discover(root)
    if not csvs:
        raise SystemExit(f"No per_site_compactness_metrics.csv found under {root}")
    summaries = []
    for run_idx, csv_path in enumerate(csvs, start=1):
        rows = read_rows(csv_path)
        prefix = f"run{run_idx}_{csv_path.parent.parent.name}_{csv_path.parent.name}".replace("/", "_")
        summary = draw_summary(rows, out_dir / f"{prefix}_summary.png")
        draw_ratio_plot(rows, "ratio_centroid_rms_l2", "ODE / CONTEXT WITHIN-SITE RMS RADIUS", out_dir / f"{prefix}_ratio_rms.png")
        draw_ratio_plot(rows, "ratio_pairwise_mean_l2", "ODE / CONTEXT WITHIN-SITE PAIRWISE L2", out_dir / f"{prefix}_ratio_pairwise_l2.png")
        draw_ratio_plot(rows, "ratio_pairwise_mean_cosine", "ODE / CONTEXT WITHIN-SITE PAIRWISE COSINE", out_dir / f"{prefix}_ratio_pairwise_cosine.png")
        summaries.append({"run_dir": str(csv_path.parent), "charts_prefix": prefix, "summary": summary})
    (out_dir / "png_chart_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(out_dir), "runs": len(summaries)}, indent=2))


if __name__ == "__main__":
    main()
