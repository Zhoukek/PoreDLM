#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None


METRIC_LABELS = {
    "centroid_rms_l2": "Within-site RMS radius",
    "pairwise_mean_l2": "Within-site mean pairwise L2",
    "pairwise_mean_cosine": "Within-site mean pairwise cosine distance",
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


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def weighted_mean(rows: list[dict[str, Any]], field: str) -> float:
    total_weight = 0.0
    total = 0.0
    for row in rows:
        value = safe_float(row.get(field))
        weight = safe_float(row.get("n_reads"))
        if math.isfinite(value) and math.isfinite(weight) and weight > 0:
            total += value * weight
            total_weight += weight
    return total / total_weight if total_weight > 0 else float("nan")


def cluster_metrics(x: Any, max_pairwise_pairs: int = 200_000) -> dict[str, float]:
    if np is None or x.shape[0] == 0:
        return {
            "centroid_rms_l2": float("nan"),
            "pairwise_mean_l2": float("nan"),
            "pairwise_mean_cosine": float("nan"),
        }
    centered = x - x.mean(axis=0, keepdims=True)
    centroid_dist = np.linalg.norm(centered, axis=1)
    out = {
        "centroid_rms_l2": float(np.sqrt(np.mean(centroid_dist**2))),
        "pairwise_mean_l2": float("nan"),
        "pairwise_mean_cosine": float("nan"),
    }
    if x.shape[0] < 2:
        return out
    n_pairs = x.shape[0] * (x.shape[0] - 1) // 2
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    normalized = x / np.maximum(norms, 1e-12)
    if n_pairs <= max_pairwise_pairs:
        left, right = np.triu_indices(x.shape[0], k=1)
    else:
        rng = np.random.default_rng(20260920)
        left = rng.integers(0, x.shape[0], size=max_pairwise_pairs, endpoint=False)
        right = rng.integers(0, x.shape[0] - 1, size=max_pairwise_pairs, endpoint=False)
        right = right + (right >= left)
    out["pairwise_mean_l2"] = float(np.mean(np.linalg.norm(x[left] - x[right], axis=1)))
    out["pairwise_mean_cosine"] = float(1.0 - np.mean(np.sum(normalized[left] * normalized[right], axis=1)))
    return out


def separation_metrics(vectors: Any, sites: list[str]) -> dict[str, float]:
    if np is None:
        return {}
    unique_sites = sorted(set(sites))
    centers = []
    within_rms = []
    within_pairwise = []
    counts = []
    for site in unique_sites:
        idx = np.asarray([i for i, value in enumerate(sites) if value == site], dtype=np.int64)
        if idx.size == 0:
            continue
        site_vectors = vectors[idx]
        metrics = cluster_metrics(site_vectors)
        centers.append(site_vectors.mean(axis=0))
        within_rms.append(metrics["centroid_rms_l2"])
        within_pairwise.append(metrics["pairwise_mean_l2"])
        counts.append(int(idx.size))
    if not centers:
        return {}
    weights = np.asarray(counts, dtype=np.float64)
    weights = weights / max(float(weights.sum()), 1.0)
    out = {
        "within_site_rms_weighted_mean": float(np.nansum(np.asarray(within_rms) * weights)),
        "within_site_pairwise_weighted_mean": float(np.nansum(np.asarray(within_pairwise) * weights)),
        "between_site_center_l2_mean": float("nan"),
        "between_site_center_l2_min": float("nan"),
        "between_site_center_cosine_mean": float("nan"),
        "separation_over_within_rms": float("nan"),
        "separation_over_within_pairwise": float("nan"),
    }
    centers_arr = np.stack(centers, axis=0)
    if centers_arr.shape[0] < 2:
        return out
    left, right = np.triu_indices(centers_arr.shape[0], k=1)
    center_l2 = np.linalg.norm(centers_arr[left] - centers_arr[right], axis=1)
    center_norms = np.linalg.norm(centers_arr, axis=1, keepdims=True)
    normalized = centers_arr / np.maximum(center_norms, 1e-12)
    center_cosine = 1.0 - np.sum(normalized[left] * normalized[right], axis=1)
    out.update(
        {
            "between_site_center_l2_mean": float(np.mean(center_l2)),
            "between_site_center_l2_min": float(np.min(center_l2)),
            "between_site_center_cosine_mean": float(np.mean(center_cosine)),
            "separation_over_within_rms": ratio(float(np.mean(center_l2)), out["within_site_rms_weighted_mean"]),
            "separation_over_within_pairwise": ratio(
                float(np.mean(center_l2)), out["within_site_pairwise_weighted_mean"]
            ),
        }
    )
    return out


def summarize_from_csv(metrics_csv: Path) -> dict[str, Any]:
    rows = read_csv_rows(metrics_csv)
    summary: dict[str, Any] = {
        "run_dir": str(metrics_csv.parent),
        "metrics_csv": str(metrics_csv),
        "n_sites": len(rows),
        "n_reads": int(sum(safe_float(row.get("n_reads")) for row in rows if math.isfinite(safe_float(row.get("n_reads"))))),
        "per_site": rows,
        "within": {},
        "inter": None,
        "has_embeddings": False,
    }
    for metric in METRIC_LABELS:
        context = weighted_mean(rows, f"context_{metric}")
        ode = weighted_mean(rows, f"ode_{metric}")
        site_ratios = finite([safe_float(row.get(f"ratio_{metric}")) for row in rows])
        summary["within"][metric] = {
            "label": METRIC_LABELS[metric],
            "context": context,
            "ode": ode,
            "ratio": ratio(ode, context),
            "site_ratio_mean": mean(site_ratios) if site_ratios else float("nan"),
            "site_ratio_min": min(site_ratios) if site_ratios else float("nan"),
            "site_ratio_max": max(site_ratios) if site_ratios else float("nan"),
        }

    npz_path = metrics_csv.parent / "same_site_read_embeddings.npz"
    if npz_path.exists() and np is not None:
        data = np.load(npz_path, allow_pickle=True)
        sites = [str(x) for x in data["sites"].tolist()]
        context = data["context_hidden"]
        ode = data["ode_hidden"]
        context_sep = separation_metrics(context, sites)
        ode_sep = separation_metrics(ode, sites)
        summary["has_embeddings"] = True
        summary["embeddings_npz"] = str(npz_path)
        summary["inter"] = {
            "context": context_sep,
            "ode": ode_sep,
            "ratios": {
                key: ratio(ode_sep.get(key, float("nan")), context_sep.get(key, float("nan")))
                for key in [
                    "between_site_center_l2_mean",
                    "between_site_center_l2_min",
                    "between_site_center_cosine_mean",
                    "separation_over_within_rms",
                    "separation_over_within_pairwise",
                ]
            },
        }
    return summary


def fmt(value: float, digits: int = 3) -> str:
    if not math.isfinite(value):
        return "NA"
    return f"{value:.{digits}g}"


def svg_bar_pair(title: str, labels: list[str], context_values: list[float], ode_values: list[float]) -> str:
    width, height = 820, 330
    left, right, top, bottom = 72, 30, 42, 60
    plot_w, plot_h = width - left - right, height - top - bottom
    max_val = max(finite(context_values + ode_values) or [1.0])
    max_val *= 1.12
    group_w = plot_w / max(len(labels), 1)
    bar_w = min(52, group_w * 0.28)
    parts = [
        f'<svg viewBox="0 0 {width} {height}" class="chart" role="img">',
        f"<title>{html.escape(title)}</title>",
        f'<text x="{left}" y="24" class="chart-title">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>',
    ]
    for tick in range(5):
        value = max_val * tick / 4
        y = top + plot_h - value / max_val * plot_h
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" class="tick">{fmt(value, 2)}</text>')
    for i, label in enumerate(labels):
        cx = left + group_w * (i + 0.5)
        for offset, value, cls, name in [(-bar_w * 0.58, context_values[i], "bar-context", "context"), (bar_w * 0.58, ode_values[i], "bar-ode", "ode")]:
            h = 0 if not math.isfinite(value) else value / max_val * plot_h
            x = cx + offset - bar_w / 2
            y = top + plot_h - h
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" class="{cls}">'
                f"<title>{html.escape(label)} {name}: {fmt(value)}</title></rect>"
            )
        parts.append(f'<text x="{cx:.1f}" y="{height - 28}" text-anchor="middle" class="x-label">{html.escape(label)}</text>')
    parts.extend(
        [
            f'<rect x="{width - 210}" y="16" width="14" height="14" class="bar-context"/><text x="{width - 190}" y="28" class="legend">context</text>',
            f'<rect x="{width - 112}" y="16" width="14" height="14" class="bar-ode"/><text x="{width - 92}" y="28" class="legend">ODE</text>',
            "</svg>",
        ]
    )
    return "\n".join(parts)


def svg_ratio_dots(title: str, rows: list[dict[str, Any]], ratio_field: str) -> str:
    width, height = 860, max(260, 34 * len(rows) + 90)
    left, right, top, bottom = 230, 36, 48, 36
    plot_w, plot_h = width - left - right, height - top - bottom
    ratios = [safe_float(row.get(ratio_field)) for row in rows]
    finite_ratios = finite(ratios)
    if finite_ratios:
        xmin = min(0.0, min(finite_ratios) * 0.95)
        xmax = max(1.05, max(finite_ratios) * 1.05)
    else:
        xmin, xmax = 0.0, 1.2
    def x_pos(value: float) -> float:
        return left + (value - xmin) / max(xmax - xmin, 1e-12) * plot_w
    parts = [
        f'<svg viewBox="0 0 {width} {height}" class="chart" role="img">',
        f"<title>{html.escape(title)}</title>",
        f'<text x="{left}" y="26" class="chart-title">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>',
    ]
    for tick in [0.5, 0.75, 1.0, 1.25]:
        if xmin <= tick <= xmax:
            x = x_pos(tick)
            cls = "one-line" if abs(tick - 1.0) < 1e-9 else "grid"
            parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" class="{cls}"/>')
            parts.append(f'<text x="{x:.1f}" y="{height - 12}" text-anchor="middle" class="tick">{tick:g}</text>')
    for i, row in enumerate(rows):
        y = top + (i + 0.5) * plot_h / max(len(rows), 1)
        value = ratios[i]
        label = str(row.get("site", f"site_{i}"))
        parts.append(f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" class="site-label">{html.escape(label)}</text>')
        if math.isfinite(value):
            x = x_pos(value)
            cls = "dot-good" if value < 1.0 else "dot-bad"
            parts.append(f'<line x1="{x_pos(1.0):.1f}" y1="{y:.1f}" x2="{x:.1f}" y2="{y:.1f}" class="stem"/>')
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" class="{cls}"><title>{fmt(value)}</title></circle>')
            parts.append(f'<text x="{x + 10:.1f}" y="{y + 4:.1f}" class="value-label">{fmt(value, 3)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def svg_ratio_summary(summary: dict[str, Any]) -> str:
    labels = []
    context = []
    ode = []
    for metric, payload in summary["within"].items():
        labels.append(metric.replace("_", " "))
        context.append(payload["context"])
        ode.append(payload["ode"])
    return svg_bar_pair("Class-internal compactness: context vs ODE", labels, context, ode)


def write_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    fields = [
        "run_dir",
        "n_sites",
        "n_reads",
        "metric",
        "context",
        "ode",
        "ode_over_context",
        "site_ratio_min",
        "site_ratio_max",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            for metric, payload in summary["within"].items():
                writer.writerow(
                    {
                        "run_dir": summary["run_dir"],
                        "n_sites": summary["n_sites"],
                        "n_reads": summary["n_reads"],
                        "metric": metric,
                        "context": payload["context"],
                        "ode": payload["ode"],
                        "ode_over_context": payload["ratio"],
                        "site_ratio_min": payload["site_ratio_min"],
                        "site_ratio_max": payload["site_ratio_max"],
                    }
                )


def render_html(summaries: list[dict[str, Any]], out_html: Path) -> None:
    sections = []
    for summary in summaries:
        rows = summary["per_site"]
        run_name = Path(summary["run_dir"]).as_posix()
        cards = []
        for metric, payload in summary["within"].items():
            cards.append(
                f"""
                <div class="metric-card">
                  <div class="metric-name">{html.escape(payload['label'])}</div>
                  <div class="metric-value">{fmt(payload['ratio'])}<span> ODE/context</span></div>
                  <div class="metric-note">context {fmt(payload['context'])} · ODE {fmt(payload['ode'])}</div>
                  <div class="metric-note">site ratios: {fmt(payload['site_ratio_min'])} to {fmt(payload['site_ratio_max'])}</div>
                </div>
                """
            )
        inter_block = '<p class="warn">No <code>same_site_read_embeddings.npz</code> or global separation metrics were found in this run directory, so between-site metrics cannot be recomputed here. Re-run extraction with NPZ output to enable this panel.</p>'
        if summary.get("inter"):
            inter = summary["inter"]
            lines = []
            for key, value in inter["ratios"].items():
                lines.append(f"<tr><td>{html.escape(key)}</td><td>{fmt(value)}</td></tr>")
            inter_block = "<table><tr><th>Between-site / separation metric</th><th>ODE/context</th></tr>" + "".join(lines) + "</table>"
        sections.append(
            f"""
            <section>
              <h2>{html.escape(run_name)}</h2>
              <div class="cards">{''.join(cards)}</div>
              {svg_ratio_summary(summary)}
              {svg_ratio_dots('Per-site ODE/context RMS radius ratio', rows, 'ratio_centroid_rms_l2')}
              {svg_ratio_dots('Per-site ODE/context mean pairwise L2 ratio', rows, 'ratio_pairwise_mean_l2')}
              {svg_ratio_dots('Per-site ODE/context mean pairwise cosine ratio', rows, 'ratio_pairwise_mean_cosine')}
              <h3>Between-Site Panel</h3>
              {inter_block}
            </section>
            """
        )
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>ODE Feature-Space Metrics</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; color: #17202a; background: #f6f7f9; }}
    header {{ padding: 28px 36px 18px; background: #ffffff; border-bottom: 1px solid #dde2e7; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    h2 {{ margin-top: 28px; font-size: 20px; }}
    h3 {{ margin: 24px 0 8px; }}
    code {{ background: #eef1f4; padding: 1px 4px; border-radius: 4px; }}
    main {{ padding: 10px 36px 36px; }}
    section {{ background: #fff; border: 1px solid #dde2e7; border-radius: 8px; padding: 18px 22px 26px; margin: 18px 0; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 12px; }}
    .metric-card {{ border: 1px solid #dfe4ea; border-radius: 8px; padding: 12px 14px; background: #fbfcfd; }}
    .metric-name {{ font-size: 13px; color: #53616f; }}
    .metric-value {{ font-size: 28px; font-weight: 720; margin-top: 4px; }}
    .metric-value span {{ font-size: 13px; color: #53616f; font-weight: 500; }}
    .metric-note {{ color: #53616f; font-size: 12px; margin-top: 3px; }}
    .warn {{ color: #7a4b00; background: #fff7df; border: 1px solid #f0d38a; padding: 10px 12px; border-radius: 6px; }}
    .chart {{ width: 100%; max-width: 980px; margin-top: 18px; background: #fff; }}
    .chart-title {{ font-weight: 700; font-size: 15px; }}
    .axis {{ stroke: #65717e; stroke-width: 1; }}
    .grid {{ stroke: #d9dee5; stroke-width: 1; }}
    .one-line {{ stroke: #2c3e50; stroke-width: 1.4; stroke-dasharray: 4 4; }}
    .bar-context {{ fill: #4c78a8; }}
    .bar-ode {{ fill: #f58518; }}
    .tick, .legend, .x-label, .site-label, .value-label {{ font-size: 11px; fill: #4e5b68; }}
    .site-label {{ font-size: 10px; }}
    .stem {{ stroke: #aab3bd; stroke-width: 1; }}
    .dot-good {{ fill: #2f9e44; }}
    .dot-bad {{ fill: #d9480f; }}
    table {{ border-collapse: collapse; margin-top: 10px; }}
    th, td {{ border: 1px solid #dce2e8; padding: 7px 10px; text-align: left; }}
    th {{ background: #f1f4f7; }}
  </style>
</head>
<body>
  <header>
    <h1>ODE Feature-Space Metrics</h1>
    <p>Ratios below 1 mean ODE made reads more compact within the same site. For between-site separation, ratios near or above 1 are usually preferable.</p>
  </header>
  <main>
    {''.join(sections)}
  </main>
</body>
</html>
"""
    out_html.write_text(html_text, encoding="utf-8")


def discover_metric_csvs(root: Path) -> list[Path]:
    return sorted(root.glob("**/per_site_compactness_metrics.csv"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an HTML/SVG dashboard for context-vs-ODE feature metrics.")
    parser.add_argument("--root", default="/Users/kexuanzhou/project/PoreDLM/data/eval-data", help="Evaluation-data root.")
    parser.add_argument("--output-dir", default=None, help="Defaults to ROOT/feature-space-metrics-dashboard.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else root / "feature-space-metrics-dashboard"
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_csvs = discover_metric_csvs(root)
    if not metric_csvs:
        raise SystemExit(f"No per_site_compactness_metrics.csv files found under {root}")
    summaries = [summarize_from_csv(path) for path in metric_csvs]
    (output_dir / "feature_space_metric_summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_summary_csv(output_dir / "feature_space_metric_summary.csv", summaries)
    render_html(summaries, output_dir / "index.html")
    print(json.dumps({"dashboard": str(output_dir / "index.html"), "runs": len(summaries)}, indent=2))


if __name__ == "__main__":
    main()
