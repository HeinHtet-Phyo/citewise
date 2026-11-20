"""Render eval summaries as CSV and a readable markdown table (percentages, with n)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

# (column heading, summary key, n key or None, unit)
COLUMNS: list[tuple[str, str, str | None, str]] = [
    ("Hit@6", "retrieval_hit_rate_at_6_pct", "retrieval_hit_rate_at_6_pct_n", "%"),
    ("MRR", "mrr", "mrr_n", ""),
    (
        "Correct (answerable)",
        "answer_correct_pct_of_answerable",
        "answer_correct_pct_of_answerable_n",
        "%",
    ),
    ("Partly (answerable)", "answer_partly_correct_pct_of_answerable", None, "%"),
    (
        "Correct (answered)",
        "answer_correct_pct_of_answered",
        "answer_correct_pct_of_answered_n",
        "%",
    ),
    ("Partly (answered)", "answer_partly_correct_pct_of_answered", None, "%"),
    ("Fully faithful", "fully_faithful_pct", "fully_faithful_pct_n", "%"),
    ("Made-up rate", "made_up_rate_pct", "made_up_rate_pct_n", "%"),
    ("Citation correct", "citation_correct_pct", "citation_correct_pct_n", "%"),
    ("Refusal recall", "refusal_recall_pct", "refusal_recall_pct_n", "%"),
    ("False refusal", "false_refusal_rate_pct", "false_refusal_rate_pct_n", "%"),
    ("Adversarial pass", "adversarial_pass_rate_pct", "adversarial_pass_rate_pct_n", "%"),
    ("p50 ms", "latency_p50_ms", None, ""),
    ("p95 ms", "latency_p95_ms", None, ""),
    ("Cost/answer (GBP)", "cost_per_answer_gbp", None, ""),
]


def _fmt(value: Any, unit: str) -> str:
    """Format one metric value; None renders as n/a."""
    if value is None:
        return "n/a"
    return f"{value}{unit}"


def _cell(m: dict[str, Any], key: str, n_key: str | None, unit: str) -> str:
    """Format a metric with its n in brackets when available."""
    text = _fmt(m.get(key), unit)
    return f"{text} (n={m[n_key]})" if n_key and n_key in m else text


def write_csv(summary: dict[str, dict[str, Any]], path: Path) -> None:
    """Write one row per config with every summary key as a column."""
    keys = sorted({k for m in summary.values() for k in m})
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["config", *keys])
        for config, m in summary.items():
            writer.writerow([config, *[m.get(k, "") if m.get(k) is not None else "" for k in keys]])


def markdown_table(summary: dict[str, dict[str, Any]]) -> str:
    """Return a markdown table with one row per config."""
    head = "| Config | Questions | " + " | ".join(c[0] for c in COLUMNS) + " |"
    sep = "|" + "---|" * (len(COLUMNS) + 2)
    rows = [head, sep]
    for config, m in summary.items():
        cells = [_cell(m, k, n, u) for _, k, n, u in COLUMNS]
        rows.append(f"| {config} | {m.get('n_questions', 0)} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def write_markdown(results_dir: Path) -> Path:
    """Write summary.md covering every summary_<split>.json present."""
    parts = ["# Citewise evaluation summary\n"]
    for split in ("dev", "test"):
        path = results_dir / f"summary_{split}.json"
        if not path.exists():
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        parts.append(f"## {split} split\n\n{markdown_table(summary)}\n")
        errors = {c: m.get("judge_errors_excluded", 0) for c, m in summary.items()}
        parts.append(f"Judge errors excluded from judged metrics: {errors}\n")
    parts.append(
        "Denominators: 'answerable' counts refusals as not correct; 'answered' covers only "
        "non-refused answers. 'Partly' means exactly the judge verdict 'partly'. Latency and "
        "cost cover non-refused answers. n/a means the denominator was empty.\n"
    )
    out = results_dir / "summary.md"
    out.write_text("\n".join(parts), encoding="utf-8")
    return out
