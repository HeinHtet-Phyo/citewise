"""Draw the four result charts into images/ from eval/results/summary_*.json."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path(__file__).parent / "results"
IMAGES = Path(__file__).parent.parent / "images"
ORDER = [
    "dense_400",
    "hybrid_400_rerank",
    "B_v2",
    "C_verify",
    "D_s2b",
    "E_k10",
    "L_dense",
    "L2_dense",
]


def _load(split: str) -> dict[str, dict]:
    path = RESULTS / f"summary_{split}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _bars(metric: str, title: str, ylabel: str, filename: str) -> None:
    """Grouped bars (dev and test) of one summary metric per config."""
    dev, test = _load("dev"), _load("test")
    configs = [c for c in ORDER if c in dev]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.4
    xs = range(len(configs))
    ax.bar([x - width / 2 for x in xs], [dev[c][metric] or 0 for c in configs], width, label="dev")
    ax.bar(
        [x + width / 2 for x in xs],
        [
            test[c][metric] if c in test and test[c][metric] is not None else float("nan")
            for c in configs
        ],
        width,
        label="test (only 3 configs run)",
    )
    ax.set_xticks(list(xs))
    ax.set_xticklabels(configs, rotation=30, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    fig.savefig(IMAGES / filename, dpi=150)
    plt.close(fig)


def _threshold_chart() -> None:
    """Refusal recall and false refusals across refusal thresholds (dev, hybrid_400_rerank)."""
    data = json.loads((RESULTS / "threshold_hybrid_400_rerank.json").read_text(encoding="utf-8"))
    ts = sorted(data["sweep"], key=float)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        [float(t) for t in ts],
        [data["sweep"][t]["refusal_recall"] for t in ts],
        label="refusal recall %",
    )
    ax.plot(
        [float(t) for t in ts],
        [data["sweep"][t]["false_refusal_rate"] for t in ts],
        label="false refusal %",
    )
    ax.set_xlabel("refusal threshold (dev split)")
    ax.set_ylabel("%")
    ax.set_title("Refusal recall vs false refusals")
    ax.legend()
    fig.tight_layout()
    fig.savefig(IMAGES / "refusal_threshold.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """Write all charts."""
    IMAGES.mkdir(exist_ok=True)
    _bars(
        "made_up_rate_pct",
        "Made-up rate by config (lower is better)",
        "% of answered",
        "made_up_by_config.png",
    )
    _bars(
        "answer_correct_pct_of_answerable",
        "Correct answers by config",
        "% of answerable",
        "correctness_by_config.png",
    )
    _bars(
        "retrieval_hit_rate_at_6_pct",
        "Retrieval hit rate @6 by config",
        "% of answerable",
        "retrieval_hit_rate.png",
    )
    _threshold_chart()


if __name__ == "__main__":
    main()
