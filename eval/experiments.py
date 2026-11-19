"""Run all experiment configs, judge answers, and produce a results table.

Usage:
    python eval/experiments.py --split dev
    python eval/experiments.py --split dev --configs hybrid_400,hybrid_400_rerank

Workflow:
1. For each config in CONFIGS (or --configs), generate candidate answers
   if eval/results/<config>_<split>.jsonl does not exist.
2. Judge all answers with the LLM judge.
3. Aggregate metrics and print a summary table.
4. Write eval/results/summary_<split>.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from generate_candidates import CONFIGS
from generate_candidates import run as generate_run
from judge import score_results_file

RESULTS_DIR = Path(__file__).parent / "results"
log = logging.getLogger(__name__)


def _aggregate(judgements: list) -> dict[str, float]:
    """Compute mean scores from a list of Judgement objects."""
    if not judgements:
        return {}
    faith = statistics.mean(j.faithfulness for j in judgements)
    relev = statistics.mean(j.relevance for j in judgements)
    citq = statistics.mean(j.citation_quality for j in judgements)
    refusal_rate = sum(1 for j in judgements if j.refused) / len(judgements)
    return {
        "n": len(judgements),
        "faithfulness_mean": round(faith, 3),
        "relevance_mean": round(relev, 3),
        "citation_quality_mean": round(citq, 3),
        "refusal_rate": round(refusal_rate, 3),
        "composite_mean": round((faith + relev + citq) / 3, 3),
    }


def _load_latency_cost(results_path: Path) -> tuple[float, float]:
    """Return (mean_latency_ms, total_cost_gbp) from a results file."""
    latencies: list[int] = []
    total_cost = 0.0
    for line in results_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        latencies.append(row.get("latency_ms", 0))
        total_cost += row.get("cost_gbp", 0.0)
    mean_lat = statistics.mean(latencies) if latencies else 0.0
    return round(mean_lat, 1), round(total_cost, 4)


def run_experiments(split: str, config_names: list[str]) -> None:
    """Generate candidates (if needed), judge, aggregate, and save summary."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict[str, float]] = {}

    for config_name in config_names:
        results_path = RESULTS_DIR / f"{config_name}_{split}.jsonl"

        if not results_path.exists():
            log.info("Generating candidates for config=%s split=%s", config_name, split)
            generate_run(split=split, config_name=config_name)
        else:
            log.info("Reusing existing results: %s", results_path.name)

        log.info("Judging %s ...", results_path.name)
        judgements = score_results_file(results_path)

        metrics = _aggregate(judgements)
        mean_lat, total_cost = _load_latency_cost(results_path)
        metrics["mean_latency_ms"] = mean_lat
        metrics["total_cost_gbp"] = total_cost

        summary[config_name] = metrics
        log.info(
            "%s → faith=%.2f rel=%.2f cit=%.2f refusal=%.0f%%",
            config_name,
            metrics["faithfulness_mean"],
            metrics["relevance_mean"],
            metrics["citation_quality_mean"],
            metrics["refusal_rate"] * 100,
        )

    # Print aligned results table
    _print_table(summary)

    # Save machine-readable summary
    output = RESULTS_DIR / f"summary_{split}.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("Summary saved → %s", output)


def _print_table(summary: dict[str, dict[str, float]]) -> None:
    header = f"{'Config':<25} {'Faith':>6} {'Relev':>6} {'Cit':>6} {'Comp':>6} {'Refusal':>8} {'Lat(ms)':>8} {'Cost(£)':>8}"
    print("\n" + header)
    print("-" * len(header))
    for config, m in summary.items():
        print(
            f"{config:<25} "
            f"{m['faithfulness_mean']:>6.2f} "
            f"{m['relevance_mean']:>6.2f} "
            f"{m['citation_quality_mean']:>6.2f} "
            f"{m['composite_mean']:>6.2f} "
            f"{m['refusal_rate'] * 100:>7.0f}% "
            f"{m['mean_latency_ms']:>8.0f} "
            f"{m['total_cost_gbp']:>8.4f}"
        )
    print()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )
    parser = argparse.ArgumentParser(description="Run eval experiments")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument(
        "--configs",
        default=",".join(CONFIGS.keys()),
        help="Comma-separated list of config names (default: all)",
    )
    args = parser.parse_args()
    config_names = [c.strip() for c in args.configs.split(",")]
    run_experiments(args.split, config_names)


if __name__ == "__main__":
    main()
