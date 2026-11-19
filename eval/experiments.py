"""Run all experiment configs, judge answers, and produce a results table.

Usage:
    python eval/experiments.py --split dev
    python eval/experiments.py --split test --configs hybrid_400_rerank

Workflow:
1. For each config in CONFIGS (or --configs), generate candidate answers
   if eval/results/<config>_<split>.jsonl does not exist.
2. Judge all answers with the LLM judge.
3. Aggregate metrics and print a summary table.
4. Write eval/results/summary_<split>.json.
5. If split=dev, also tune the refusal threshold and write
   eval/results/threshold_<config>.json for the best config.
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

from citewise.retrieve import retrieve

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

        needs_generate = not results_path.exists()
        if not needs_generate:
            # Re-generate if any rows have generation errors (API was down on last run).
            error_count = sum(
                1
                for line in results_path.read_text(encoding="utf-8").splitlines()
                if line.strip() and json.loads(line).get("refusal_reason") == "generation_error"
            )
            if error_count:
                log.info(
                    "Re-generating %s — %d generation errors remain", results_path.name, error_count
                )
                results_path.unlink()
                needs_generate = True

        if needs_generate:
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

    # Threshold tuning on dev split only
    if split == "dev" and config_names:
        # Tune on the highest-composite config
        best_config = max(summary, key=lambda c: summary[c].get("composite_mean", 0.0))
        log.info("Tuning refusal threshold on dev split for best config: %s", best_config)
        all_judgements = []
        for config_name in config_names:
            results_path = RESULTS_DIR / f"{config_name}_{split}.jsonl"
            all_judgements = score_results_file(results_path)
        best_threshold = tune_threshold(
            best_config,
            RESULTS_DIR / f"{best_config}_{split}.jsonl",
            all_judgements,
        )
        log.info("Recommended refusal_threshold: %.2f", best_threshold)
        print(f"\n→ Recommended refusal_threshold for {best_config}: {best_threshold:.2f}")


def tune_threshold(config_name: str, results_path: Path, judgements: list) -> float:
    """Find the refusal threshold that maximises composite score on the dev split.

    Runs retrieval-only (no LLM) for each question to get the top-chunk score,
    then sweeps threshold values and picks the one with the best mean composite
    score on non-refused answers.

    Result written to eval/results/threshold_<config>.json.
    Returns the optimal threshold.
    """
    if config_name not in CONFIGS:
        log.warning("Config %s not found — skipping threshold tuning", config_name)
        return 0.0

    config = CONFIGS[config_name]

    # Load questions from results file (preserves order)
    rows = []
    for line in results_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))

    log.info("Retrieving scores for %d dev questions (threshold tuning)…", len(rows))
    top_scores: list[float] = []
    for row in rows:
        try:
            chunks = retrieve(row["question"], config)
            top_scores.append(chunks[0].score if chunks else 0.0)
        except Exception as exc:
            log.warning("Retrieval failed for %s: %s", row.get("question_id"), exc)
            top_scores.append(0.0)

    # Map question_id → (top_score, judge_composite)
    judge_map: dict[str, float] = {}
    for j in judgements:
        composite = (j.faithfulness + j.relevance + j.citation_quality) / 3.0
        judge_map[j.question_id] = composite

    id_list = [r.get("question_id", f"q{i}") for i, r in enumerate(rows)]

    best_threshold = 0.0
    best_score = -1.0
    threshold_results: dict[str, object] = {}

    for t in [round(x * 0.05, 2) for x in range(21)]:  # 0.00 to 1.00 in steps of 0.05
        kept = [
            judge_map[qid]
            for qid, sc in zip(id_list, top_scores, strict=False)
            if sc >= t and qid in judge_map
        ]
        refused_count = sum(1 for sc in top_scores if sc < t)
        mean_comp = statistics.mean(kept) if kept else 0.0
        threshold_results[str(t)] = {
            "mean_composite": round(mean_comp, 3),
            "n_answered": len(kept),
            "n_refused": refused_count,
        }
        if mean_comp > best_score:
            best_score = mean_comp
            best_threshold = t

    output = {
        "config": config_name,
        "best_threshold": best_threshold,
        "best_composite": round(best_score, 3),
        "sweep": threshold_results,
    }
    out_path = RESULTS_DIR / f"threshold_{config_name}.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    log.info(
        "Best threshold for %s: %.2f (composite=%.3f) → %s",
        config_name,
        best_threshold,
        best_score,
        out_path,
    )
    return best_threshold


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
