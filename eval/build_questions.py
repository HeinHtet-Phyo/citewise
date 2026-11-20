"""Rebuild the evaluation question set with ground truth.

Run from the repo root:  python eval/build_questions.py [--dry-run]
Needs NEBIUS_API_KEY for the drafting step (responses are cached in .llm_cache/eval_questions).
"""

from __future__ import annotations

import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from build_questions_corpus import Corpus, load_chunks  # noqa: E402
from build_questions_draft import draft_existing, draft_new, pick_seeds  # noqa: E402
from build_questions_llm import MODEL, Spend  # noqa: E402
from build_questions_readme import write_readme  # noqa: E402
from build_questions_refusals import ADVERSARIAL, UNANSWERABLE  # noqa: E402
from question_schema import EvalQuestion  # noqa: E402

SEED = 20261119
N_ANSWERABLE_TARGET = 150
N_NEW_SEEDS = 140
DEV_FRACTION = 1 / 3
OUT = Path("eval/questions")
TOPICS = {"benefits", "employment", "housing", "transport"}
TYPES = {"single_fact", "multi_page", "how_to"}


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def _accept(corpus: Corpus, d: dict[str, object] | None) -> tuple[dict[str, object] | None, str]:
    """Validate a draft; return (clean record, reason) where record is None on failure."""
    if not d:
        return None, "unparseable_reply"
    if d.get("answerable") is False or d.get("usable") is False:
        return None, "model_says_not_answerable"
    answer, evidence, kind = d.get("reference_answer"), d.get("evidence"), d.get("type")
    if not isinstance(answer, str) or not answer.strip() or not isinstance(evidence, list):
        return None, "malformed_draft"
    if kind not in TYPES:
        return None, "bad_type"
    ev = [e for e in evidence if isinstance(e, dict)]
    urls = list(dict.fromkeys(str(e.get("url", "")) for e in ev))
    if not 1 <= len(urls) <= 2:
        return None, "bad_gold_url_count"
    reason = corpus.verify(answer, [{k: str(v) for k, v in e.items()} for e in ev])
    if reason:
        return None, reason.split(":")[0]
    if kind == "multi_page" and len(urls) != 2:
        kind = "single_fact"
    if kind != "multi_page" and len(urls) == 2:
        kind = "multi_page"
    return {"reference_answer": answer.strip(), "gold_urls": urls, "type": kind}, "ok"


def _load_originals() -> list[dict[str, str]]:
    """Read the original 150 questions, archiving them on first run so reruns are repeatable."""
    archive = OUT / "original_questions.jsonl"
    if not archive.exists():
        rows = [q for f in ("dev", "test") for q in _read(OUT / f"questions_{f}.jsonl")]
        archive.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return _read(archive)


def _answerable_existing(
    corpus: Corpus, originals: list[dict[str, str]], drops: Counter[str]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for q in originals:
        rec, why = _accept(corpus, draft_existing(corpus, q["question"]))
        if rec is None:
            drops[f"existing:{why}"] += 1
            continue
        rows.append({"question_id": q["question_id"], "question": q["question"],
                     "topic": q["topic"], **rec})  # fmt: skip
    return rows


def _new_questions(
    corpus: Corpus, have: int, existing: list[dict[str, object]], drops: Counter[str]
) -> list[dict[str, object]]:
    rng = random.Random(SEED)
    wanted = ["multi_page", "how_to", "multi_page", "single_fact"]
    known = {str(r["question"]).lower() for r in existing}
    rows: list[dict[str, object]] = []
    for n, seed in enumerate(pick_seeds(corpus, N_NEW_SEEDS, rng)):
        if have + len(rows) >= N_ANSWERABLE_TARGET:
            break
        want = wanted[n % len(wanted)]
        d = draft_new(corpus, seed, want, with_related=want == "multi_page")
        rec, why = _accept(corpus, d)
        if rec is None or d is None or d.get("topic") not in TOPICS:
            drops[f"new:{why if rec is None else 'bad_topic'}"] += 1
            continue
        if str(d["question"]).lower() in known:
            drops["new:duplicate"] += 1
            continue
        known.add(str(d["question"]).lower())
        rows.append({"question_id": "", "question": str(d["question"]).strip(),
                     "topic": d["topic"], **rec})  # fmt: skip
    return rows


def _refusals() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for kind, items in (("unanswerable", UNANSWERABLE), ("adversarial", ADVERSARIAL)):
        for topic, question, notes in items:
            rows.append({"question_id": "", "question": question, "topic": topic, "type": kind,
                         "reference_answer": None, "gold_urls": [], "notes": notes})  # fmt: skip
    return rows


def _assign_splits(rows: list[dict[str, object]]) -> None:
    """Stratified dev/test split by (type, topic) using the fixed seed."""
    rng = random.Random(SEED)
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for r in rows:
        groups[(str(r["type"]), str(r["topic"]))].append(r)
    for key in sorted(groups):
        members = groups[key]
        rng.shuffle(members)
        n_dev = round(len(members) * DEV_FRACTION)
        for i, r in enumerate(members):
            r["split"] = "dev" if i < n_dev else "test"


def _write(rows: list[dict[str, object]]) -> list[EvalQuestion]:
    qs = [
        EvalQuestion(
            **{
                **r,
                "must_refuse": r["type"] in {"unanswerable", "adversarial"},
                "notes": r.get("notes"),
            }
        )  # type: ignore[arg-type]
        for r in rows
    ]
    for split in ("dev", "test"):
        with (OUT / f"questions_{split}.jsonl").open("w", encoding="utf-8") as fh:
            for q in (x for x in qs if x.split == split):
                fh.write(q.model_dump_json() + "\n")
    return qs


def _hand_check(qs: list[EvalQuestion]) -> None:
    pool = [q for q in qs if not q.must_refuse]
    sample = random.Random(SEED).sample(pool, 30)
    with (OUT / "hand_check_questions.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["question_id", "question", "reference_answer", "gold_url", "human_ok"])
        for q in sample:
            w.writerow([q.question_id, q.question, q.reference_answer, " | ".join(q.gold_urls), ""])


def _counts(qs: list[EvalQuestion]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for name in ("split", "type", "topic"):
        out[name] = dict(sorted(Counter(getattr(q, name) for q in qs).items()))
    out["split_x_type"] = dict(sorted(Counter(f"{q.split}/{q.type}" for q in qs).items()))
    out["split_x_topic"] = dict(sorted(Counter(f"{q.split}/{q.topic}" for q in qs).items()))
    return out


def main() -> None:
    """Draft, verify, split and write the question files and reports."""
    corpus = Corpus(load_chunks())
    originals = _load_originals()
    n_original = len(originals)
    drops: Counter[str] = Counter()
    kept = _answerable_existing(corpus, originals, drops)
    new = _new_questions(corpus, len(kept), kept, drops)
    rows = kept + new + _refusals()
    next_id = 151
    for r in rows:
        if not r["question_id"]:
            r["question_id"] = f"q{next_id:03d}"
            next_id += 1
    _assign_splits(rows)
    qs = _write(rows)
    _hand_check(qs)
    report = {
        "seed": SEED, "model": MODEL, "original_questions": n_original,
        "existing_kept": len(kept), "new_kept": len(new), "refusals_added": len(rows) - len(kept) - len(new),
        "dropped_by_reason": dict(sorted(drops.items())), "total": len(qs),
        "counts": _counts(qs), "nebius_calls": Spend.calls, "cache_hits": Spend.cached,
        "nebius_spend_usd": round(Spend.usd, 4),
    }  # fmt: skip
    (OUT / "verification_report.json").write_text(json.dumps(report, indent=2) + "\n")
    write_readme(OUT / "README.md", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
