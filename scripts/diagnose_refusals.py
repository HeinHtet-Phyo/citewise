"""Show where each question is refused: input guard, score, LLM refusal or claim verifier."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))

from generate_candidates import CONFIGS

from citewise.generate import context_text, generate_answer
from citewise.guardrails import check_input
from citewise.rag import _REFUSAL_PREFIX, _verify_and_repair
from citewise.retrieve import retrieve

QUESTIONS = [
    "What is the State Pension age?",
    "How much notice must my landlord give?",
    "How much is the basic State Pension per week?",
    "How long is the notice period for redundancy?",
    "What is statutory sick pay per week?",
    "How much is the National Minimum Wage?",
    "How many days of paid holiday am I entitled to?",
    "How much is statutory maternity pay?",
    "How do I apply for Universal Credit?",
    "When does my MOT expire?",
    "How much does a UK passport cost?",
    "How do I renew my driving licence?",
    "How much is Council Tax?",
    "What is the Pension Credit age?",
    "How much is Child Benefit?",
    "Can my landlord increase my rent?",
    "What is the Personal Independence Payment?",
    "How do I claim Jobseeker's Allowance?",
    "What is the vehicle tax rate?",
    "How long can I get statutory paternity leave?",
]


def main(config_name: str) -> None:
    """Run each question through the pipeline steps and print the outcome."""
    config = CONFIGS[config_name]
    for q in QUESTIONS:
        safe, why = check_input(q)
        if not safe:
            print(f"INPUT-REFUSED ({why}) | {q}")
            continue
        chunks = retrieve(q, config)
        top = f"{chunks[0].score:.3f}" if chunks else "none"
        urls = ",".join(c.chunk.url.removeprefix("https://www.gov.uk/")[:28] for c in chunks[:3])
        text, _, _ = generate_answer(q, chunks, config)
        if text.strip().startswith(_REFUSAL_PREFIX):
            outcome = "LLM-REFUSED"
        elif config.verify_claims:
            _, _, path = _verify_and_repair(q, text, chunks, config)
            outcome = f"verifier={path}"
        else:
            outcome = "answered"
        n_ctx = len(chunks)
        print(f"{outcome:20} top={top} ctx={n_ctx} | {q} | {urls}")
        _ = context_text


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "L_dense")
