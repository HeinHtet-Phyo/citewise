"""Measure Docker memory and cold start per config (idle and after 10 sequential requests).

Usage: python scripts/measure_memory.py L_dense hybrid_400_rerank
Needs the image `citewise:measure` and a local Qdrant on :6333 with citewise_400.
Writes eval/results/memory.json. The Nebius key is passed to the container via --env-file
and is never printed.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).parent.parent
OUT = ROOT / "eval" / "results" / "memory.json"
PORT = 8765


def _mem_mib(name: str) -> float:
    """Current container memory in MiB from docker stats."""
    out = subprocess.check_output(
        ["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", name], text=True
    )
    match = re.match(r"([\d.]+)(MiB|GiB)", out.split("/")[0].strip())
    assert match, out
    value = float(match.group(1))
    return value * 1024 if match.group(2) == "GiB" else value


def _questions() -> list[str]:
    rows = [json.loads(line) for line in (ROOT / "eval/questions/questions_dev.jsonl").open()]
    return [r["question"] for r in rows if not r["must_refuse"]][:10]


def measure(config: str) -> dict[str, float]:
    """Run one container for one config and return the measurements."""
    name = f"citewise-mem-{config.lower()}"
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    t0 = time.monotonic()
    subprocess.check_call(
        ["docker", "run", "-d", "--name", name, "-p", f"{PORT}:8000",
         "--env-file", str(ROOT / ".env"),
         "-e", "QDRANT_URL=http://host.docker.internal:6333", "-e", "QDRANT_API_KEY=",
         "-e", f"CITEWISE_CONFIG={config}", "-e", "LLM_CACHE=false",
         "-e", "RATE_LIMIT_PER_MINUTE=1000", "-e", "DAILY_BUDGET_GBP=100",
         "citewise:measure"],
        stdout=subprocess.DEVNULL,
    )  # fmt: skip
    while True:
        try:
            if httpx.get(f"http://localhost:{PORT}/health", timeout=2).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.5)
    cold_start_s = time.monotonic() - t0
    time.sleep(10)
    idle = _mem_mib(name)
    peak = idle
    first_request_s = 0.0
    for i, q in enumerate(_questions()):
        started = time.monotonic()
        httpx.post(
            f"http://localhost:{PORT}/ask", json={"question": q}, timeout=120
        ).raise_for_status()
        if i == 0:
            first_request_s = time.monotonic() - started
        peak = max(peak, _mem_mib(name))
    after = _mem_mib(name)
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    return {
        "cold_start_to_health_s": round(cold_start_s, 1),
        "first_request_s": round(first_request_s, 1),
        "idle_mib": round(idle, 1),
        "peak_mib_during_10_requests": round(peak, 1),
        "after_10_requests_mib": round(after, 1),
    }


def main() -> None:
    """Measure each config named on the command line."""
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    for config in sys.argv[1:]:
        results[config] = measure(config)
        print(config, results[config])
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
