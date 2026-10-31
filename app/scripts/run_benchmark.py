"""Run the 100-prompt benchmark against each comparison arm via the API."""

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

from app.benchmarks.prompts import BENCHMARK_PROMPTS


ARMS = ("normal_rag", "llmlingua_only", "toon_only", "toon_plus_skill")
FAILURE_FIELDS = ("batch_id", "prompt_id", "arm", "error", "failed_at")


def post_chat(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {detail}") from error
    except URLError as error:
        raise RuntimeError(f"Request failed: {error.reason}") from error

    try:
        result = json.loads(body)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"API returned invalid JSON: {body[:500]}") from error
    if not isinstance(result, dict) or result.get("status") != "success":
        raise RuntimeError(f"API reported failure: {result!r}")
    return result


def parse_case(value: str) -> tuple[str, str]:
    try:
        prompt_id, arm = value.split(":", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("case must look like P001:normal_rag") from error
    if arm not in ARMS:
        raise argparse.ArgumentTypeError(f"unknown arm {arm!r}; choose from {', '.join(ARMS)}")
    return prompt_id, arm


def append_failure(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FAILURE_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)
        file.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url", default="http://127.0.0.1:8000",
        help="Base URL of the running FastAPI service (default: %(default)s)",
    )
    parser.add_argument(
        "--failure-log", type=Path, default=Path("benchmark_failures.csv"),
        help="Append prompt ID, failed arm, and error details here",
    )
    parser.add_argument(
        "--timeout", type=float, default=180,
        help="Per-request timeout in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--prompt-id", action="append", dest="prompt_ids",
        help="Run all four arms for this prompt; may be supplied more than once",
    )
    parser.add_argument(
        "--case", action="append", type=parse_case,
        help="Run only one prompt/arm pair for retry (e.g. P001:toon_only)",
    )
    args = parser.parse_args()

    prompts_by_id = {prompt["id"]: prompt for prompt in BENCHMARK_PROMPTS}
    requested_ids = set(args.prompt_ids or [])
    requested_cases = args.case or []
    case_ids = {prompt_id for prompt_id, _ in requested_cases}
    unknown_ids = (requested_ids | case_ids) - prompts_by_id.keys()
    if unknown_ids:
        parser.error(f"unknown prompt ID(s): {', '.join(sorted(unknown_ids))}")

    if requested_cases:
        cases = [(prompts_by_id[prompt_id], arm) for prompt_id, arm in requested_cases]
        cases.extend(
            (prompts_by_id[prompt_id], arm)
            for prompt_id in requested_ids
            for arm in ARMS
        )
    else:
        selected = [
            prompt for prompt in BENCHMARK_PROMPTS
            if not requested_ids or prompt["id"] in requested_ids
        ]
        cases = [(prompt, arm) for prompt in selected for arm in ARMS]

    endpoint = args.base_url.rstrip("/") + "/api/v1/agent/chat"
    batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    failures = 0
    print(f"Batch {batch_id}: {len(cases)} requests to {endpoint}", flush=True)

    for index, (prompt, arm) in enumerate(cases, start=1):
        prompt_id = prompt["id"]
        run_id = f"{batch_id}-{prompt_id}-{arm}"
        category = prompt["category"]
        payload = {
            "message": prompt["prompt"],
            "thread_id": run_id,
            "process_type": arm,
            "run_id": run_id,
            "prompt_id": prompt_id,
            "task_type": category.value if hasattr(category, "value") else category,
            "arm": arm,
            "repeat": 1,
        }
        try:
            result = post_chat(endpoint, payload, args.timeout)
            print(
                f"[{index}/{len(cases)}] OK {prompt_id} {arm} "
                f"run_id={result.get('run_id', run_id)}",
                flush=True,
            )
        except Exception as error:
            failures += 1
            append_failure(args.failure_log, {
                "batch_id": batch_id,
                "prompt_id": prompt_id,
                "arm": arm,
                "error": str(error),
                "failed_at": datetime.now(timezone.utc).isoformat(),
            })
            print(f"[{index}/{len(cases)}] FAIL {prompt_id} {arm}: {error}", flush=True)

        if index < len(cases):
            time.sleep(0.1)

    print(
        f"Finished: {len(cases) - failures} succeeded, {failures} failed. "
        f"Failure log: {args.failure_log}",
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
