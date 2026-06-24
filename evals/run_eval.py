"""
Ground Truth Evaluation Harness
--------------------------------
Sends each case from ground_truth.json to the live agent API and checks:
  1. must_contain  — every keyword must appear in the answer (case-insensitive)
  2. must_not_contain — none of these keywords should appear
  3. faithfulness_ok  — agent's own LLM-as-judge flag must be True (RAG cases only)

Usage:
  python evals/run_eval.py                  # all cases
  python evals/run_eval.py --category billing   # filter by category
  python evals/run_eval.py --id gt_001      # single case
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests

BASE_URL = "http://localhost:8000"
CHAT_URL = f"{BASE_URL}/v1/agent/chat"
GROUND_TRUTH_FILE = Path(__file__).parent / "ground_truth.json"

# ANSI colours
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


def check_health():
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        r.raise_for_status()
    except Exception as e:
        print(f"{RED}Backend not reachable at {BASE_URL}: {e}{RESET}")
        sys.exit(1)


def call_agent(case: dict, retries: int = 2) -> dict:
    payload = {
        "merchant_id": case["merchant_id"],
        "tier": case["tier"],
        "message": case["question"],
    }
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(CHAT_URL, json=payload, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as e:
            last_err = e
            if attempt < retries and resp.status_code == 500:
                print(f"       {YELLOW}⟳ 500 error, retrying ({attempt + 1}/{retries})...{RESET}")
                time.sleep(2)
            else:
                raise
    raise last_err


def evaluate_case(case: dict, response: dict) -> dict:
    answer = response.get("agent_response", "")
    faithfulness_ok = response.get("faithfulness_ok")
    answer_lower = answer.lower()

    failures = []

    # Check must_contain keywords
    for keyword in case.get("must_contain", []):
        if keyword.lower() not in answer_lower:
            failures.append(f"missing '{keyword}'")

    # Check must_not_contain keywords
    for keyword in case.get("must_not_contain", []):
        if keyword.lower() in answer_lower:
            failures.append(f"hallucinated '{keyword}'")

    # Check faithfulness flag for non-guardrail cases
    if case["category"] != "guardrail" and faithfulness_ok is False:
        failures.append("faithfulness check failed (LLM-as-judge flagged it)")

    return {
        "id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "tier": case["tier"],
        "answer": answer,
        "faithfulness_ok": faithfulness_ok,
        "passed": len(failures) == 0,
        "failures": failures,
    }


def print_result(result: dict, verbose: bool = True):
    icon  = f"{GREEN}PASS{RESET}" if result["passed"] else f"{RED}FAIL{RESET}"
    print(f"[{icon}] {result['id']} ({result['category']}) — tier={result['tier']}")
    print(f"       Q: {result['question']}")

    if not result["passed"]:
        for f in result["failures"]:
            print(f"       {YELLOW}✗ {f}{RESET}")

    if verbose and not result["passed"]:
        # Show a truncated answer to help debug
        snippet = result["answer"][:300].replace("\n", " ")
        print(f"       A: {snippet}{'...' if len(result['answer']) > 300 else ''}")

    print()


def print_summary(results: list[dict], elapsed: float):
    passed = sum(1 for r in results if r["passed"])
    total  = len(results)
    pct    = (passed / total * 100) if total else 0

    colour = GREEN if pct == 100 else (YELLOW if pct >= 70 else RED)
    print("=" * 60)
    print(f"{BOLD}EVAL SUMMARY{RESET}")
    print(f"  Total cases : {total}")
    print(f"  Passed      : {colour}{passed}{RESET}")
    print(f"  Failed      : {RED}{total - passed}{RESET}")
    print(f"  Pass rate   : {colour}{pct:.1f}%{RESET}")
    print(f"  Time        : {elapsed:.1f}s")

    # Category breakdown
    categories = {}
    for r in results:
        cat = r["category"]
        categories.setdefault(cat, {"pass": 0, "total": 0})
        categories[cat]["total"] += 1
        if r["passed"]:
            categories[cat]["pass"] += 1

    print("\n  By category:")
    for cat, counts in sorted(categories.items()):
        cat_pct = counts["pass"] / counts["total"] * 100
        c = GREEN if cat_pct == 100 else RED
        print(f"    {cat:<20} {c}{counts['pass']}/{counts['total']}{RESET}")

    print("=" * 60)

    # Write JSON report
    report_path = Path(__file__).parent / "last_eval_report.json"
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pass_rate": round(pct, 1),
        "passed": passed,
        "total": total,
        "results": results,
    }
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nFull report saved → {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Run ground truth eval against live agent")
    parser.add_argument("--category", help="Filter by category (e.g. billing, returns)")
    parser.add_argument("--id",       help="Run a single case by ID (e.g. gt_001)")
    parser.add_argument("--quiet",    action="store_true", help="Hide answer snippet on failure")
    args = parser.parse_args()

    check_health()

    cases = json.loads(GROUND_TRUTH_FILE.read_text())

    if args.id:
        cases = [c for c in cases if c["id"] == args.id]
        if not cases:
            print(f"{RED}No case found with id={args.id}{RESET}")
            sys.exit(1)

    if args.category:
        cases = [c for c in cases if c["category"] == args.category]
        if not cases:
            print(f"{RED}No cases found for category={args.category}{RESET}")
            sys.exit(1)

    print(f"{BOLD}Running {len(cases)} eval case(s) against {BASE_URL}{RESET}\n")

    results = []
    start = time.time()

    for case in cases:
        try:
            response = call_agent(case)
            result   = evaluate_case(case, response)
        except Exception as e:
            result = {
                "id": case["id"],
                "category": case["category"],
                "question": case["question"],
                "tier": case["tier"],
                "answer": "",
                "faithfulness_ok": None,
                "passed": False,
                "failures": [f"API error: {e}"],
            }

        results.append(result)
        print_result(result, verbose=not args.quiet)

    elapsed = time.time() - start
    print_summary(results, elapsed)

    # Exit non-zero if any case failed (useful in CI)
    sys.exit(0 if all(r["passed"] for r in results) else 1)


if __name__ == "__main__":
    main()
