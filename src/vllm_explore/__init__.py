from __future__ import annotations

import sys


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print("Hello from vllm-explore.")
        print("This repo is set up for local LLM workflow experiments and runtime benchmarks.")
        print("Run `uv run vllm-explore benchmark run --help` to use the benchmark harness.")
        return 0

    if args[0] == "benchmark":
        try:
            from vllm_explore.benchmark import BenchmarkError, run_cli

            return run_cli(args[1:])
        except BenchmarkError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

    print(f"Unknown command: {args[0]}", file=sys.stderr)
    print("Run `uv run vllm-explore benchmark run --help` for benchmark options.", file=sys.stderr)
    return 2
