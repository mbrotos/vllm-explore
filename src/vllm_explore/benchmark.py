from __future__ import annotations

import argparse
import csv
import json
import os
import select
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import tiktoken
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]


TASK_LIBRARY: dict[str, dict[str, str]] = {
    "calendar_reasoning": {
        "title": "Calendar Reasoning",
        "facts": """
Mina is scheduling a 90 minute review.
Mina is unavailable on Monday.
The review must finish before 17:00.
If the review is on Thursday, it must start after 15:00.
Jae can only attend on Tuesday or Thursday.
Available candidate slots are:
- Tuesday 13:00 to 14:30
- Thursday 14:30 to 16:00
- Thursday 15:30 to 17:00
Choose the single valid slot.
""".strip(),
        "question": "Which slot should Mina choose? Return the exact slot text.",
    },
    "ledger_reasoning": {
        "title": "Ledger Reasoning",
        "facts": """
Start from a subtotal of 2,480.
Apply a discount of 12 percent to the subtotal.
Add a flat compliance fee of 95.
Add shipping of 40.
Do not tax shipping.
Apply 8 percent tax only after the discount and only to the discounted subtotal plus the compliance fee.
Round to two decimals.
""".strip(),
        "question": "What is the final amount? Return only the numeric total.",
    },
    "policy_reasoning": {
        "title": "Policy Reasoning",
        "facts": """
Policy rules:
- Requests tagged urgent may skip queue review only if the requester is a director.
- External sharing is forbidden for restricted documents.
- Summaries of restricted documents may be shared internally.
- Legal approval is required before any external sharing of confidential documents.

Candidate actions:
A. Share a restricted document with an external partner because the request is urgent.
B. Share an internal summary of a restricted document with the finance team.
C. Share a confidential document externally without legal approval because the requester is a director.
Choose the single policy-compliant action.
""".strip(),
        "question": "Which action is compliant? Return only A, B, or C.",
    },
    "wiki_ada_profile": {
        "title": "Wikipedia Long Context: Ada Lovelace",
        "context_files": [
            {
                "title": "Reference bundle",
                "path": "benchmarks/context/wiki_computing_pioneers.txt",
            }
        ],
        "question": (
            "Based only on the reference bundle, which person translated Menabrea's article "
            "about the Analytical Engine, added extensive notes, and argued that such a machine "
            "could work on symbols beyond arithmetic? Return only the full name."
        ),
    },
    "wiki_turing_profile": {
        "title": "Wikipedia Long Context: Alan Turing",
        "context_files": [
            {
                "title": "Reference bundle",
                "path": "benchmarks/context/wiki_computing_pioneers.txt",
            }
        ],
        "question": (
            "Based only on the reference bundle, which person combined wartime Enigma "
            "cryptanalysis with later work on the Automatic Computing Engine, the Turing test, "
            "and morphogenesis? Return only the full name."
        ),
    },
    "wiki_hopper_profile": {
        "title": "Wikipedia Long Context: Grace Hopper",
        "context_files": [
            {
                "title": "Reference bundle",
                "path": "benchmarks/context/wiki_computing_pioneers.txt",
            }
        ],
        "question": (
            "Based only on the reference bundle, which person served in the U.S. Navy, worked on "
            "the Harvard Mark I and UNIVAC, and pushed machine-independent English-like "
            "programming that influenced FLOW-MATIC and COBOL? Return only the full name."
        ),
    },
    "wiki_space_program_bridge": {
        "title": "Wikipedia Long Context: Space Program Bridge",
        "context_files": [
            {
                "title": "Reference bundle",
                "path": "benchmarks/context/wiki_us_space_programs.txt",
            }
        ],
        "question": (
            "Based only on the reference bundle, which U.S. human spaceflight program acted as "
            "the bridge between Mercury and Apollo by focusing on long-duration flight, EVA, "
            "rendezvous, and docking with a two-astronaut crew? Return only the program name."
        ),
    },
}


DISTRACTOR_TEMPLATE = (
    "Archive note {index}: team={team}; quarter={quarter}; "
    "status={status}; region={region}; priority={priority}; "
    "retention={retention}; checksum={checksum}."
)

DISTRACTOR_TEAMS = [
    "alpha",
    "beta",
    "gamma",
    "delta",
    "omega",
    "sigma",
]
DISTRACTOR_REGIONS = ["north", "south", "east", "west", "central"]
DISTRACTOR_STATUSES = ["draft", "queued", "approved", "archived"]
DISTRACTOR_PRIORITIES = ["low", "medium", "high"]
DISTRACTOR_RETENTION = ["30d", "90d", "180d", "365d"]


@dataclass
class TargetConfig:
    label: str
    runtime: str
    kind: str
    model: str
    tokenizer: str
    size: str
    base_url: str | None = None
    start_command: str | None = None
    health_url: str | None = None
    endpoint: str = "/v1/chat/completions"
    python_bin: str | None = None
    ollama_model: str | None = None
    keep_alive: str | None = None
    trust_remote_code: bool = False
    startup_timeout_seconds: float = 600.0
    request_timeout_seconds: float = 600.0


@dataclass
class BenchmarkDefaults:
    task_names: list[str]
    input_token_targets: list[int]
    max_output_tokens: int
    repeats: int
    temperature: float


@dataclass
class ResultRow:
    timestamp: str
    label: str
    runtime: str
    size: str
    model: str
    tokenizer: str
    task: str
    input_token_target: int
    input_tokens: int
    prompt_tokens: int | None
    output_tokens: int
    ttft_seconds: float | None
    generation_seconds: float
    output_tps: float | None
    total_seconds: float
    repeat_index: int
    generated_text: str


@dataclass
class LoadedTokenizer:
    kind: str
    handle: Any


class BenchmarkError(RuntimeError):
    pass


def run_cli(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="vllm-explore benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the benchmark harness")
    run_parser.add_argument("--config", required=True, help="Path to a benchmark JSON config")
    run_parser.add_argument(
        "--match",
        help="Only run targets whose label contains this substring",
    )
    run_parser.add_argument(
        "--task",
        action="append",
        help="Only run the named task. Can be passed multiple times.",
    )
    run_parser.add_argument(
        "--input-token-target",
        dest="input_token_targets",
        action="append",
        type=int,
        help="Only run the specified approximate input token target. Can be passed multiple times.",
    )
    run_parser.add_argument(
        "--output-dir",
        help="Directory for benchmark results (defaults to a timestamped folder under results/benchmarks)",
    )
    run_parser.add_argument(
        "--csv-path",
        help="Optional path for a single consolidated CSV output",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and print the expanded benchmark matrix without running models",
    )

    args = parser.parse_args(argv)
    if args.command == "run":
        return run_benchmark_command(
            config_path=Path(args.config),
            match=args.match,
            task_filters=args.task,
            input_token_filters=args.input_token_targets,
            output_dir=Path(args.output_dir) if args.output_dir else None,
            csv_path=Path(args.csv_path) if args.csv_path else None,
            dry_run=args.dry_run,
        )
    return 2


def run_benchmark_command(
    config_path: Path,
    match: str | None,
    task_filters: list[str] | None,
    input_token_filters: list[int] | None,
    output_dir: Path | None,
    csv_path: Path | None,
    dry_run: bool,
) -> int:
    config = load_config(config_path)
    defaults = load_defaults(config)
    defaults = apply_default_filters(defaults, task_filters, input_token_filters)
    targets = load_targets(config, match)
    if not targets:
        raise BenchmarkError("No targets matched the current config and filters.")

    cases = [
        (target, task_name, input_token_target, repeat_index)
        for target in targets
        for task_name in defaults.task_names
        for input_token_target in defaults.input_token_targets
        for repeat_index in range(1, defaults.repeats + 1)
    ]

    if dry_run:
        print(f"Config: {config_path}")
        print(f"Targets: {len(targets)}")
        print(f"Cases: {len(cases)}")
        for target, task_name, input_token_target, repeat_index in cases:
            print(
                f"- {target.label}: task={task_name}, input_tokens~{input_token_target}, repeat={repeat_index}"
            )
        return 0

    output_dir = output_dir or default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[ResultRow] = []
    tokenizer_cache: dict[tuple[str, bool], LoadedTokenizer] = {}

    for target in targets:
        tokenizer = load_tokenizer(tokenizer_cache, target)
        prompts: dict[tuple[str, int], tuple[str, int]] = {}
        for task_name in defaults.task_names:
            for input_token_target in defaults.input_token_targets:
                prompts[(task_name, input_token_target)] = build_prompt(
                    tokenizer=tokenizer,
                    task_name=task_name,
                    input_token_target=input_token_target,
                )

        runner = make_runner(target, tokenizer)
        print(f"Starting target {target.label} ({target.runtime}, {target.model})")
        with runner:
            for task_name in defaults.task_names:
                for input_token_target in defaults.input_token_targets:
                    prompt_text, actual_input_tokens = prompts[(task_name, input_token_target)]
                    for repeat_index in range(1, defaults.repeats + 1):
                        result = runner.generate(
                            prompt=prompt_text,
                            max_output_tokens=defaults.max_output_tokens,
                            temperature=defaults.temperature,
                        )
                        row = ResultRow(
                            timestamp=datetime.utcnow().isoformat(timespec="seconds") + "Z",
                            label=target.label,
                            runtime=target.runtime,
                            size=target.size,
                            model=target.model,
                            tokenizer=target.tokenizer,
                            task=task_name,
                            input_token_target=input_token_target,
                            input_tokens=actual_input_tokens,
                            prompt_tokens=result.get("prompt_tokens"),
                            output_tokens=result["output_tokens"],
                            ttft_seconds=result.get("ttft_seconds"),
                            generation_seconds=result["generation_seconds"],
                            output_tps=result.get("output_tps"),
                            total_seconds=result["total_seconds"],
                            repeat_index=repeat_index,
                            generated_text=result["generated_text"],
                        )
                        results.append(row)
                        print_result(row)

    write_results(output_dir, results, csv_path)
    print(f"Wrote benchmark results to {output_dir}")
    if csv_path:
        print(f"Wrote consolidated CSV to {csv_path}")
    return 0


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise BenchmarkError(f"Config does not exist: {config_path}")
    try:
        return json.loads(config_path.read_text())
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"Invalid JSON in {config_path}: {exc}") from exc


def load_defaults(config: dict[str, Any]) -> BenchmarkDefaults:
    defaults = config.get("defaults", {})
    task_names = defaults.get("tasks", list(TASK_LIBRARY))
    for task_name in task_names:
        if task_name not in TASK_LIBRARY:
            raise BenchmarkError(f"Unknown task: {task_name}")

    input_token_targets = defaults.get("input_token_targets", [512, 2048, 8192])
    return BenchmarkDefaults(
        task_names=task_names,
        input_token_targets=input_token_targets,
        max_output_tokens=int(defaults.get("max_output_tokens", 96)),
        repeats=int(defaults.get("repeats", 1)),
        temperature=float(defaults.get("temperature", 0.0)),
    )


def apply_default_filters(
    defaults: BenchmarkDefaults,
    task_filters: list[str] | None,
    input_token_filters: list[int] | None,
) -> BenchmarkDefaults:
    task_names = defaults.task_names
    if task_filters:
        unknown_tasks = [task for task in task_filters if task not in TASK_LIBRARY]
        if unknown_tasks:
            raise BenchmarkError(f"Unknown task filters: {', '.join(unknown_tasks)}")
        task_names = [task for task in defaults.task_names if task in task_filters]
        if not task_names:
            raise BenchmarkError("No tasks matched the provided task filters.")

    input_token_targets = defaults.input_token_targets
    if input_token_filters:
        input_token_targets = [
            target for target in defaults.input_token_targets if target in input_token_filters
        ]
        if not input_token_targets:
            raise BenchmarkError("No input token targets matched the provided filters.")

    return BenchmarkDefaults(
        task_names=task_names,
        input_token_targets=input_token_targets,
        max_output_tokens=defaults.max_output_tokens,
        repeats=defaults.repeats,
        temperature=defaults.temperature,
    )


def load_targets(config: dict[str, Any], match: str | None) -> list[TargetConfig]:
    raw_targets = config.get("targets", [])
    if not raw_targets:
        raise BenchmarkError("Config must contain at least one target.")

    targets: list[TargetConfig] = []
    for raw in raw_targets:
        target = TargetConfig(
            label=raw["label"],
            runtime=raw["runtime"],
            kind=raw["kind"],
            model=raw["model"],
            tokenizer=raw.get("tokenizer", raw["model"]),
            size=raw["size"],
            base_url=raw.get("base_url"),
            start_command=raw.get("start_command"),
            health_url=raw.get("health_url"),
            endpoint=raw.get("endpoint", "/v1/chat/completions"),
            python_bin=raw.get("python_bin"),
            ollama_model=raw.get("ollama_model"),
            keep_alive=raw.get("keep_alive"),
            trust_remote_code=bool(raw.get("trust_remote_code", False)),
            startup_timeout_seconds=float(raw.get("startup_timeout_seconds", 600.0)),
            request_timeout_seconds=float(raw.get("request_timeout_seconds", 600.0)),
        )
        if match and match not in target.label:
            continue
        targets.append(target)
    return targets


def load_tokenizer(
    cache: dict[tuple[str, bool], LoadedTokenizer], target: TargetConfig
) -> LoadedTokenizer:
    cache_key = (target.tokenizer, target.trust_remote_code)
    if cache_key not in cache:
        if target.tokenizer.startswith("openai:"):
            cache[cache_key] = LoadedTokenizer(
                kind="tiktoken",
                handle=tiktoken.get_encoding(target.tokenizer.split(":", maxsplit=1)[1]),
            )
        else:
            cache[cache_key] = LoadedTokenizer(
                kind="transformers",
                handle=AutoTokenizer.from_pretrained(
                    target.tokenizer,
                    trust_remote_code=target.trust_remote_code,
                ),
            )
    return cache[cache_key]


def build_prompt(
    tokenizer: LoadedTokenizer, task_name: str, input_token_target: int
) -> tuple[str, int]:
    task = TASK_LIBRARY[task_name]
    preface = (
        "You are participating in a local LLM benchmark. "
        "Use only the provided notes and reference material. "
        "Reason privately and return only compact JSON with keys \"answer\" and \"reason\"."
    )
    base_sections = [
        preface,
        f"Task: {task['title']}",
    ]

    if task.get("facts"):
        base_sections.extend(["Relevant facts:", task["facts"]])

    for context_file in task.get("context_files", []):
        base_sections.extend(
            [
                f"{context_file['title']}:",
                load_context_text(context_file["path"]),
            ]
        )

    base_sections.append(f"Question: {task['question']}")

    filler_blocks: list[str] = []
    prompt_text = render_prompt(base_sections, filler_blocks)
    while count_tokens(tokenizer, prompt_text) < input_token_target:
        filler_blocks.append(make_distractor_block(len(filler_blocks) + 1))
        prompt_text = render_prompt(base_sections, filler_blocks)

    return prompt_text, count_tokens(tokenizer, prompt_text)


def render_prompt(base_sections: list[str], filler_blocks: list[str]) -> str:
    midpoint = len(filler_blocks) // 2
    before = filler_blocks[:midpoint]
    after = filler_blocks[midpoint:]
    sections = []
    if before:
        sections.extend(["Background notes:", *before])
    sections.extend(base_sections)
    if after:
        sections.extend(["Supplemental notes:", *after])
    return "\n\n".join(sections)


def make_distractor_block(index: int) -> str:
    return DISTRACTOR_TEMPLATE.format(
        index=index,
        team=DISTRACTOR_TEAMS[index % len(DISTRACTOR_TEAMS)],
        quarter=(index % 4) + 1,
        status=DISTRACTOR_STATUSES[index % len(DISTRACTOR_STATUSES)],
        region=DISTRACTOR_REGIONS[index % len(DISTRACTOR_REGIONS)],
        priority=DISTRACTOR_PRIORITIES[index % len(DISTRACTOR_PRIORITIES)],
        retention=DISTRACTOR_RETENTION[index % len(DISTRACTOR_RETENTION)],
        checksum=10_000 + index,
    )


def count_tokens(tokenizer: LoadedTokenizer, text: str) -> int:
    if tokenizer.kind == "tiktoken":
        return len(tokenizer.handle.encode(text))
    return len(tokenizer.handle.encode(text, add_special_tokens=False))


def load_context_text(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text()


class OpenAIServerRunner:
    def __init__(self, target: TargetConfig, tokenizer: Any) -> None:
        if not target.base_url:
            raise BenchmarkError(f"Target {target.label} is missing base_url.")
        self.target = target
        self.tokenizer = tokenizer
        self.process: subprocess.Popen[str] | None = None
        self.log_path: Path | None = None
        self.log_handle: Any | None = None

    def __enter__(self) -> "OpenAIServerRunner":
        if self.target.start_command:
            command = self.target.start_command.format(model_path=resolve_model_path(self.target))
            log_file = tempfile.NamedTemporaryFile(
                mode="w+",
                prefix=f"{self.target.label}-",
                suffix=".log",
                delete=False,
            )
            self.log_path = Path(log_file.name)
            self.log_handle = log_file
            self.process = subprocess.Popen(
                shlex.split(command),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
        self.wait_until_ready()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if not self.process:
            if self.log_handle:
                self.log_handle.close()
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=15)
        if self.log_handle:
            self.log_handle.close()

    def wait_until_ready(self) -> None:
        health_url = self.target.health_url or self.target.base_url.rstrip("/") + "/v1/models"
        deadline = time.perf_counter() + self.target.startup_timeout_seconds
        while time.perf_counter() < deadline:
            if self.process and self.process.poll() is not None:
                raise BenchmarkError(
                    f"{self.target.label} exited during startup.\n{self.read_startup_log_tail()}"
                )
            try:
                with urllib.request.urlopen(health_url, timeout=5):
                    return
            except urllib.error.URLError:
                time.sleep(1)
        details = self.read_startup_log_tail()
        suffix = f"\n{details}" if details else ""
        raise BenchmarkError(f"Timed out waiting for {self.target.label} at {health_url}{suffix}")

    def read_startup_log_tail(self) -> str:
        if not self.log_path or not self.log_path.exists():
            return ""
        log_text = self.log_path.read_text(errors="replace")
        tail_lines = log_text.splitlines()[-40:]
        if not tail_lines:
            return ""
        return "Last startup log lines:\n" + "\n".join(tail_lines)

    def generate(self, prompt: str, max_output_tokens: int, temperature: float) -> dict[str, Any]:
        url = self.target.base_url.rstrip("/") + self.target.endpoint
        payload = {
            "model": self.target.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_output_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        request = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        generated_chunks: list[str] = []
        usage: dict[str, Any] | None = None
        start_time = time.perf_counter()
        first_token_at: float | None = None

        with urllib.request.urlopen(request, timeout=self.target.request_timeout_seconds) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break

                event = json.loads(data)
                if event.get("usage"):
                    usage = event["usage"]

                choices = event.get("choices") or []
                if not choices:
                    continue

                delta = choices[0].get("delta") or {}
                content = normalize_stream_content(delta.get("content", ""))
                if content:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    generated_chunks.append(content)

        end_time = time.perf_counter()
        generated_text = "".join(generated_chunks)
        output_tokens = (
            int(usage["completion_tokens"])
            if usage and usage.get("completion_tokens") is not None
            else count_tokens(self.tokenizer, generated_text)
        )
        generation_seconds = max(0.0, end_time - (first_token_at or end_time))
        ttft_seconds = None if first_token_at is None else first_token_at - start_time
        output_tps = None
        if output_tokens > 0 and generation_seconds > 0:
            output_tps = output_tokens / generation_seconds

        return {
            "generated_text": generated_text,
            "output_tokens": output_tokens,
            "prompt_tokens": usage.get("prompt_tokens") if usage else None,
            "ttft_seconds": ttft_seconds,
            "generation_seconds": generation_seconds,
            "output_tps": output_tps,
            "total_seconds": end_time - start_time,
        }


class MLXLMRunner:
    def __init__(self, target: TargetConfig) -> None:
        self.target = target
        self.process: subprocess.Popen[str] | None = None

    def __enter__(self) -> "MLXLMRunner":
        python_bin = self.target.python_bin or detect_mlx_python()
        src_dir = str(Path(__file__).resolve().parents[1])
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = src_dir if not existing_pythonpath else f"{src_dir}{os.pathsep}{existing_pythonpath}"

        command = [python_bin, "-m", "vllm_explore.mlx_lm_worker", "--model", self.target.model]
        if self.target.trust_remote_code:
            command.append("--trust-remote-code")

        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )

        ready_line = self._read_stdout_line(timeout=self.target.startup_timeout_seconds)
        ready_event = json.loads(ready_line)
        if ready_event.get("event") != "ready":
            raise BenchmarkError(f"mlx-lm worker failed to start: {ready_event}")
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if not self.process:
            return
        if self.process.stdin:
            try:
                self.process.stdin.write(json.dumps({"command": "shutdown"}) + "\n")
                self.process.stdin.flush()
            except BrokenPipeError:
                pass
            finally:
                self.process.stdin.close()
        try:
            self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=15)

    def generate(self, prompt: str, max_output_tokens: int, temperature: float) -> dict[str, Any]:
        if not self.process or not self.process.stdin:
            raise BenchmarkError("mlx-lm worker is not running.")

        payload = {
            "prompt": prompt,
            "max_output_tokens": max_output_tokens,
            "temperature": temperature,
        }
        self.process.stdin.write(json.dumps(payload) + "\n")
        self.process.stdin.flush()

        line = self._read_stdout_line(timeout=self.target.request_timeout_seconds)
        event = json.loads(line)
        if event.get("event") != "result":
            raise BenchmarkError(f"mlx-lm worker returned an unexpected payload: {event}")
        return {
            "generated_text": event["generated_text"],
            "output_tokens": event["output_tokens"],
            "prompt_tokens": event.get("prompt_tokens"),
            "ttft_seconds": event.get("ttft_seconds"),
            "generation_seconds": event["generation_seconds"],
            "output_tps": event.get("output_tps"),
            "total_seconds": event["total_seconds"],
        }

    def _read_stdout_line(self, timeout: float) -> str:
        assert self.process is not None
        deadline = time.perf_counter() + timeout
        assert self.process.stdout is not None
        while time.perf_counter() < deadline:
            if self.process.poll() is not None:
                stderr = ""
                if self.process.stderr:
                    stderr = self.process.stderr.read()
                raise BenchmarkError(f"mlx-lm worker exited early. {stderr.strip()}")

            remaining = max(0.0, deadline - time.perf_counter())
            readable, _, _ = select.select([self.process.stdout], [], [], remaining)
            if not readable:
                continue

            line = self.process.stdout.readline()
            if line:
                return line
        raise BenchmarkError("Timed out waiting for mlx-lm worker output.")


class OllamaRunner:
    def __init__(self, target: TargetConfig, tokenizer: Any) -> None:
        if not target.base_url:
            raise BenchmarkError(f"Target {target.label} is missing base_url.")
        self.target = target
        self.tokenizer = tokenizer

    def __enter__(self) -> "OllamaRunner":
        self.wait_until_ready()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.unload_model()
        return None

    def wait_until_ready(self) -> None:
        health_url = self.target.health_url or self.target.base_url.rstrip("/") + "/api/tags"
        deadline = time.perf_counter() + self.target.startup_timeout_seconds
        while time.perf_counter() < deadline:
            try:
                with urllib.request.urlopen(health_url, timeout=5):
                    return
            except urllib.error.URLError:
                time.sleep(1)
        raise BenchmarkError(f"Timed out waiting for {self.target.label} at {health_url}")

    def generate(self, prompt: str, max_output_tokens: int, temperature: float) -> dict[str, Any]:
        url = self.target.base_url.rstrip("/") + (self.target.endpoint or "/api/chat")
        payload = {
            "model": self.target.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_output_tokens,
            },
        }
        if self.target.keep_alive:
            payload["keep_alive"] = self.target.keep_alive

        request = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        generated_chunks: list[str] = []
        final_event: dict[str, Any] | None = None
        start_time = time.perf_counter()
        first_token_at: float | None = None

        with urllib.request.urlopen(request, timeout=self.target.request_timeout_seconds) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                event = json.loads(line)
                message = event.get("message") or {}
                chunk_parts = []
                if message.get("thinking"):
                    chunk_parts.append(message["thinking"])
                if message.get("content"):
                    chunk_parts.append(message["content"])
                chunk_text = "".join(chunk_parts)
                if chunk_text:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    generated_chunks.append(chunk_text)
                if event.get("done"):
                    final_event = event
                    break

        end_time = time.perf_counter()
        generated_text = "".join(generated_chunks)
        output_tokens = (
            int(final_event["eval_count"])
            if final_event and final_event.get("eval_count") is not None
            else count_tokens(self.tokenizer, generated_text)
        )
        generation_seconds = max(0.0, end_time - (first_token_at or end_time))
        ttft_seconds = None if first_token_at is None else first_token_at - start_time
        output_tps = None
        if output_tokens > 0 and generation_seconds > 0:
            output_tps = output_tokens / generation_seconds

        return {
            "generated_text": generated_text,
            "output_tokens": output_tokens,
            "prompt_tokens": final_event.get("prompt_eval_count") if final_event else None,
            "ttft_seconds": ttft_seconds,
            "generation_seconds": generation_seconds,
            "output_tps": output_tps,
            "total_seconds": end_time - start_time,
        }

    def unload_model(self) -> None:
        url = self.target.base_url.rstrip("/") + "/api/generate"
        payload = {
            "model": self.target.model,
            "keep_alive": 0,
        }
        request = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30):
                return
        except urllib.error.URLError as exc:
            raise BenchmarkError(f"Failed to unload Ollama model {self.target.model}: {exc}") from exc


class OpencodeRunner:
    def __init__(self, target: TargetConfig) -> None:
        if not target.base_url:
            raise BenchmarkError(f"Target {target.label} is missing base_url.")
        self.target = target
        self.process: subprocess.Popen[str] | None = None
        self.log_path: Path | None = None
        self.log_handle: Any | None = None

    def __enter__(self) -> "OpencodeRunner":
        if self.target.start_command:
            log_file = tempfile.NamedTemporaryFile(
                mode="w+",
                prefix=f"{self.target.label}-",
                suffix=".log",
                delete=False,
            )
            self.log_path = Path(log_file.name)
            self.log_handle = log_file
            self.process = subprocess.Popen(
                shlex.split(self.target.start_command),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
        self.wait_until_ready()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=15)
        if self.log_handle:
            self.log_handle.close()

    def wait_until_ready(self) -> None:
        health_url = self.target.health_url or self.target.base_url.rstrip("/") + "/global/health"
        deadline = time.perf_counter() + self.target.startup_timeout_seconds
        while time.perf_counter() < deadline:
            if self.process and self.process.poll() is not None:
                raise BenchmarkError(
                    f"{self.target.label} exited during startup.\n{self.read_startup_log_tail()}"
                )
            try:
                with urllib.request.urlopen(health_url, timeout=5):
                    return
            except urllib.error.URLError:
                time.sleep(1)
        details = self.read_startup_log_tail()
        suffix = f"\n{details}" if details else ""
        raise BenchmarkError(f"Timed out waiting for {self.target.label} at {health_url}{suffix}")

    def generate(self, prompt: str, max_output_tokens: int, temperature: float) -> dict[str, Any]:
        command = [
            "opencode",
            "run",
            "--attach",
            self.target.base_url,
            "--format",
            "json",
            "-m",
            self.target.model,
            prompt,
        ]
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        generated_chunks: list[str] = []
        prompt_tokens: int | None = None
        output_tokens: int | None = None
        step_start_ms: int | None = None
        first_text_ms: int | None = None
        last_text_ms: int | None = None
        step_finish_ms: int | None = None

        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.strip()
            if not line:
                continue
            event = json.loads(line)
            if event.get("type") == "step_start":
                step_start_ms = event.get("timestamp")
                continue
            if event.get("type") == "text":
                part = event.get("part") or {}
                if first_text_ms is None:
                    first_text_ms = (part.get("time") or {}).get("start") or event.get("timestamp")
                last_text_ms = (part.get("time") or {}).get("end") or event.get("timestamp")
                generated_chunks.append(part.get("text", ""))
                continue
            if event.get("type") == "step_finish":
                step_finish_ms = event.get("timestamp")
                tokens = (event.get("part") or {}).get("tokens") or {}
                prompt_tokens = tokens.get("input")
                output_tokens = tokens.get("output")

        return_code = process.wait(timeout=self.target.request_timeout_seconds)
        stderr = process.stderr.read().strip() if process.stderr else ""
        if return_code != 0:
            raise BenchmarkError(f"OpenCode run failed for {self.target.label}: {stderr}")
        if output_tokens is None:
            raise BenchmarkError(f"OpenCode did not return token usage for {self.target.label}.")

        ttft_seconds = None
        if step_start_ms is not None and first_text_ms is not None:
            ttft_seconds = max(0.0, (first_text_ms - step_start_ms) / 1000.0)
        generation_seconds = 0.0
        if first_text_ms is not None and last_text_ms is not None:
            generation_seconds = max(0.0, (last_text_ms - first_text_ms) / 1000.0)
        total_seconds = 0.0
        if step_start_ms is not None and step_finish_ms is not None:
            total_seconds = max(0.0, (step_finish_ms - step_start_ms) / 1000.0)
        output_tps = None
        if output_tokens > 0 and generation_seconds > 0:
            output_tps = output_tokens / generation_seconds

        return {
            "generated_text": "".join(generated_chunks),
            "output_tokens": output_tokens,
            "prompt_tokens": prompt_tokens,
            "ttft_seconds": ttft_seconds,
            "generation_seconds": generation_seconds,
            "output_tps": output_tps,
            "total_seconds": total_seconds,
        }

    def read_startup_log_tail(self) -> str:
        if not self.log_path or not self.log_path.exists():
            return ""
        log_text = self.log_path.read_text(errors="replace")
        tail_lines = log_text.splitlines()[-40:]
        if not tail_lines:
            return ""
        return "Last startup log lines:\n" + "\n".join(tail_lines)


def detect_mlx_python() -> str:
    cli_path = shutil.which("mlx_lm.generate")
    if not cli_path:
        raise BenchmarkError(
            "Could not locate `mlx_lm.generate`. Set `python_bin` in the benchmark config."
        )
    first_line = Path(cli_path).read_text().splitlines()[0]
    if not first_line.startswith("#!"):
        raise BenchmarkError(f"Could not determine python for mlx-lm from {cli_path}")
    return first_line[2:].strip()


def resolve_model_path(target: TargetConfig) -> str:
    if target.ollama_model:
        return str(resolve_ollama_blob_path(target.ollama_model))
    return target.model


def resolve_ollama_blob_path(model_ref: str) -> Path:
    namespace_and_name, _, tag = model_ref.partition(":")
    tag = tag or "latest"
    if "/" in namespace_and_name:
        namespace, name = namespace_and_name.split("/", maxsplit=1)
    else:
        namespace, name = "library", namespace_and_name

    manifest_path = (
        Path.home()
        / ".ollama"
        / "models"
        / "manifests"
        / "registry.ollama.ai"
        / namespace
        / name
        / tag
    )
    if not manifest_path.exists():
        raise BenchmarkError(
            f"Could not find Ollama manifest for {model_ref} at {manifest_path}."
        )

    manifest = json.loads(manifest_path.read_text())
    for layer in manifest.get("layers", []):
        if layer.get("mediaType") != "application/vnd.ollama.image.model":
            continue
        digest = layer["digest"].replace(":", "-")
        blob_path = Path.home() / ".ollama" / "models" / "blobs" / digest
        if blob_path.exists():
            return blob_path
        raise BenchmarkError(f"Resolved Ollama blob for {model_ref} is missing: {blob_path}")

    raise BenchmarkError(f"No model layer found in Ollama manifest for {model_ref}")


def make_runner(target: TargetConfig, tokenizer: Any) -> Any:
    if target.kind == "openai_server":
        return OpenAIServerRunner(target, tokenizer)
    if target.kind == "mlx_lm_worker":
        return MLXLMRunner(target)
    if target.kind == "ollama_api":
        return OllamaRunner(target, tokenizer)
    if target.kind == "opencode_cli":
        return OpencodeRunner(target)
    raise BenchmarkError(f"Unsupported target kind: {target.kind}")


def normalize_stream_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "".join(parts)
    return ""


def print_result(row: ResultRow) -> None:
    ttft = "n/a" if row.ttft_seconds is None else f"{row.ttft_seconds:.3f}s"
    tps = "n/a" if row.output_tps is None else f"{row.output_tps:.2f} tok/s"
    print(
        f"  {row.label} | {row.task} | input={row.input_tokens} | output={row.output_tokens} | "
        f"TTFT={ttft} | TPS={tps}"
    )


def default_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("results") / "benchmarks" / timestamp


def write_results(output_dir: Path, results: list[ResultRow], csv_path: Path | None = None) -> None:
    json_path = output_dir / "results.json"
    default_csv_path = output_dir / "results.csv"

    json_payload = [result.__dict__ for result in results]
    json_path.write_text(json.dumps(json_payload, indent=2))

    write_csv(default_csv_path, results)
    if csv_path:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_csv(csv_path, results)


def write_csv(csv_path: Path, results: list[ResultRow]) -> None:
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ResultRow.__dataclass_fields__.keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(result.__dict__)


def main() -> int:
    try:
        return run_cli(sys.argv[1:])
    except BenchmarkError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
