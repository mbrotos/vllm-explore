# vllm-explore

Experimental local-first LLM runtime benchmarks and agent workflow prototypes, currently focused on Apple Silicon-friendly setups.

> Warning ⚠️
> This project is heavily vibe-coded and should be treated as directional exploration, not production-quality software or rigorous benchmark infrastructure.

## Status

- Early-stage and intentionally minimal.
- Current benchmark focus: `MLX-LM`, `Ollama`, and an `OpenCode`-backed reference API target.
- Despite the repository name, `vLLM` is currently more of a future direction than an implemented default.

## Why This Repo Exists

This project is a place to explore what local models can do in practical workflows without committing too early to one runtime, serving stack, or product shape.

Current directions include:

- privacy-respecting local email agents
- local research assistants
- benchmark comparisons across `MLX-LM`, `Ollama`, and future `vLLM` paths

For now, the repository is intentionally small: a `uv`-managed Python project, a simple runnable app, a benchmark harness, and lightweight notes to guide future experiments.

## Focus Areas

- `MLX-LM` for Apple Silicon-native experiments
- `Ollama` for easy local model management
- simple local benchmarking harnesses for repeatable runtime comparisons
- future `vLLM` experiments for higher-throughput serving

## Public Repo Hygiene

- Do not commit secrets, auth tokens, `.env` files, or machine-specific credentials.
- Do not commit model weights, local caches, or large transient benchmark artifacts.
- Do not add private or restricted datasets unless their redistribution terms are clear.
- Prefer relative paths and reproducible commands in docs and scripts.
- If you publish benchmark results, include enough context to reproduce them: machine specs, config path, model labels, and major caveats.

## Getting Started

Install dependencies and create the virtual environment:

```bash
uv sync
```

Run the hello-world app:

```bash
uv run vllm-explore
```

Or run the package directly:

```bash
uv run python -m vllm_explore
```

## Benchmark Harness

The repo includes a simple benchmark harness focused on:

- TTFT (time to first token)
- output tokens per second
- a few built-in zero-shot reasoning tasks
- multiple approximate input sizes
- longer-context prompts backed by local Wikipedia-derived reference bundles

The first config targets Gemma 4 across two local runtimes and one OpenCode-backed reference model:

- `MLX-LM`
- `Ollama`
- `OpenCode` with `openai/gpt-5.4`

And three model sizes:

- `31B`
- `26B-A4B`
- `E4B` as the small `4B`-class target

Dry-run the matrix first:

```bash
uv run vllm-explore benchmark run --config benchmarks/gemma4.json --dry-run
```

Run only one target while iterating:

```bash
uv run vllm-explore benchmark run --config benchmarks/gemma4.json --match mlx-gemma4-e4b
```

Results are written to `results/benchmarks/<timestamp>/` as both JSON and CSV.

If you want one stable final CSV path for the full run, use:

```bash
uv run vllm-explore benchmark run --config benchmarks/gemma4.json --csv-path results/final-benchmark-results.csv
```

## Benchmark Notes

- `MLX-LM` is measured through a persistent Python worker so the model loads once and each prompt measures inference rather than repeated cold starts.
- `Ollama` is measured through its native streamed `/api/chat` responses.
- `OpenCode` is measured through streamed `opencode run --attach ... --format json` output against a local `opencode serve` instance.
- `input_tokens` is the safer cross-runtime prompt-length field to compare. `prompt_tokens` can differ materially across tokenizers.
- If the `26B` model is not installed locally, pull it first with `ollama pull gemma4:26b`.
- The built-in input sizes are `512`, `2048`, and `8192` approximate prompt tokens.
- The default Gemma config uses longer-context tasks built from local files in `benchmarks/context/`.
- Those context bundles are condensed from public Wikipedia articles and then padded further when needed to hit the target context sizes.
- The benchmark prompts ask the model to reason privately and return compact JSON so output length stays relatively controlled.
- The `opencode-gpt-5.4` target assumes your local OpenCode install is already authenticated with the `openai` provider.

## Results and Analysis

- Raw benchmark outputs: `results/`
- Example consolidated CSV: `results/final-benchmark-results.csv`
- Example write-up: `report.md`
- Example generated figures: `figs/`

## Current Layout

```text
.
├── AGENT.md
├── benchmarks/
├── figs/
├── README.md
├── report.md
├── pyproject.toml
├── results/
├── uv.lock
└── src/
    └── vllm_explore/
        ├── __init__.py
        ├── __main__.py
        ├── benchmark.py
        └── mlx_lm_worker.py
```

## Contributing

Small, reproducible changes are preferred over large speculative refactors. If you add a benchmark, keep the config explicit, document the runtime assumptions, and include a short note on how to reproduce the result.

## Early Experiment Ideas

1. Serve a local model with `vLLM` and drive it through a small agent loop.
2. Prototype a privacy-first email triage assistant against local test data.
3. Build a benchmark harness for latency, throughput, memory use, and qualitative output notes.
4. Compare Gemma-family models against a stronger OpenCode-accessible reference model.

## References

- `Ollama` docs: https://docs.ollama.com/
- `MLX-LM`: https://github.com/ml-explore/mlx-lm
- Google `Gemma 4`: https://huggingface.co/collections/google/gemma-4
- `mlx-community` Gemma 4 ports: https://huggingface.co/collections/mlx-community/gemma-4
