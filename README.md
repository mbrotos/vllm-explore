# vllm-explore

Local-first experiments for agentic workflows and LLM runtime benchmarking.

## Why This Repo Exists

This project is a place to explore what local models can do in practical workflows without immediately committing to one product shape.

Current directions include:

- privacy-respecting local email agents
- local research assistants
- benchmark comparisons across `vLLM`, `MLX-LM`, and `llama.cpp`

For now, the repository is intentionally minimal: a `uv`-managed Python project, a simple runnable app, and a lightweight set of notes to guide future experiments.

## Focus Areas

- `vLLM` for fast inference and OpenAI-compatible serving
- `Ollama` for easy local model management
- `MLX-LM` for Apple Silicon-native experiments
- `llama.cpp` for GGUF-based local inference and benchmarking

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

## Current Layout

```text
.
├── AGENT.md
├── README.md
├── pyproject.toml
├── uv.lock
└── src/
    └── vllm_explore/
        ├── __init__.py
        └── __main__.py
```

## Early Experiment Ideas

1. Serve a local model with `vLLM` and drive it through a small agent loop.
2. Prototype a privacy-first email triage assistant against local test data.
3. Build a benchmark harness for latency, throughput, memory use, and qualitative output notes.
4. Compare Gemma-family models across `vLLM`, `MLX-LM`, and `llama.cpp` where compatible formats exist.

## References

- `vLLM` docs: https://docs.vllm.ai/
- `Ollama` docs: https://docs.ollama.com/
- `MLX-LM`: https://github.com/ml-explore/mlx-lm
- `llama.cpp`: https://github.com/ggml-org/llama.cpp
- Google `Gemma 4`: https://huggingface.co/collections/google/gemma-4
- `mlx-community` Gemma 4 ports: https://huggingface.co/collections/mlx-community/gemma-4
