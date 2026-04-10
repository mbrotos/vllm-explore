# AGENT.md

## Purpose

This file is guidance for human contributors and coding agents working in this repository.

The repo is for small, reproducible experiments around local-first LLM workflows and runtime benchmarking. Today the practical focus is `MLX-LM`, `Ollama`, and benchmark harnesses on Apple Silicon. `vLLM` remains a future direction rather than the current default.

## Public Repo Rules

- Do not commit secrets, tokens, `.env` files, or machine-specific credentials.
- Do not commit model weights, caches, or large transient artifacts.
- Do not add private, proprietary, or unclear-license datasets.
- Prefer relative paths in docs, scripts, and examples.
- Treat anything committed here as public.

## Working Style

- Prefer small, runnable experiments over broad abstractions.
- Keep behavior explicit and configuration-driven.
- Favor reproducible benchmark scripts over one-off notebooks.
- Keep the local-first bias unless there is a clear reason to add hosted dependencies.
- Document benchmark caveats when results are shared.

## Benchmark Guidance

- Keep TTFT and TPS definitions consistent across runtimes.
- Separate inference timing from cold-start/model-load timing when possible.
- Note tokenizer differences when comparing prompt length across runtimes.
- Keep output constraints explicit so throughput comparisons stay interpretable.
- If publishing results, include machine specs, config path, and any major caveats.

## Current Priorities

- Improve repeatable benchmarking across local runtimes.
- Compare the same model family across different inference stacks.
- Keep the codebase easy to run and easy to inspect.
- Add only enough agentic workflow code to support concrete experiments.

## Runtime Notes

- `MLX-LM` is the best fit for Apple Silicon-native local inference.
- `Ollama` is useful for fast local setup and API ergonomics.
- `llama.cpp` and GGUF-style workflows are relevant comparison points.
- `vLLM` is still in scope, but not yet the main implemented path in this repo.

## References

- `vLLM`: https://docs.vllm.ai/
- `Ollama`: https://docs.ollama.com/
- `MLX-LM`: https://github.com/ml-explore/mlx-lm
- `llama.cpp`: https://github.com/ggml-org/llama.cpp
- `Gemma 4`: https://huggingface.co/collections/google/gemma-4
- `MLX Gemma 4`: https://huggingface.co/collections/mlx-community/gemma-4
