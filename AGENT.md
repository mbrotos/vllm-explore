# AGENT.md

## Purpose

This repository is a sandbox for exploring local LLM capabilities with an emphasis on `vLLM` and custom agentic workflows.

The likely directions are:

- a privacy-respecting local email agent
- a local research assistant
- repeatable benchmarking across `vLLM`, `MLX-LM`, and `llama.cpp`

## Working Style

- Prefer small, runnable experiments over framework-heavy abstractions.
- Keep everything local-first by default.
- Avoid introducing hosted dependencies unless there is a clear need.
- Favor reproducible benchmark scripts and explicit configuration.
- Do not commit model weights, caches, datasets with sensitive content, or local secrets.

## Near-Term Priorities

- Stand up a minimal Python app and project structure.
- Add simple wrappers or scripts for trying local runtimes.
- Establish a benchmark shape for latency, throughput, memory use, and output quality notes.
- Compare how the same model family behaves across runtimes where possible.

## Runtime Notes

- `vLLM` is the primary serving/runtime focus for high-throughput inference and OpenAI-compatible serving.
- `Ollama` is useful for fast local model setup and API ergonomics.
- `MLX-LM` is relevant for Apple Silicon-native local inference.
- `llama.cpp` is relevant for broad local deployment and GGUF-based workflows.

## Candidate Models

- Google Gemma 4 family on Hugging Face.
- MLX community Gemma 4 ports and quantizations for Apple Silicon testing.

## References

- `vLLM`: https://docs.vllm.ai/
- `Ollama`: https://docs.ollama.com/
- `MLX-LM`: https://github.com/ml-explore/mlx-lm
- `llama.cpp`: https://github.com/ggml-org/llama.cpp
- `Gemma 4`: https://huggingface.co/collections/google/gemma-4
- `MLX Gemma 4`: https://huggingface.co/collections/mlx-community/gemma-4
