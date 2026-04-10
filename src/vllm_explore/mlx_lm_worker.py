from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from typing import Any

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler


def build_chat_prompt(tokenizer: Any, prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
    return prompt


def handle_request(model: Any, tokenizer: Any, payload: dict[str, Any]) -> dict[str, Any]:
    prompt = build_chat_prompt(tokenizer, payload["prompt"])
    max_output_tokens = int(payload.get("max_output_tokens", 96))
    temperature = float(payload.get("temperature", 0.0))
    sampler = make_sampler(temp=temperature)

    generated_chunks: list[str] = []
    first_token_at: float | None = None
    last_response: Any = None
    start_time = time.perf_counter()

    for response in stream_generate(
        model,
        tokenizer,
        prompt,
        max_tokens=max_output_tokens,
        sampler=sampler,
    ):
        if first_token_at is None:
            first_token_at = time.perf_counter()
        generated_chunks.append(response.text)
        last_response = response

    end_time = time.perf_counter()
    generated_text = "".join(generated_chunks)
    generation_seconds = max(0.0, end_time - (first_token_at or end_time))
    output_tokens = 0 if last_response is None else int(last_response.generation_tokens)
    output_tps = None
    if output_tokens > 0 and generation_seconds > 0:
        output_tps = output_tokens / generation_seconds

    return {
        "event": "result",
        "generated_text": generated_text,
        "prompt_tokens": None if last_response is None else int(last_response.prompt_tokens),
        "output_tokens": output_tokens,
        "ttft_seconds": None if first_token_at is None else first_token_at - start_time,
        "generation_seconds": generation_seconds,
        "output_tps": output_tps,
        "total_seconds": end_time - start_time,
    }


def cleanup_model(model: Any, tokenizer: Any) -> None:
    del model
    del tokenizer
    gc.collect()
    mx.metal.clear_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    tokenizer_config = {"trust_remote_code": True} if args.trust_remote_code else None
    model, tokenizer = load(args.model, tokenizer_config=tokenizer_config)
    print(json.dumps({"event": "ready", "model": args.model}), flush=True)

    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if payload.get("command") == "shutdown":
                print(json.dumps({"event": "shutdown"}), flush=True)
                break
            result = handle_request(model, tokenizer, payload)
            print(json.dumps(result), flush=True)
    finally:
        cleanup_model(model, tokenizer)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
