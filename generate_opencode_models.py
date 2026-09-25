#!/usr/bin/env python3
"""Generate a full opencode.jsonc from local llama.cpp .gguf files + mcp.json.

Scans a models directory (default: ./models next to this script) for *.gguf files
and combines them with the MCP fragment in ./mcp.json into a single full-formed
opencode.jsonc written into this directory (default: ./opencode.jsonc).

Thinking-capable models (Qwen, DeepSeek, gpt-oss, ...) get `off` / `low` /
`medium` / `high` reasoning-effort variants. Non-thinking models (e.g. Gemma)
get a plain entry with no variants.

Why `reasoningEffort` is the right key (verified, not guessed):
- opencode v1 model/variant `options` are passed through as AI SDK provider
  options; `@ai-sdk/openai-compatible` maps `reasoningEffort` to the
  `reasoning_effort` field of the `/v1/chat/completions` body.
- llama-server (verified against ggml-org/llama.cpp commit f805c57a2, the
  0.5.0-dev build our container runs) parses per-request `reasoning_effort`:
  `"none"` disables thinking, any other value is forwarded into the jinja
  `chat_template_kwargs["reasoning_effort"]`. Per-request
  `chat_template_kwargs: {enable_thinking: bool}` also works.
- Model IDs are the filename stems WITHOUT `.gguf`: that is what this
  llama-server reports on `GET /v1/models` (it sets `--alias <stem>` per file).

Usage:
    python3 generate_opencode_models.py
    python3 generate_opencode_models.py --mcp-file ./mcp.json --full-output ./opencode.jsonc
    python3 generate_opencode_models.py --models-only --output opencode.models.json

- Default run writes ./opencode.jsonc (full config: $schema + provider + mcp).
- --models-only restores the old behaviour (models object only).
- Embedding-only files (name contains 'embed') are skipped unless --include-embeddings.
- mcp.json may be a full JSON object ({"mcp": {...}}) or a bare fragment
  ("mcp": {...}); both are accepted.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

QUANT_SUFFIX_RE = re.compile(
    r"[-_](UD-)?(Q[234568](_K_[SMLX]{1,2}|_0)?|IQ[1234]_X{0,2}[SM]|F16|BF16|QAT)",
    re.IGNORECASE,
)

EFFORTS = ("off", "low", "medium", "high")
EFFORT_VALUE = {"off": "none", "low": "low", "medium": "medium", "high": "high"}

# Models whose jinja template supports thinking/reasoning. Everything else
# (e.g. Gemma) gets a plain entry with no reasoning variants.
THINKING_RE = re.compile(r"qwen|deepseek|gpt-oss|glm|kimi|think", re.IGNORECASE)


def is_thinking_model(stem: str) -> bool:
    return bool(THINKING_RE.search(stem))


def pretty_name(stem: str) -> str:
    """Turn 'Qwen3.8-27B-UD-Q2_K_XL' into 'Qwen3.8 27B Q2_K_XL'."""
    name = QUANT_SUFFIX_RE.split(stem)[0].rstrip("-_")
    name = name.replace("-", " ").replace("_", " ").strip()
    name = re.sub(r"\s+", " ", name)
    return name or stem


def quant_tag(filename: str) -> str:
    m = re.search(
        r"(UD-)?(Q[234568](_K_[SMLX]{1,2}|_0)?|IQ[1234]_X{0,2}[SM]|F16|BF16|QAT[^.]*)\.gguf$",
        filename,
        re.IGNORECASE,
    )
    return m.group(0).removesuffix(".gguf") if m else ""


def load_mcp_fragment(mcp_file: Path) -> dict:
    """Load mcp.json which may be a full object or a bare `"mcp": {...}` fragment."""
    text = mcp_file.read_text().strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "mcp" in data:
            return data["mcp"]
        return data
    except json.JSONDecodeError:
        pass
    # Fall back: wrap bare fragment in braces.
    data = json.loads("{" + text + "}")
    return data.get("mcp", {})


def build_full_config(provider_id: str, provider_name: str, base_url: str,
                       models: dict, mcp: dict) -> dict:
    config: dict = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            provider_id: {
                "npm": "@ai-sdk/openai-compatible",
                "name": provider_name,
                "options": {"baseURL": base_url},
                "models": models,
            }
        },
    }
    if mcp:
        config["mcp"] = mcp
    return config


def build_models(
    models_dir: Path,
    context: int,
    output_tokens: int,
    include_embeddings: bool,
    with_ext: bool = False,
) -> dict:
    models: dict = {}
    for gguf in sorted(models_dir.glob("*.gguf")):
        if not include_embeddings and "embed" in gguf.name.lower():
            continue
        # llama-server --models-dir reports IDs as filename stems (no .gguf).
        model_id = gguf.name if with_ext else gguf.stem
        stem = gguf.stem
        tag = quant_tag(gguf.name)
        display = pretty_name(stem)
        if tag:
            display = f"{display} ({tag})"
        entry: dict = {
            "name": f"{display} (local)",
            "limit": {"context": context, "output": output_tokens},
        }
        if is_thinking_model(stem):
            entry["variants"] = {
                effort: {"reasoningEffort": EFFORT_VALUE[effort]}
                for effort in EFFORTS
            }
        models[model_id] = entry
    return models


def main() -> None:
    here = Path(__file__).parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", default=str(here / "models"))
    ap.add_argument("--mcp-file", default=str(here / "mcp.json"), help="MCP fragment to merge into full config")
    ap.add_argument("--no-mcp", action="store_true", help="skip MCP merge even if mcp.json exists")
    ap.add_argument("--provider-id", default="llama.cpp")
    ap.add_argument("--provider-name", default="llama-server (local)")
    ap.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    ap.add_argument("--context", type=int, default=262144, help="ctx-size from compose.yaml")
    ap.add_argument("--output-tokens", type=int, default=65536)
    ap.add_argument("--full-output", default=str(here / "opencode.jsonc"),
                    help="file to write full opencode config to (in this directory)")
    ap.add_argument("--output", default="opencode.models.json", help="file for --models-only output")
    ap.add_argument("--with-ext", action="store_true",
                    help="use full .gguf filename as model id (default: stem, matching /v1/models)")
    ap.add_argument("--include-embeddings", action="store_true")
    ap.add_argument("--models-only", action="store_true", help="only write/print the models object")
    args = ap.parse_args()

    models_dir = Path(args.models_dir)
    if not models_dir.is_dir():
        raise SystemExit(f"models dir not found: {models_dir}")

    models = build_models(models_dir, args.context, args.output_tokens, args.include_embeddings, args.with_ext)
    if not models:
        raise SystemExit(f"no .gguf files found in {models_dir}")

    if args.models_only:
        Path(args.output).write_text(json.dumps(models, indent=2) + "\n")
        print(f"# wrote {len(models)} models -> {args.output}")
        print(json.dumps(models, indent=2))
        return

    mcp: dict = {}
    mcp_path = Path(args.mcp_file)
    if not args.no_mcp and mcp_path.is_file():
        mcp = load_mcp_fragment(mcp_path)
        print(f"# merged MCP entries from {mcp_path}: {sorted(mcp)}")
    elif not args.no_mcp:
        print(f"# note: {mcp_path} not found, generating without MCP block")

    config = build_full_config(args.provider_id, args.provider_name, args.base_url, models, mcp)
    out = Path(args.full_output)
    out.write_text(json.dumps(config, indent=2) + "\n")
    print(f"# wrote full opencode config ({len(models)} models) -> {out}")
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
