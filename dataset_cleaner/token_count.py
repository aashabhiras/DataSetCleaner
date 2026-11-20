"""Streaming JSONL token count utility for estimating training corpus size."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def count_jsonl_tokens(path: Path | str, encoding_name: str = "cl100k_base", fields: list[str] | None = None) -> dict[str, Any]:
    try:
        import tiktoken
    except ImportError as exc:
        raise RuntimeError("Token counting requires tiktoken. Install with: pip install '.[tokens]'") from exc
    encoding = tiktoken.get_encoding(encoding_name)
    path = Path(path)
    field_counts: Counter[str] = Counter()
    records = malformed = serialized_tokens = 0
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            records += 1
            serialized_tokens += len(encoding.encode_ordinary(line))
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if isinstance(row, dict):
                values = row.items()
            else:
                values = (("content", row),)
            for key, value in values:
                if fields and key not in fields:
                    continue
                text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                field_counts[key] += len(encoding.encode_ordinary(text))
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "tokenizer": encoding_name,
        "records": records,
        "malformed_records": malformed,
        "serialized_jsonl_tokens": serialized_tokens,
        "field_value_tokens": dict(sorted(field_counts.items())),
        "field_value_tokens_total": sum(field_counts.values()),
        "fields_counted": fields or "all fields",
        "note": "Serialized JSONL is counted per nonblank line. Field values are counted separately; a model's training template adds its own tokens.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Count JSONL token volume locally using a named tiktoken encoding.")
    parser.add_argument("input", type=Path, help="JSONL file to count.")
    parser.add_argument("--encoding", default="cl100k_base", help="tiktoken encoding (default: cl100k_base).")
    parser.add_argument("--fields", nargs="+", help="Count only these fields for the field-value totals; serialized JSONL count always includes the whole line.")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(count_jsonl_tokens(args.input, args.encoding, args.fields), indent=2, ensure_ascii=False))
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
