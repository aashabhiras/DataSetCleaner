"""Thread-safe multi-channel streaming deduplication engine.

Implements:
1. Exact SHA-256 fingerprinting (instruction, completion, combined pair).
2. MinHash / LSH banded near-duplicate detection (Jaccard similarity).
3. Sparse TF-IDF token cosine similarity with the Algorithmic Preservation Rule
   (preserves distinct algorithmic implementations for identical/near-identical prompts).
"""

from __future__ import annotations

import collections
import hashlib
import math
import re
import threading
from typing import Any, Sequence


def _normalize_ws(text: str) -> str:
    """Normalize consecutive whitespace and trim."""
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _tokenize(text: str) -> list[str]:
    """Extract lowercased word/identifier tokens."""
    return [m.group(0).lower() for m in re.finditer(r"[A-Za-z0-9_#$@\-]+", str(text or ""))]


def _ngrams(tokens: Sequence[str], n: int = 3) -> set[str]:
    """Generate n-gram shingles from token sequence."""
    if len(tokens) < n:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


# 128 deterministic seed pairs for MinHash permutations
_PRIME = 4_294_967_311
_MINHASH_NUM_PERM = 128
_SEEDS_A = [((i * 10007 + 3) % (_PRIME - 1)) + 1 for i in range(_MINHASH_NUM_PERM)]
_SEEDS_B = [((i * 49999 + 17) % (_PRIME - 1)) + 1 for i in range(_MINHASH_NUM_PERM)]


def _compute_minhash(shingles: set[str]) -> list[int]:
    """Compute 128-dim MinHash signature vector."""
    if not shingles:
        return [0] * _MINHASH_NUM_PERM
    hashes = [int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16) for s in shingles]
    sig = []
    for a, b in zip(_SEEDS_A, _SEEDS_B):
        min_val = min((a * h + b) % _PRIME for h in hashes)
        sig.append(min_val)
    return sig


def _sparse_tfidf_vector(tokens: Sequence[str]) -> dict[str, float]:
    """Compute L2-normalized term-frequency sparse vector."""
    if not tokens:
        return {}
    counts = collections.Counter(tokens)
    norm = math.sqrt(sum(c * c for c in counts.values()))
    if norm <= 0:
        return {}
    return {k: v / norm for k, v in counts.items()}


def _sparse_cosine_similarity(v1: dict[str, float], v2: dict[str, float]) -> float:
    """Compute cosine similarity between two normalized sparse vectors."""
    if not v1 or not v2:
        return 0.0
    # Iterate over smaller dictionary
    if len(v1) > len(v2):
        v1, v2 = v2, v1
    return sum(val * v2.get(k, 0.0) for k, val in v1.items())


class StreamingDeduplicator:
    """Thread-safe streaming deduplication engine.

    Can run concurrently on the CPU while GPU models perform continuous batch inference.
    """

    def __init__(
        self,
        threshold_near: float = 0.85,
        threshold_semantic: float = 0.92,
        num_perm: int = 128,
        num_bands: int = 16,
    ):
        self.threshold_near = float(threshold_near)
        self.threshold_semantic = float(threshold_semantic)
        self.num_perm = num_perm
        self.num_bands = num_bands
        self.rows_per_band = num_perm // num_bands

        self._lock = threading.RLock()

        # Level 1: Exact SHA-256 sets
        self._exact_prompts: set[str] = set()
        self._exact_completions: set[str] = set()
        self._exact_pairs: set[str] = set()
        self._exact_raw_sources: set[str] = set()

        # Level 2: MinHash LSH buckets: band_idx -> bucket_val -> list[record_id]
        self._lsh_buckets: list[dict[str, list[int]]] = [collections.defaultdict(list) for _ in range(self.num_bands)]
        self._signatures: dict[int, list[int]] = {}

        # Level 3: Semantic sparse index
        self._semantic_instructions: dict[int, dict[str, float]] = {}
        self._semantic_completions: dict[int, dict[str, float]] = {}

        # Stats
        self._next_id = 0
        self._stats = {
            "total_checked": 0,
            "total_indexed": 0,
            "exact_duplicate_pairs": 0,
            "exact_duplicate_completions": 0,
            "near_duplicate_lsh": 0,
            "semantic_duplicates": 0,
            "algorithmic_preservations": 0,
            "raw_sources_checked": 0,
            "raw_sources_duplicates": 0,
        }

    def check_raw_source(self, content: str, source_path: str = "") -> tuple[bool, str]:
        """Check if raw source file is an exact duplicate before sending to GPU.

        Returns (is_duplicate, reason). Atomically records if not duplicate.
        """
        norm = _normalize_ws(content)
        if not norm:
            return False, ""
        sha = hashlib.sha256(norm.encode("utf-8")).hexdigest()
        with self._lock:
            self._stats["raw_sources_checked"] += 1
            if sha in self._exact_raw_sources:
                self._stats["raw_sources_duplicates"] += 1
                return True, f"exact_raw_duplicate:{sha[:16]}"
            self._exact_raw_sources.add(sha)
            return False, ""

    def check_and_add(
        self,
        instruction: str,
        completion: str,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[bool, str, dict[str, Any]]:
        """Atomically checks an instruction/completion pair against all deduplication channels.

        Returns (is_duplicate, reason, details). If not duplicate, indexes it.
        """
        norm_inst = _normalize_ws(instruction)
        norm_comp = _normalize_ws(completion)

        p_hash = hashlib.sha256(norm_inst.encode("utf-8")).hexdigest() if norm_inst else None
        c_hash = hashlib.sha256(norm_comp.encode("utf-8")).hexdigest() if norm_comp else None
        pair_hash = (
            hashlib.sha256(f"{norm_inst}\n---\n{norm_comp}".encode("utf-8")).hexdigest()
            if (norm_inst or norm_comp)
            else None
        )

        with self._lock:
            self._stats["total_checked"] += 1

            # 1. Exact pair match
            if pair_hash and pair_hash in self._exact_pairs:
                self._stats["exact_duplicate_pairs"] += 1
                return True, "exact_pair_duplicate", {"pair_hash": pair_hash[:16]}

            # 2. Exact completion match (identical code generated)
            if c_hash and c_hash in self._exact_completions:
                self._stats["exact_duplicate_completions"] += 1
                return True, "exact_completion_duplicate", {"completion_hash": c_hash[:16]}

            # Compute MinHash & Sparse Vectors
            comp_tokens = _tokenize(completion)
            inst_tokens = _tokenize(instruction)
            comp_shingles = _ngrams(comp_tokens, n=3)
            sig = _compute_minhash(comp_shingles)
            inst_vec = _sparse_tfidf_vector(inst_tokens)
            comp_vec = _sparse_tfidf_vector(comp_tokens)

            # 3. MinHash LSH check for candidate collisions
            candidate_ids: set[int] = set()
            for band_idx in range(self.num_bands):
                start = band_idx * self.rows_per_band
                band_slice = tuple(sig[start : start + self.rows_per_band])
                bucket_key = hashlib.md5(str(band_slice).encode("utf-8")).hexdigest()
                for match_id in self._lsh_buckets[band_idx].get(bucket_key, []):
                    candidate_ids.add(match_id)

            for cand_id in candidate_ids:
                cand_sig = self._signatures.get(cand_id)
                if not cand_sig:
                    continue
                # Exact Jaccard similarity estimate from MinHash
                jaccard_est = sum(s1 == s2 for s1, s2 in zip(sig, cand_sig)) / float(self.num_perm)
                if jaccard_est >= self.threshold_near:
                    # Near duplicate completion
                    self._stats["near_duplicate_lsh"] += 1
                    return True, "minhash_near_duplicate", {
                        "matched_id": cand_id,
                        "jaccard_similarity": round(jaccard_est, 3),
                    }

            # 4. Sparse Cosine Semantic Deduplication with Algorithmic Preservation Rule
            for cand_id, other_inst_vec in self._semantic_instructions.items():
                other_comp_vec = self._semantic_completions.get(cand_id, {})
                sim_inst = _sparse_cosine_similarity(inst_vec, other_inst_vec)
                sim_comp = _sparse_cosine_similarity(comp_vec, other_comp_vec)

                # Algorithmic Preservation Rule:
                # If prompt is identical/high-similarity, but completion is distinct (sim_comp < 0.85),
                # preserve it! (e.g. recursive vs iterative solutions).
                if sim_inst >= self.threshold_semantic and sim_comp < self.threshold_near:
                    self._stats["algorithmic_preservations"] += 1
                    continue

                if sim_inst >= self.threshold_semantic and sim_comp >= self.threshold_semantic:
                    self._stats["semantic_duplicates"] += 1
                    return True, "semantic_duplicate", {
                        "matched_id": cand_id,
                        "sim_instruction": round(sim_inst, 3),
                        "sim_completion": round(sim_comp, 3),
                    }
                elif sim_comp >= self.threshold_semantic:
                    self._stats["semantic_duplicates"] += 1
                    return True, "semantic_code_duplicate", {
                        "matched_id": cand_id,
                        "sim_completion": round(sim_comp, 3),
                    }

            # Passed all deduplication gates -> Index record
            rec_id = self._next_id
            self._next_id += 1

            if pair_hash:
                self._exact_pairs.add(pair_hash)
            if c_hash:
                self._exact_completions.add(c_hash)
            if p_hash:
                self._exact_prompts.add(p_hash)

            self._signatures[rec_id] = sig
            self._semantic_instructions[rec_id] = inst_vec
            self._semantic_completions[rec_id] = comp_vec

            for band_idx in range(self.num_bands):
                start = band_idx * self.rows_per_band
                band_slice = tuple(sig[start : start + self.rows_per_band])
                bucket_key = hashlib.md5(str(band_slice).encode("utf-8")).hexdigest()
                self._lsh_buckets[band_idx][bucket_key].append(rec_id)

            self._stats["total_indexed"] += 1
            return False, "", {"record_id": rec_id}

    def stats(self) -> dict[str, Any]:
        """Return snapshot of deduplication statistics."""
        with self._lock:
            return dict(self._stats)

    def dump_state(self) -> dict[str, Any]:
        """Serialize deduplication state for checkpoint persistence."""
        with self._lock:
            lsh_serializable = [
                {k: list(v) for k, v in self._lsh_buckets[i].items()}
                for i in range(self.num_bands)
            ]
            return {
                "threshold_near": self.threshold_near,
                "threshold_semantic": self.threshold_semantic,
                "num_perm": self.num_perm,
                "num_bands": self.num_bands,
                "exact_prompts": list(self._exact_prompts),
                "exact_completions": list(self._exact_completions),
                "exact_pairs": list(self._exact_pairs),
                "exact_raw_sources": list(self._exact_raw_sources),
                "lsh_buckets": lsh_serializable,
                "signatures": {str(k): v for k, v in self._signatures.items()},
                "semantic_instructions": {str(k): v for k, v in self._semantic_instructions.items()},
                "semantic_completions": {str(k): v for k, v in self._semantic_completions.items()},
                "next_id": self._next_id,
                "stats": dict(self._stats),
            }

    def load_state(self, state: dict[str, Any]) -> None:
        """Restore deduplication state from serialized checkpoint."""
        with self._lock:
            self.threshold_near = float(state.get("threshold_near", self.threshold_near))
            self.threshold_semantic = float(state.get("threshold_semantic", self.threshold_semantic))
            self.num_perm = int(state.get("num_perm", self.num_perm))
            self.num_bands = int(state.get("num_bands", self.num_bands))
            self.rows_per_band = self.num_perm // self.num_bands

            self._exact_prompts = set(state.get("exact_prompts", []))
            self._exact_completions = set(state.get("exact_completions", []))
            self._exact_pairs = set(state.get("exact_pairs", []))
            self._exact_raw_sources = set(state.get("exact_raw_sources", []))

            raw_lsh = state.get("lsh_buckets", [{} for _ in range(self.num_bands)])
            self._lsh_buckets = [
                collections.defaultdict(list, {k: list(v) for k, v in raw_lsh[i].items()})
                if i < len(raw_lsh) else collections.defaultdict(list)
                for i in range(self.num_bands)
            ]
            self._signatures = {int(k): v for k, v in state.get("signatures", {}).items()}
            self._semantic_instructions = {int(k): v for k, v in state.get("semantic_instructions", {}).items()}
            self._semantic_completions = {int(k): v for k, v in state.get("semantic_completions", {}).items()}
            self._next_id = int(state.get("next_id", len(self._signatures)))
            self._stats = dict(state.get("stats", self._stats))
