# Enterprise Dataset Quality Audit Report (Option A: Pure SFT & Reasoning CoT)

**Dataset**: `the-stack-python-shard0-curated-complete.jsonl`  
**Language**: `PYTHON`  
**Audit Timestamp**: `2026-11-16T17:39:40.471247+00:00`  
**Dataset SHA-256 Digest**: `b9184de14e1cebd9dfacc47602d173cf97b4a2ea375ad4cd73bb17e47be861cb`  
**Total Curated Records**: `12,988`  

---

## Executive Quality Metrics (Dataset-Only Benchmarks)

| Benchmark Metric | Result | Target Benchmark | Status |
| :--- | :--- | :--- | :--- |
| **Parse Rate** | **99.94%** (12,980 / 12,988) | >= 98.0% | **PASS** |
| **Compile Rate** | **99.87%** (12,971 / 12,988) | >= 90.0% | **PASS** |
| **Instruction-to-Code Alignment** | **94.72%** (12,302 / 12,988) | >= 85.0% | **PASS** |
| **Reasoning CoT Explanation Presence** | **100.0%** (12,988 / 12,988) | >= 95.0% | **PASS** |
| **Deduplication & Exact Uniqueness** | **100.0%** (12,988 / 12,988) | >= 99.0% | **PASS** |
| **SPDX License Compliance** | **100.0%** (12,988 / 12,988) | >= 90.0% | **PASS** |
| **Security Safety Rate** | **99.21%** (12,885 / 12,988) | >= 95.0% | **PASS** |
| **Human Review Rate** | **0.85%** | <= 5.0% | **PASS** |

---

## Commercial Licensing Breakdown (>98.2% Permissive)
- **MIT**: 7,556 records (58.2%)
- **Apache-2.0**: 3,319 records (25.6%)
- **BSD (2/3-Clause)**: 1,562 records (12.0%)
- **Unlicense / CC0 / Permissive**: 551 records (4.2%)
- **Copyleft / GPL Contamination**: **0.0%**

---

## Corpus & Token Distribution Statistics
- **Total Verified Pairs**: `12,988`
- **Exact Duplicates Detected**: `0`
- **Security Flagged Examples**: `103`
- **Average Prompt Tokens**: `~49.6` (Max: `266`)
- **Average Completion Tokens**: `~133.7` (Max: `983`)

*Verified and Certified by DataSetCleaner Multi-Channel Deduplication & Quality Engine.*
