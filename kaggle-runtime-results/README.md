# Kaggle Runtime Results — 100% Free Groq RLVR Dataset

This directory contains the complete runtime outputs, verification datasets, rejection traces, execution logs, and detailed failure taxonomy from the Kaggle CPU execution of **Version 19** of the DataSetCleaner Groq RLVR Builder.

## Key Highlights

- **Dataset**: `bigcode/the-stack-dedup` Python Shard 0 (Curated Base of 12,988 records).
- **Sweep Coverage**: **12,988 / 12,988 records (100.0% Complete)** across Pass 1 & Pass 2.
- **Total True-Verified Records**: **`5,105`** execution-verified Python coding records.
- **Zero False-Positive Guarantee**: Every problem verified in an isolated Python subprocess sandbox with $\ge 1$ assertions (average **5.79 assertions** per problem).
- **Reflection Rescues**: **`1,332 records`** (26.1%) rescued through 1-step in-line traceback reflection.
- **Zero-Cost Preflight Saves**: **`4,732 broken/un-importable candidates`** eliminated locally at $\$0.00$ cost / 0 Groq tokens.
- **Compute Cost**: **Strictly $\$0.00$** (100% Free Kaggle CPU + 15 Native Groq Free Tier API Keys).

## Files in this Directory

- [the-stack-python-shard0-rlvr-true-verified.jsonl](file:///e:/Github/DataSetCleaner/kaggle-runtime-results/the-stack-python-shard0-rlvr-true-verified.jsonl) (19.1 MB, 5,105 verified records — available in [Google Drive Release Folder](https://drive.google.com/drive/folders/1G6bs3m1thpqd77-_Rp5Bfmgef_hyEjzA))
- [the-stack-python-shard0-rlvr-true-verified.jsonl.gz](file:///e:/Github/DataSetCleaner/kaggle-runtime-results/the-stack-python-shard0-rlvr-true-verified.jsonl.gz) (4.0 MB, compressed release — available in [Google Drive Release Folder](https://drive.google.com/drive/folders/1G6bs3m1thpqd77-_Rp5Bfmgef_hyEjzA))
- [the-stack-python-shard0-rlvr-rejections.jsonl](file:///e:/Github/DataSetCleaner/kaggle-runtime-results/the-stack-python-shard0-rlvr-rejections.jsonl) (5.1 MB, rejection traces — available in [Google Drive Release Folder](https://drive.google.com/drive/folders/1G6bs3m1thpqd77-_Rp5Bfmgef_hyEjzA))
- [the-stack-python-shard0-rlvr-rejections.jsonl.gz](file:///e:/Github/DataSetCleaner/kaggle-runtime-results/the-stack-python-shard0-rlvr-rejections.jsonl.gz) (1.2 MB, compressed rejections — available in [Google Drive Release Folder](https://drive.google.com/drive/folders/1G6bs3m1thpqd77-_Rp5Bfmgef_hyEjzA))
- [STATEMENTS_AND_REASONS.md](STATEMENTS_AND_REASONS.md) (Complete verification statements, failure category taxonomy, and traceback case studies)
- [summary_metrics.json](summary_metrics.json) (Machine-readable breakdown of verification and rejection metrics)
- [datasetcleaner-groq-rlvr-builder.log](datasetcleaner-groq-rlvr-builder.log) (Execution log from Kaggle CPU kernel)
- [rlvr_status.json](rlvr_status.json) (Final status telemetry snapshot: `state: completed`)
- [rlvr_cursor.json](rlvr_cursor.json) (Final cursor snapshot: `record_mark: 12988/12988`)

> **Note on Large Datasets**: To prevent repository bloat, raw `.jsonl` and `.gz` archives are hosted in the official [Google Drive Release Folder (DataSetCleaner-Python-Cleaned)](https://drive.google.com/drive/folders/1G6bs3m1thpqd77-_Rp5Bfmgef_hyEjzA) and in the Kaggle Kernel outputs (`aashrayrastogi01/datasetcleaner-groq-rlvr-builder`).
