# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-10-02

First public release.

### Added

- `inspect()` / `investigate()` pipeline over pandas, CSV/TSV (incl. gzip), Parquet,
  JSON/JSONL, Polars and PyArrow inputs, with pluggable source adapters.
- Schema inference with semantic types, storage hints and explained identifier detection.
- 16 detectors: schema, missingness (group, time, co-occurrence and numeric
  association), duplicates, constant, identifier, types, rules, numeric, outliers (IQR,
  z-score, modified z-score, isolation forest), categorical, strings, datetime,
  temporal (CUSUM change points), correlation (Pearson, Spearman, Cramér's V,
  correlation ratio, mutual information), redundancy (identical, linear, sum identities
  and their breaking rows, one-to-one mappings) and target analysis.
- `inspect_train_test()` / `datasi compare`: schema, missingness, KS, Wasserstein, PSI,
  JSD, unseen categories, range checks and a domain classifier.
- Structured findings separating observation, evidence, rule, interpretation,
  suggestion and confidence; configurable, validated thresholds.
- JSON, Markdown and self-contained offline HTML reports; optional matplotlib figures.
- CLI: `inspect`, `report`, `compare`, `schema` (incl. streaming `--chunksize`),
  `config validate|init`, `detectors`, `serve`.
- Local web UI (React, TypeScript, Vite, Tailwind) served by `datasi serve`.
- Synthetic dataset generator with ground truth, detector evaluation
  (precision/recall/FPR/FNR), research experiments E1–E6 and a 10K/100K/1M benchmark.
