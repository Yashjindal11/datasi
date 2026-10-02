# Contributing to DataSI

Thanks for helping people find what is wrong with their data before their models do.

## Ground rules

- **Scientific honesty.** Never present an interpretation as an observation. Severity
  must come from an explicit, documented threshold. Phrase explanations as
  possibilities and give heuristics modest confidence.
- **No invented numbers.** Benchmark and research figures are produced by scripts in
  `benchmarks/` and committed with their raw results. Never edit them by hand.
- **No LLMs, no network.** DataSI's intelligence comes from statistics and algorithms.
  Contributions must not add model APIs, telemetry or any outbound network call.
- **Privacy.** Never write raw rows into reports by default; use row positions,
  aggregates or character-class shapes as evidence.

## Development setup

```bash
git clone https://github.com/Yashjindal11/datasi && cd datasi
python3.12 -m venv .venv                # 3.11+
.venv/bin/pip install -e ".[dev]"
scripts/check.sh                        # format check, lint, mypy --strict, tests
.venv/bin/python scripts/run_examples.py
```

Web UI: `cd web && npm ci && npm run dev` (proxies `/api` to `datasi serve`). See
[docs/web.md](docs/web.md).

## Adding a detector

1. Subclass `BaseDetector` in `src/datasi/detectors/`, set `name`, `category`,
   `description`, `assumptions` and `limitations`, decorate with `@register_detector`
   and import it from `detectors/__init__.py` in pipeline order.
2. Add its name to `KNOWN_DETECTORS` in `configuration/config.py`; put any new
   threshold in `Thresholds` with bounds and a description.
3. Every finding needs: a stable `code`, a factual `observation`, JSON-safe `evidence`,
   the `rule` that set its severity, and (where it applies) an `interpretation`,
   `suggestion` and `confidence`.
4. Tests in `tests/detectors/`: a true-positive case **and** a clean case that must stay
   quiet. If the issue can be synthesised, add it to `datasi.synthetic`, map its code in
   `datasi.evaluation.ISSUE_CODES` and include it in the research experiments.
5. Document it in `docs/detectors.md` and `docs/thresholds.md`.

## Statistical changes

Changes to statistics or thresholds must explain what is measured, the assumptions and
the limitations, and should be backed by a test in `tests/statistical/` (known
distribution → known answer) or by rerunning `benchmarks/experiments.py` and committing
the regenerated `docs/research.md`.

## Commits and pull requests

- Conventional Commits (`feat: ...`, `fix(evaluation): ...`, `docs: ...`, `perf: ...`).
- One logical change per commit; keep the project working at every commit.
- Update `CHANGELOG.md` under *Unreleased* when behaviour changes.

## Releases

1. Move *Unreleased* entries in `CHANGELOG.md` under a new version heading.
2. Bump `src/datasi/_version.py`.
3. Commit `chore: release vX.Y.Z`, tag `vX.Y.Z` and push the tag. The release workflow
   tests, builds the web UI into the wheel and publishes a GitHub release with the
   changelog section as notes.
