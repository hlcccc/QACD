# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.0.0] — 2026-09-24

First standalone release. Consolidates the QACD method that had been developed
across the research repository and eighteen `qacd_*` result directories into a
single integration-ready project.

### Added

- **Decision layer** (`qacd/`), NumPy-only, CPU-testable:
  - `decompose.py` — question-conditioned atomic claim decomposition with a
    checked-LLM path and a deterministic rule fallback.
  - `features.py` — ordered feature assembly bound to a versioned schema.
  - `align.py` — evidence-claim alignment features.
  - `mechanical.py` — 19-feature mechanical OCR channel and the 7-feature MVR
    multi-view resampling channel.
  - `calibrate.py` — `BICLiteCalibrator` (λ=0.05) and `RidgeLogistic`, both solved
    by damped Newton iterations with no scikit-learn dependency.
  - `aggregate.py` — max-operator claim-to-response projection.
  - `conformal.py` — split-conformal p-values with BH and BY multiplicity control,
    carrying an explicit `validated = False` flag.
  - `pipeline.py` — end-to-end orchestration and pickle-free JSON scorer export.
  - `providers.py` — `EvidenceProvider` protocol plus `MockProvider`,
    `RapidOCRProvider` and a `LLaVAProvider` adapter.
  - `service.py` — FastAPI service exposing the four integration endpoints.
  - `cli.py` — `score`, `demo`, `fit`, `serve`, `version`.
- **Tests** — 176 tests covering decomposition, mechanical/MVR features,
  calibration correctness and determinism, aggregation, conformal selection,
  pipeline fitting, scorer round-trip and the response payload contract.
- **Documentation** (Chinese, with an English README) — method, interface
  contract, deployment and resource requirements, evidence grading, platform
  integration guide and the 课题一 technical integration table.
- **Results** — frozen experimental numbers with provenance back to the research
  artifacts on the remote server.

See `docs/` for the method write-up, the interface contract and the platform
integration guide.
