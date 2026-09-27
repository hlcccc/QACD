# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `calibrated_confidence` on the `POST /v1/qacd/risk` response. It is the
  complement of `risk_score`, implemented as a derived property on
  `ResponseRisk` so the two can never disagree. When the MVR fusion head is
  fitted it is a genuine response-level calibrated probability; otherwise it
  inherits the max operator's conservatism. The integration table had promised
  this field since 1.0.0 while the service returned no such key.
- `tests/test_integration_docx.py`, which pins the integration-table generator
  against the live response contract.

### Changed

- `scripts/make_integration_docx.py` now derives the 出参 column from
  `qacd.types.ResponseRisk` instead of carrying a hardcoded copy, and aborts if a
  listed field does not exist. The hardcoded copy is exactly how the table came
  to advertise a field the service never returned.
- `scripts/make_integration_docx.py` requires `--owner` and refuses to emit a
  document whose 对接负责人 is still a placeholder.
- `docs/05-课题一技术对接表.md` no longer claims to be a verbatim mirror of the
  Word deliverable, which it was not, and no longer publishes contact details in
  the public repository.
- `README.en.md` was brought back in line with `README.md`: it was missing the
  decomposition-version section, the real-hardware verification record and the
  code-completeness table.

### Removed

- Experimental data, frozen feature matrices, raw evidence tables and result
  values are no longer part of the repository. They are project deliverables.
  The 1.0.0 entry below records that they shipped at the time; that is no longer
  true of this repository.

### Fixed

- `README.md` quoted a stale test count in three places, as did `CHANGELOG.md`.

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
- **Tests** — the offline suite covering decomposition, mechanical/MVR features,
  calibration correctness and determinism, aggregation, conformal selection,
  pipeline fitting, scorer round-trip and the response payload contract.
- **Documentation** (Chinese, with an English README) — method, interface
  contract, deployment and resource requirements, evidence grading, platform
  integration guide and the 课题一 technical integration table.
- **Results** — frozen experimental numbers with provenance back to the research
  artifacts on the remote server. *Subsequently withdrawn; see Unreleased.*

See `docs/` for the method write-up, the interface contract and the platform
integration guide.
