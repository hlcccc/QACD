# QACD — Question-Conditioned Atomic Claim Decomposition

**Post-hoc answer-error risk scoring for LVLM visual question answering**

[![CI](https://github.com/hlcccc/QACD/actions/workflows/ci.yml/badge.svg)](https://github.com/hlcccc/QACD/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.9%2B-blue)](pyproject.toml)

QACD is a *post-processing* risk scorer. It does not retrain or modify the model
under evaluation: given a frozen LVLM's answer together with the question and the
image, it returns an answer-level error-risk score in `[0, 1]`, a matching
`calibrated_confidence` (its complement), and per-claim risk detail. It is
intended for early warning, human-review triage and abstention policies.

```python
from qacd import QACDPipeline, QACDConfig

pipeline = QACDPipeline(config=QACDConfig(k=3))    # k > 0 enables the MVR channel
result = pipeline.score(
    question="What brand is the camera?",
    answer="Dakota digital",
    image="demo.jpg",
)
print(result.risk_score, result.calibrated_confidence, result.is_high_risk)
```

## Decomposition versions v1 and v2

**The reported numbers were produced by v1. `qacd/decompose.py` — the repository
default — implements v2.** The two are not merely labelled differently:

| | v1 (produced the results) | v2 (current default) |
|---|---|---|
| Fallback splitting | sentences / clauses only | adds sub-clauses, commas, 22-word chunks, de-duplication |
| Type inference | `question + conditioned claim`, digits first | source span, claim first |
| `is_atomic` threshold | 0.75 | 0.9 |
| Quoted questions | counted in the length penalty | stripped before measuring |
| Claim-set validation | none | coverage / atomicity / duplication checks |

**Both versions ship.** `qacd/decompose.py` (v2, default) and
`qacd/decompose_v1.py` (v1, ported verbatim from the upgrade commit `f03b27a^`).
Every behavioural difference is pinned by `tests/test_decompose_v1.py`.

> If you are reproducing the delivered numbers, use v1. If you are extending the
> method, be aware that v2 has **not** been evaluated on the frozen split.

## Two scoring paths, and which one the numbers came from

This distinction matters more than any other in this repository:

| | Path A — live | Path B — frozen |
|---|---|---|
| Entry point | `QACDPipeline.score(question, answer, image)` | `QACDPipeline.score_evidence_frame(evidence)` |
| Features | `qacd/features.py`, **48 columns** | `frozen/`, **112 columns** |
| HTTP API | `POST /v1/qacd/risk` (reports `feature_dim: 48`) | not served |
| Produces the reported numbers | **no** | **yes** |

Passing a `FrozenFeatureAssembler` to `QACDPipeline(assembler=...)` switches the
*column names* to the frozen 112 but does **not** change where the live path gets
its values, so the two must not be mixed. See `docs/02-接口文档.md` for the full
contract.

## Experimental data

This repository ships **without** experimental data or result values; the frozen
feature matrices, raw evidence tables and result tables are project deliverables
and are not published here. What ships is the method and the runnable code.

To run the verification scripts, export the data from the research host first:

```bash
python tools/export_raw_evidence.py    # raw evidence tables -> artifacts/raw/
python tools/export_bundle.py          # feature matrix + reference scores -> artifacts/
```

| Script | What it verifies |
|---|---|
| `scripts/run_frozen_evaluation.py --from-raw` | rebuild features → calibrate → aggregate → metrics |
| `scripts/verify_feature_port.py` | 112-column features align element-wise with the frozen matrix |
| `scripts/verify_decomposition_v1.py` | v1 decomposition reproduces the frozen claim table |
| `scripts/verify_pipeline_frozen.py` | `QACDPipeline` closes the loop over the frozen feature set |

Without the data those scripts exit with status **2** and name the missing inputs;
they never pass silently on empty input. 15 tests report as `skipped` in the
documented quick-start environment (`pytest -rs` prints why): 14 need the data and
1 checks the integration-table generator, which needs `python-docx`. The other
**174** need nothing.

## Pipeline

| Stage | What it does | Code |
|---|---|---|
| ① | Question-conditioned atomic claim decomposition (checked LLM + deterministic fallback) | `qacd/decompose.py` |
| ② | Belief-consistency views (4 views) | `qacd/features.py` |
| ③ | Type-routed direct verification | `qacd/features.py` |
| ④ | Evidence-claim alignment (**declared: no independent gain**) | `qacd/align.py` |
| ⑤ | Mechanical OCR instrument channel (19 deterministic features) | `qacd/mechanical.py` |
| ⑥ | BIC-lite risk calibration (L2 logistic, λ=0.05, NumPy only) | `qacd/calibrate.py` |
| ⑦ | Max-operator claim → response aggregation | `qacd/aggregate.py` |
| ⑧ | MVR multi-view resampling consistency (K=3 is the knee) | `qacd/mechanical.py` |

## Quick start

The decision layer depends on **NumPy and pandas only** — no GPU, no model weights.

```bash
git clone https://github.com/hlcccc/QACD.git && cd QACD

# package + the `qacd` console script + test dependencies
pip install -e ".[test]"
```

> Installing only the runtime dependencies (`pip install -r requirements.txt`) does
> **not** create the `qacd` command and does **not** install pytest. That route is
> for embedding QACD in your own code; use `python -m qacd.cli` in place of `qacd`.

```bash
python -m pytest -q                  # 193 passed + 15 skipped (14 need data, 1 needs python-docx)
python scripts/demo_offline.py       # end-to-end demo on a synthetic dev set
qacd demo
```

That step needs no model and no GPU; it verifies the install and the plumbing.
To get a score that **means** anything you have to attach a real evidence
provider:

```bash
# 1. model dependencies + weights (~26 GB, not distributed here)
pip install -e ".[llava,ocr]"
huggingface-cli download llava-hf/llava-1.5-13b-hf --local-dir /models/llava-1.5-13b-hf

# 2. fit a scorer on your own development set
# dev.jsonl: {"question":..., "answer":..., "image":..., "failed":0|1, "group":...}
qacd fit --data dev.jsonl --out scorer.json --k 3 \
         --provider llava --model-path /models/llava-1.5-13b-hf

# 3. serve the four integration endpoints
pip install -e ".[service]"
qacd serve --scorer scorer.json --port 8080 \
           --provider llava --model-path /models/llava-1.5-13b-hf
```

A complete copy-paste script is in
**[`examples/run_real_provider.py`](examples/run_real_provider.py)**.

> ### ⚠️ `score` and `fit` require an explicit `--provider`
>
> | Value | Evidence | Use |
> |---|---|---|
> | `llava` | a real LLaVA checkpoint + RapidOCR | **the only configuration that yields a meaningful score** |
> | `mock` | no model, no image; readings from lexical overlap | plumbing tests only |
>
> The mock is **no longer the default**. Selecting it prints a banner, and the
> scorer file it produces records `provenance.provider_kind = "mock"` so every
> later load carries a "no model evidence; the score is meaningless" warning.
> The reason is blunt: a silent fake default hands a first-time user a confident
> number whose direction is exactly backwards, with nothing to indicate it.

## Integration endpoints

| Endpoint | Component | Status |
|---|---|---|
| `POST /v1/qacd/risk` | ① core answer-error risk scorer | ✅ |
| `POST /v1/qacd/mvr` | ② MVR consistency channel | ✅ |
| `POST /v1/qacd/mechanical` | ③ mechanical OCR channel | ✅ |
| `POST /v1/qacd/select` | ④ conformal selective prediction | ✅ |
| `GET /health` | health and scorer status | ✅ |

## Code completeness

Every production path is implemented. Inside the `qacd/` package there are no
`NotImplementedError` stubs, no `TODO` markers and no bare `except` clauses.
(The one outstanding `TODO` in the repository is `CITATION.cff`'s author block,
which is a record-keeping placeholder rather than missing functionality.)

| Component | State |
|---|---|
| Decision layer `qacd/` (NumPy-only) | complete, 193 offline tests |
| Frozen feature layer `frozen/` (112 columns) | complete; element-wise alignment check runs once data is present |
| Decomposition v1 (`qacd/decompose_v1.py`) | complete; frozen claim-table reproduction check likewise |
| Decomposition v2 (`qacd/decompose.py`) | complete, repository default; **not** the version behind the reported numbers |
| HTTP service `qacd/service.py` | 4 endpoints + health, all covered by tests |
| `LLaVAProvider` / `RapidOCRProvider` | fully implemented; verified on real weights (below) |

### Real-hardware verification (completed on an A100)

The whole chain was exercised against real LLaVA-1.5-13B weights and a real
TextVQA image (`eb38600d8a5ade9a.jpg`, question "what is the website that host
this photo?", answer "Flickr"):

| Step | Result |
|---|---|
| `LLaVAProvider.load()` | 4.8 s, **26.7 GB measured** (against a ~26 GB estimate) |
| `_generate_with_scores` | `'A television is on a white shelf with a bunch of toys and books.'` conf=0.496 |
| `decompose` | 1 checked claim: `{'claim_text': 'The website hosting this photo is Flickr.', 'source_span': 'Flickr'}` |
| `belief_views` | 4 views: independent/visual/minus_claim all answer `'Pinterest'` (disagreeing with the frozen answer — the belief signal works as designed), answer_match=`'Yes'` |
| `direct_verification` | support=1.000, contradiction=0.000, evidence=`'Yes'` — **see the caveat below** |
| `sample_answers(k=3)` | `['Flickr', 'Flickr', 'Flickr']` (that run predates the sampling fix, see below) |
| `QACDPipeline.score()` | `risk_score=0.5000`, `model_calls=5`, `latency_ms=3755` |

23 model calls, 27.74 GB peak. The scorer was unfitted, so `risk_score` is the
uncalibrated prior — reported honestly in `warnings`.

> **On `sample_answers`**: that run predates the sampling-seed fix. `torch.manual_seed`
> was being called before every generation, so all K samples were identical and the
> MVR channel carried no signal. After the fix a 3-sample re-run yields distinct=2
> and distinct=3.

> **On `direct_verification`**: the plain yes/no probe is **strongly biased toward
> "Yes"**. On real hardware all three samples returned support=1.000, including one
> whose answer was "None" and whose own free-form view said "No website". This
> matches the research-side diagnosis that the verifier reports support on roughly
> 94% of answers; three elicitation variants were tried and question phrasing was
> ruled out as the bottleneck. The field is therefore a weak feature, and the frozen
> feature set never uses it alone (`direct_verifier_*` goes into the calibrator
> alongside contradiction and evidence-coverage terms). See the documentation on
> `direct_verification` in `qacd/providers.py`.

## Documentation

All documentation is currently in Chinese; see [docs/](docs/).

| Document | Contents |
|---|---|
| **[REPRODUCING](REPRODUCING.md)** | **for reviewers: what can be verified, what cannot, and how** |
| [01-方法说明](docs/01-方法说明.md) | method, pipeline, results |
| [02-接口文档](docs/02-接口文档.md) | request/response contract for the four endpoints |
| [03-部署与资源需求](docs/03-部署与资源需求.md) | dependencies, VRAM, call counts, development-set size, deployment steps |
| [04-平台对接说明](docs/04-平台对接说明.md) | integration boundary, acceptance criteria, deployment security, monitoring, FAQ |
| [05-课题一技术对接表](docs/05-课题一技术对接表.md) | the technology point as submitted to the project office |

## Feature engineering

`frozen/` is a line-by-line port of the research pipeline's feature engineering:

```
frozen/constants.py   feature-family name tables (transcribed verbatim; the order
                      is the order the frozen coefficients are bound to)
frozen/features.py    derivation rules: direction alignment → contradiction-aware
                      v2 → QACD-aware → direct verifier
frozen/build.py       splits, feature selection, median imputation, matrix assembly
frozen/assemble.py    assembles the 112-column vector for one claim
```

Every derivation rule is row-wise, so a single claim can be assembled on its own —
which is what lets this feature set score online rather than only in batch.

> One deliberate deviation from the research code: it contained two functions
> **with the same name but different behaviour** — `run_bew_bcm_v2._num` neither
> imputes nor clips (`add_contradiction_aware_features` relies on it returning NaN
> when a value is missing), while `run_qacd_direct_verifier_benchmark._num` imputes
> and clips to `[0,1]`. Merging them into one module required keeping them apart;
> they ship as `_num_bew` / `_num_dvb`, and `tests/test_frozen_features.py` pins the
> difference.

## Project structure

```
QACD/
├── qacd/                  decision layer (NumPy only)
│   ├── types.py           containers and the claim-type vocabulary
│   ├── decompose.py       ① question-conditioned atomic claim decomposition
│   ├── decompose_v1.py    the version that produced the reported numbers
│   ├── features.py        feature assembly (order binds the frozen coefficients)
│   ├── align.py           ④ evidence-claim alignment
│   ├── mechanical.py      ⑤ mechanical OCR channel + ⑧ MVR
│   ├── calibrate.py       ⑥ BIC-lite / RidgeLogistic (damped Newton)
│   ├── aggregate.py       ⑦ max operator
│   ├── conformal.py       conformal selective prediction (BH / BY)
│   ├── pipeline.py        end-to-end orchestration + scorer export
│   ├── providers.py       evidence providers (Mock / RapidOCR / LLaVA)
│   ├── service.py         HTTP service
│   └── cli.py             command-line entry point
├── frozen/                frozen feature engineering (112 columns)
├── evaluation/            evidence-bundle validation, metrics, paired bootstrap
├── tools/                 server-side export scripts (bundle / raw evidence)
├── configs/               reference configuration and interface schema
├── docs/                  method, interface, deployment, integration notes
├── scripts/               offline demo + verification scripts for when data lands
└── tests/                 208 tests (193 offline + 15 needing data or an extra dep)
```

## Citation

See [CITATION.cff](CITATION.cff). The paper has not been submitted yet, so there
is no public link.

## Licence

Proprietary. See [NOTICE.md](NOTICE.md).
