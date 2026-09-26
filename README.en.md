# QACD — Question-Conditioned Atomic Claim Decomposition

**Post-hoc answer-error risk scoring for LVLM visual question answering**

QACD is a *post-processing* risk scorer. It does not retrain or modify the model
under evaluation: given a frozen LVLM's answer together with the question and the
image, it returns an answer-level error-risk score in `[0, 1]` plus per-claim
risk detail. It is intended for early warning, human-review triage and
abstention policies.

```python
from qacd import QACDPipeline, QACDConfig

pipeline = QACDPipeline(config=QACDConfig(k=3))    # k > 0 enables the MVR channel
result = pipeline.score(
    question="What brand is the camera?",
    answer="Dakota digital",
    image="demo.jpg",
)
print(result.risk_score, result.is_high_risk, result.num_claims)
```

## Experimental data

This repository ships **without** experimental data or result values; the frozen
feature matrices, raw evidence tables and result tables are project deliverables
and are not published here. What ships is the method and the runnable code.

To run the verification scripts, export the data from the research host first:

```bash
python tools/export_raw_evidence.py    # raw evidence tables -> artifacts/raw/
python tools/export_bundle.py          # feature matrix + reference scores -> artifacts/
```

Without it those scripts exit with status **2** and name the missing inputs; they
never pass silently on empty input. The 14 data-dependent tests report as
`skipped` (`pytest -rs` prints why); the other 162 need no data.

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

The decision layer depends on **NumPy only** — no GPU, no model weights:

```bash
git clone https://github.com/hlcccc/QACD.git && cd QACD
pip install -r requirements.txt

python -m pytest -q                  # 162 passed + 14 skipped (data-dependent)
python scripts/demo_offline.py       # end-to-end demo on a synthetic dev set
qacd demo

# Fit a scorer on your own development set
# dev.jsonl: {"question":..., "answer":..., "image":..., "failed":0|1, "group":...}
qacd fit --data dev.jsonl --out scorer.json --k 3

# Serve the four integration endpoints
pip install "fastapi>=0.110" "uvicorn>=0.27" "pydantic>=2.0"
qacd serve --scorer scorer.json --port 8080
```

`qacd fit` and `qacd serve` default to `MockProvider` (deterministic, model-free)
so the plumbing can be exercised anywhere. Production deployments must inject
`LLaVAProvider` + `RapidOCRProvider`; see `qacd/providers.py`.

## Integration endpoints

| Endpoint | Component | Status |
|---|---|---|
| `POST /v1/qacd/risk` | ① core answer-error risk scorer | ✅ |
| `POST /v1/qacd/mvr` | ② MVR consistency channel | ✅ |
| `POST /v1/qacd/mechanical` | ③ mechanical OCR channel | ✅ |
| `POST /v1/qacd/select` | ④ conformal selective prediction | ✅ |
| `GET /health` | health and scorer status | ✅ |

## Documentation

All documentation is currently in Chinese; see [docs/](docs/).

## Licence

Proprietary. See [NOTICE.md](NOTICE.md).
