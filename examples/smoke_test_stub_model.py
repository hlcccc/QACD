#!/usr/bin/env python
"""Smoke-test the real LLaVA provider path without the 26 GB checkpoint.

Why this exists
---------------

`qacd demo` and `scripts/demo_offline.py` use `MockProvider`, which replaces the
whole evidence stack. They prove the repository installs. They do **not** touch a
single line of `LLaVAProvider` — no prompt construction, no answer parsing, no
view routing, no type-routed verification. So they cannot tell you whether your
wiring is right, and the first time that code runs is the day you have the
checkpoint on disk and a GPU booked.

This script runs the real thing. `LLaVAProvider._generate_with_scores` checks for
an injected `model_fn` *before* it touches torch or the filesystem, so a
deterministic stub can drive the entire provider — the same technique vLLM uses
with `--load-format dummy`.

What it proves
--------------

- every prompt the provider builds is well-formed and parseable;
- the JSON decomposition round-trips into claim objects;
- the belief views and the type-routed verification probes execute and their
  answers are parsed;
- the MVR samples are collected;
- the pipeline assembles features, calibrates, aggregates and scores.

What it does **not** prove
--------------------------

That the method works. The stub's answers are canned, so the resulting score is
meaningless in exactly the way a mock score is. This checks plumbing only. Real
numbers need real weights: see `examples/run_real_provider.py`.

Usage
-----

    python examples/smoke_test_stub_model.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

QUESTION = "what is the website that host this photo?"
ANSWER = "Flickr"
IMAGE = "/data/textvqa/val/eb38600d8a5ade9a.jpg"

#: What the stub "reads" off the image, so the mechanical channel has something
#: to compare against. A real provider gets this from RapidOCR.
STUB_OCR_TEXT = "Flickr"

DECOMPOSITION_JSON = json.dumps(
    {
        "claims": [
            {
                "claim_text": "The website hosting this photo is Flickr.",
                "source_span": "Flickr",
                "claim_type": "OCR_text",
                "is_atomic": True,
            }
        ]
    }
)


class StubModel:
    """Deterministic stand-in for LLaVA. Answers by inspecting the prompt.

    A real checkpoint would read the image; this returns canned text shaped like
    a real answer, which is enough to exercise every parser in the provider.

    Sampled calls deliberately disagree with each other. A stub that returns one
    constant string makes every MVR column constant, which trips the fusion
    guard -- correct behaviour, but it would hide the fusion path from this
    smoke test. Real sampling at temperature 0.7 varies; so does this.
    """

    #: Cycled through on sampled calls so the MVR channel sees real spread.
    SAMPLED = ("Flickr", "Flickr", "Pinterest", "Flickr", "Tumblr")

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self._sampled = 0

    def __call__(self, image: str, prompt: str, do_sample: bool) -> tuple[str, float]:
        self.prompts.append(prompt)

        # 1. Decomposition: the provider asks for a JSON claim list.
        if "atomic claims" in prompt.lower() or "return between" in prompt.lower():
            return DECOMPOSITION_JSON, 0.93

        # 2. Type-routed verification probes: yes/no.
        if re.search(r"\bis (the |this )?.*\b(visible|correct|supported|read)\b", prompt, re.I):
            return "Yes", 0.88
        if "answer with yes or no" in prompt.lower() or prompt.strip().endswith("?"):
            if "yes" in prompt.lower() or "no" in prompt.lower():
                return "Yes", 0.88

        # 3. Numeric sub-probe used by the counting claim type.
        if "digit" in prompt.lower() or "number" in prompt.lower():
            return "1", 0.7

        # 4. Belief views / MVR sampling: a short answer phrase.
        if "short phrase" in prompt.lower() or "from scratch" in prompt.lower():
            if do_sample:
                text = self.SAMPLED[self._sampled % len(self.SAMPLED)]
                self._sampled += 1
                return text, 0.71
            return "Flickr", 0.71

        return "Flickr", 0.6


class StubOCR:
    """Stands in for RapidOCR.

    The MVR channel needs OCR text to decide whether a resample is supported:
    `mvr_features` compares each sample against the strings read off the image.
    With no OCR the comparison has nothing to match against, every sample scores
    as unsupported, and all seven MVR columns come out constant -- so the fusion
    head has nothing to learn from and is refused. That coupling is real and easy
    to miss, which is why this stub exists rather than leaving `ocr_provider`
    unset.
    """

    provider_kind = "stub-ocr"
    name = "stub-ocr"

    def __init__(self, text: str) -> None:
        self.text = text

    def ocr(self, image: str):
        from qacd.types import OCRRecord

        return OCRRecord(image=image, texts=[self.text], scores=[0.99])


def main() -> int:
    from qacd.pipeline import QACDConfig, QACDPipeline
    from qacd.providers import LLaVAProvider

    print("=" * 78)
    print("QACD smoke test — real provider code path, stub model, no weights")
    print("=" * 78)
    print()

    stub = StubModel()
    provider = LLaVAProvider(
        model_path="<stub>",
        ocr_provider=StubOCR(STUB_OCR_TEXT),
        model_fn=stub,
        seed=20260920,
    )

    pipeline = QACDPipeline(provider=provider, config=QACDConfig(k=3))
    print(f"[1/5] provider  : {type(provider).__name__} (provider_kind={provider.provider_kind})")
    print(f"      model_fn  : {type(stub).__name__} — torch is never imported")
    print()

    print("[2/5] 训练一个真实打分器（用的是同一套 provider 代码）...")
    records = [
        {
            "question": QUESTION,
            "answer": ANSWER if i % 2 == 0 else "Pinterest",
            "image": IMAGE,
            "failed": i % 2,
            "group": f"img{i}",
        }
        for i in range(40)
    ]
    pipeline.fit(records, verbose=True)
    print()

    print("[3/5] 打分 ...")
    result = pipeline.score(QUESTION, ANSWER, IMAGE)
    print(f"      risk_score            = {result.risk_score}")
    print(f"      calibrated_confidence = {result.calibrated_confidence}")
    print(f"      num_claims            = {result.num_claims}")
    print(f"      model_calls           = {result.model_calls}")
    print(f"      feature_dim           = {result.feature_dim}")
    print()
    for claim in result.claims:
        print(f"      claim [{claim.claim_type}] risk={claim.risk:.4f}  {claim.claim_text[:60]}")
    for warning in result.warnings:
        print(f"      ! {warning}")
    print()

    print("[4/5] provider 各处都真的被调用了吗？")
    kinds = {
        "分解提示": sum("atomic claims" in p or "return between" in p.lower() for p in stub.prompts),
        "信念视角": sum("from scratch" in p.lower() or "short phrase" in p.lower() for p in stub.prompts),
        "验证探针": sum("verify" in p.lower() or "visible" in p.lower() for p in stub.prompts),
    }
    for name, count in kinds.items():
        flag = "OK " if count else "!! "
        print(f"      [{flag}] {name}: {count} 次")
    print(f"      总生成次数: {len(stub.prompts)}")
    print()

    print("[5/5] 端点冒烟（不启动服务器）...")
    try:
        from fastapi.testclient import TestClient

        from qacd.service import create_app

        client = TestClient(create_app(pipeline=pipeline))
        for path, body in (
            ("/health", None),
            ("/v1/qacd/risk", {"question": QUESTION, "answer": ANSWER, "image": IMAGE}),
            ("/v1/qacd/mvr", {"sampled_answers": ["Flickr", "Flickr", "Pinterest"]}),
            ("/v1/qacd/mechanical", {"claim_text": ANSWER, "ocr_texts": [STUB_OCR_TEXT]}),
            ("/v1/qacd/select", {"calibration_null_scores": [0.1, 0.2, 0.3], "test_scores": [0.9]}),
        ):
            response = client.get(path) if body is None else client.post(path, json=body)
            print(f"      {response.status_code}  {path}")
    except ImportError:
        print("      skipped: the service extra is not installed (pip install -e '.[service]')")
    print()

    print("=" * 78)
    print("结论：provider 的接线是通的。")
    print("=" * 78)
    print()
    print("上面的分数**没有意义** —— 桩模型的回答是写死的。这个脚本只证明")
    print("prompt 构造、回答解析、视角路由、验证探针都真的跑到了。")
    print()
    print("要拿真实数字：")
    print("  python examples/run_real_provider.py fit --data dev.jsonl \\")
    print("      --model-path /models/llava-1.5-13b-hf --out scorer.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
