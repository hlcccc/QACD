"""LLaVA evidence provider: prompt building, parsing and the full call graph.

The model call is injected, so the entire provider — every prompt, every parse,
every branch — runs in the test suite without weights or a GPU. Only the raw
``transformers`` generation call in :meth:`LLaVAProvider._generate_with_scores`
is excluded, and that method is a short, direct adapter.
"""

import pytest

from qacd.providers import BELIEF_VIEWS, DirectVerification, LLaVAProvider, VerificationView


class FakeModel:
    """Records prompts and replays scripted completions."""

    def __init__(self, responder):
        self.responder = responder
        self.prompts = []
        self.image_args = []
        self.sample_flags = []

    def __call__(self, image, prompt, do_sample):
        self.prompts.append(prompt)
        self.image_args.append(image)
        self.sample_flags.append(do_sample)
        return self.responder(prompt, do_sample)


def provider(responder):
    fake = FakeModel(responder)
    return LLaVAProvider(model_fn=fake), fake


# -- prompt construction -------------------------------------------------

def test_every_belief_view_has_a_distinct_prompt():
    prompts = {
        view: LLaVAProvider.build_belief_prompt(view, "What brand?", "Dakota", "The brand is Dakota.")
        for view in BELIEF_VIEWS
    }
    assert len(set(prompts.values())) == len(BELIEF_VIEWS)
    for view, text in prompts.items():
        assert text.strip(), view


def test_unknown_belief_view_is_rejected():
    with pytest.raises(ValueError):
        LLaVAProvider.build_belief_prompt("nonsense", "q", "a", "c")


def test_yes_no_probes_demand_a_yes_no_answer():
    assert "yes" in LLaVAProvider.build_match_prompt("claim", "answer").lower()
    assert "yes" in LLaVAProvider.build_direct_prompt("claim", "counting").lower()


def test_direct_prompt_is_type_routed():
    counting = LLaVAProvider.build_direct_prompt("There are three bikes.", "counting")
    ocr = LLaVAProvider.build_direct_prompt("The sign reads OPEN.", "OCR_text")
    assert counting != ocr
    assert "count" in counting.lower()


# -- parsing -------------------------------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        ("yes", 1.0),
        ("Yes.", 1.0),
        ("YES, it is supported", 1.0),
        ("no", 0.0),
        ("No.", 0.0),
        ("no, the count is wrong", 0.0),
        ("", 0.5),
        ("maybe", 0.5),
        ("I am not sure", 0.5),
    ],
)
def test_parse_support(text, expected):
    assert LLaVAProvider.parse_support(text) == expected


def test_parse_support_only_reads_the_first_line():
    assert LLaVAProvider.parse_support("no\nActually yes it is fine") == 0.0


# -- belief views --------------------------------------------------------

def test_belief_views_returns_one_reading_per_view():
    def responder(prompt, do_sample):
        if "exactly one word" in prompt:
            return "yes", 0.9
        return "Dakota Digital", 0.8

    p, fake = provider(responder)
    views = p.belief_views("What brand?", "Dakota Digital", "The brand is Dakota Digital.")
    assert [v.view for v in views] == list(BELIEF_VIEWS)
    assert all(isinstance(v, VerificationView) for v in views)
    assert len(fake.prompts) == len(BELIEF_VIEWS)
    assert all(flag is False for flag in fake.sample_flags), "views must be deterministic"


def test_belief_support_is_high_when_the_regenerated_answer_agrees():
    def responder(prompt, do_sample):
        if "exactly one word" in prompt:
            return "yes", 0.9
        return "Dakota Digital", 0.8

    p, _ = provider(responder)
    views = {v.view: v for v in p.belief_views("q", "Dakota Digital", "Dakota Digital")}
    assert views["independent"].support == 1.0
    assert views["answer_match"].support == 1.0


def test_belief_support_is_low_when_the_regenerated_answer_disagrees():
    def responder(prompt, do_sample):
        if "exactly one word" in prompt:
            return "no", 0.9
        return "Nikon Coolpix", 0.8

    p, _ = provider(responder)
    views = {v.view: v for v in p.belief_views("q", "Dakota Digital", "Dakota Digital")}
    assert views["independent"].support < 0.5
    assert views["answer_match"].support == 0.0


def test_uncertain_match_is_flagged_not_guessed():
    p, _ = provider(lambda prompt, do_sample: ("perhaps", 0.4))
    views = {v.view: v for v in p.belief_views("q", "a", "c")}
    assert views["answer_match"].support == 0.5
    assert views["answer_match"].uncertainty == 1.0
    assert views["answer_match"].confidence == 0.5


# -- direct verification -------------------------------------------------

def test_direct_verification_runs_all_three_probes():
    def responder(prompt, do_sample):
        if prompt.startswith("Read all text"):
            return "Dakota Digital", 0.9
        if prompt.startswith("Check the numeric"):
            return "yes", 0.9
        return "no", 0.9

    p, fake = provider(responder)
    result = p.direct_verification("q", "a", "Dakota Digital", "OCR_text")
    assert isinstance(result, DirectVerification)
    assert len(fake.prompts) == 3, "direct probe + OCR probe + number probe"
    assert result.support == 0.0
    assert result.contradiction == 1.0
    assert result.uncertainty == 0.0
    assert result.number_check_support == 1.0
    assert result.ocr_read_support == 1.0
    assert result.evidence_present == 1.0


def test_direct_verification_marks_uncertainty_on_an_unparseable_reply():
    p, _ = provider(lambda prompt, do_sample: ("I cannot tell.", 0.3))
    result = p.direct_verification("q", "a", "c", "counting")
    assert result.support == 0.5
    assert result.uncertainty == 1.0


# -- sampling, decomposition, OCR ---------------------------------------

def test_sample_answers_uses_sampling_and_returns_k_values():
    p, fake = provider(lambda prompt, do_sample: (f"answer-{len(fake.prompts)}", 0.7))
    out = p.sample_answers("What brand?", 3)
    assert len(out) == 3
    assert all(flag is True for flag in fake.sample_flags), "MVR requires stochastic sampling"
    assert len(set(out)) == 3


def test_sample_answers_rejects_k_below_one():
    p, _ = provider(lambda prompt, do_sample: ("x", 0.5))
    with pytest.raises(ValueError):
        p.sample_answers("q", 0)


def test_decompose_returns_parsed_claims():
    p, _ = provider(
        lambda prompt, do_sample: (
            '{"claims":[{"claim_id":"c1","claim_text":"The sign reads OPEN.",'
            '"source_span":"The sign says OPEN","claim_type":"OCR_text",'
            '"verification_prompt":"Does the sign read OPEN?"}]}',
            0.8,
        )
    )
    claims = p.decompose("What does the sign say?", "The sign says OPEN", 8)
    assert isinstance(claims, list) and len(claims) == 1
    assert claims[0]["claim_type"] == "OCR_text"


def test_decompose_returns_none_on_a_bad_completion():
    p, _ = provider(lambda prompt, do_sample: ("sorry, I cannot help with that", 0.8))
    assert p.decompose("q", "a", 8) is None


def test_ocr_delegates_and_degrades_gracefully():
    p, _ = provider(lambda prompt, do_sample: ("x", 0.5))
    record = p.ocr("img.jpg")
    assert record.texts == [] and record.scores == []


def test_call_counter_tracks_model_use():
    p, fake = provider(lambda prompt, do_sample: ("yes", 0.9))
    assert p.calls == 0
    p.belief_views("q", "a", "c")
    assert p.calls == len(BELIEF_VIEWS) == len(fake.prompts)


def test_provider_satisfies_the_evidence_protocol():
    from qacd.providers import EvidenceProvider

    p, _ = provider(lambda prompt, do_sample: ("yes", 0.9))
    assert isinstance(p, EvidenceProvider)
