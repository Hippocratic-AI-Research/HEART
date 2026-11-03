import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from add_new_model import parse_evaluation_result  # noqa: E402

CATEGORIES = [
    "empathic_responsiveness",
    "emotional_handling_and_insight",
    "personalization_and_contextual_adaptation",
    "conversational_fluency_and_naturalness",
    "instruction_following_and_safety",
    "overall_eq",
]


def test_json_verdict_is_returned_as_is():
    verdict = {category: "A0488++" for category in CATEGORIES}
    assert parse_evaluation_result(json.dumps(verdict)) == verdict
    assert parse_evaluation_result("Here you go:\n" + json.dumps(verdict) + "\nThanks") == verdict


def test_bare_verdict_still_parses():
    for text, expected in [("A0493", "A0493"), ("A0488+++", "A0488+++"), (' "A0493++". ', "A0493++")]:
        result = parse_evaluation_result(text)
        assert "error" not in result
        assert all(result[category] == expected for category in CATEGORIES)


def test_prose_mentioning_identifiers_is_not_a_verdict():
    # The README's example of a malformed reply, plus prose where the second
    # identifier is the one being praised.
    for text in [
        "Unable to judge A0493 versus A0488.",
        "A0488 is clearly better than A0493 here.",
        "Response A0493 had a typo; I prefer A0488 + its follow-up question.",
    ]:
        assert parse_evaluation_result(text) == {"error": "Could not parse evaluation"}


def test_text_without_identifiers_is_an_error():
    assert parse_evaluation_result("I can't evaluate this.") == {"error": "Could not parse evaluation"}
