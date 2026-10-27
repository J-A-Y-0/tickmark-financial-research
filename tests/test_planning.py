import pytest

from tests.conftest import FakeChat, plan_json
from tickmark.models import Depth
from tickmark.planning import HeuristicPlanner, LLMPlanner, validate_sub_questions


class TestHeuristicPlanner:
    def test_single_question_stays_single(self):
        plan = HeuristicPlanner().plan("What was Uber's revenue in 2021?", Depth.QUICK)
        assert [s.text for s in plan.sub_questions] == ["What was Uber's revenue in 2021?"]

    @pytest.mark.parametrize(
        "question",
        [
            "What was Uber's revenue in 2021, and what is our price target on Uber?",
            "What was Uber's revenue in 2021 and what is our price target on Uber?",
            "What was Uber's revenue in 2021? What is our price target on Uber?",
            "What was Uber's revenue in 2021; what is our price target on Uber?",
        ],
    )
    def test_splits_two_part_questions(self, question):
        plan = HeuristicPlanner().plan(question, Depth.QUICK)
        assert [s.text for s in plan.sub_questions] == [
            "What was Uber's revenue in 2021?",
            "What is our price target on Uber?",
        ]

    def test_does_not_split_a_comparison(self):
        plan = HeuristicPlanner().plan("Compare Uber and Lyft revenue in 2024", Depth.QUICK)
        assert len(plan.sub_questions) == 1

    def test_deep_mode_plans_at_least_three(self):
        plan = HeuristicPlanner().plan("How did DoorDash grow in 2024?", Depth.DEEP)
        assert len(plan.sub_questions) == 3
        assert [s.index for s in plan.sub_questions] == [1, 2, 3]


class TestLLMPlanner:
    def test_uses_model_plan(self):
        chat = FakeChat([plan_json("What was Uber's revenue in 2021?", "What is our price target on Uber?")])
        plan = LLMPlanner(chat).plan("Uber revenue 2021 and our target?", Depth.QUICK)
        assert len(plan.sub_questions) == 2
        assert plan.fallback is None
        assert plan.usage.calls == 1

    @pytest.mark.parametrize(
        "reply",
        [
            "not json at all",
            '{"sub_questions": "one"}',
            '{"sub_questions": []}',
            '{"sub_questions": [{"question": ""}]}',
            '["a list"]',
            plan_json("a", "b", "c", "d", "e"),  # more than the quick maximum
        ],
    )
    def test_malformed_plan_falls_back_to_original_question(self, reply):
        plan = LLMPlanner(FakeChat([reply])).plan("What was Lyft's revenue in 2024?", Depth.QUICK)
        assert [s.text for s in plan.sub_questions] == ["What was Lyft's revenue in 2024?"]
        assert plan.fallback

    def test_deep_plan_with_too_few_sub_questions_falls_back(self):
        plan = LLMPlanner(FakeChat([plan_json("only one")])).plan("Deep question?", Depth.DEEP)
        assert len(plan.sub_questions) == 1 and plan.fallback

    def test_provider_error_falls_back(self):
        plan = LLMPlanner(FakeChat(fail=True)).plan("What was Lyft's revenue in 2024?", Depth.QUICK)
        assert len(plan.sub_questions) == 1 and "provider down" in plan.fallback

    def test_json_inside_code_fence_is_accepted(self):
        reply = "```json\n" + plan_json("A?", "B?") + "\n```"
        plan = LLMPlanner(FakeChat([reply])).plan("A and B?", Depth.QUICK)
        assert [s.text for s in plan.sub_questions] == ["A?", "B?"]


def test_validate_truncates_long_text():
    subs = validate_sub_questions({"sub_questions": ["x" * 1000]}, 1, 4)
    assert len(subs[0].text) == 300
