import pytest

from tickmark.models import Route, Source
from tickmark.verify import answer_sentences, extract_figures, figure_supported, verify_answer

TABLE = Source(
    kind=Route.FILINGS,
    title="Uber Technologies, Inc. Form 10-K, fiscal year 2021",
    locator="Item 7. MD&A › Financial and Operational Highlights",
    text="(In millions, except percentages) | 2020 | 2021\nRevenue | 11,139 | 17,455 | 57% | 54%\nNet loss | (6,768) | (496) | 93%",
    number=1,
)
NOTE = Source(
    kind=Route.INTERNAL,
    title="Uber coverage initiation",
    text="Harbor Point initiates coverage of Uber with an Overweight house rating and a price target of $92.",
    number=2,
)


@pytest.mark.parametrize(
    "claim, ok",
    [
        ("$17.5 billion", True),  # rounded from 17,455 in millions
        ("$17,455 million", True),
        ("17,455", True),
        ("$17.46 billion", True),
        ("$17.6 billion", False),
        ("$17 billion", True),  # rounding to the nearest billion is still faithful
        ("$18 billion", False),
        ("57%", True),
        ("58%", False),
        ("$6.8 billion", True),  # the loss shown in parentheses
    ],
)
def test_figure_matching_handles_units_and_rounding(claim, ok):
    assert figure_supported(extract_figures(claim)[0], extract_figures(TABLE.text)) is ok


def test_percent_never_matches_a_plain_number():
    assert not figure_supported(extract_figures("54%")[0], extract_figures("Revenue | 54"))


def test_verdicts():
    answer = (
        "Uber's revenue was $17.5 billion in 2021, up 57% [1]. "
        "Revenue was $19.1 billion [1]. "
        "Harbor Point rates Uber Overweight with a $92 target [2]. "
        "It grew 12% [7]. "
        "Lyft revenue was $5.8 billion. "
        "Uber also launched rockets to Mars [1]."
    )
    verdicts = [s.verdict for s in verify_answer(answer, [TABLE, NOTE]).checked]
    assert verdicts == ["verified", "number_unverified", "verified", "fabricated", "uncited", "weak_support"]


def test_summary_counts():
    result = verify_answer("Revenue was $17.5 billion [1]. Revenue was $1 billion [9].", [TABLE])
    summary = result.summary()
    assert summary["verified"] == 1 and summary["fabricated_citations"] == 1
    assert summary["grounding_rate"] == 0.5


def test_years_and_small_counts_are_not_checked_as_figures():
    result = verify_answer("In 2021 Uber reported three segments across 3 regions [1].", [TABLE])
    assert result.checked[0].verdict in {"verified", "weak_support"}
    assert not result.checked[0].issues or "not found" not in result.checked[0].issues[0]


def test_statements_of_missing_information_need_no_citation():
    result = verify_answer("I could not find the 2022 figure in the filings.", [TABLE])
    assert result.checked == []


def test_trailing_citation_after_full_stop_stays_with_its_sentence():
    assert answer_sentences("Revenue was $17.5 billion. [1] Rating is Overweight. [2]") == [
        "Revenue was $17.5 billion. [1]",
        "Rating is Overweight. [2]",
    ]


def test_markdown_structure_is_ignored():
    sentences = answer_sentences("### Answer\n- **Revenue** was $17.5 billion [1].\n\n1. Rated Overweight [2].")
    assert sentences == ["Revenue was $17.5 billion [1].", "Rated Overweight [2]."]


def test_sentences_split_after_a_closing_quote():
    assert answer_sentences('The note calls it "fictional." [7] No later update exists.') == [
        'The note calls it "fictional." [7]',
        "No later update exists.",
    ]


def test_ledger_records_where_each_figure_was_found():
    check = verify_answer("Uber's revenue was $17.5 billion in 2021, up 57% [1].", [TABLE]).checked[0]
    assert check.figures == [
        {"claim": "$17.5 billion", "source": 1, "found": "17,455"},
        {"claim": "57%", "source": 1, "found": "57%"},
    ]


def test_dates_are_not_figures():
    check = verify_answer("For the year ended December 31, 2021, revenue was $17.5 billion [1].", [TABLE]).checked[0]
    assert [f["claim"] for f in check.figures] == ["$17.5 billion"]


def test_citation_on_the_next_sentence_covers_a_figure():
    answer = "Harbor Point rates Uber Overweight with a price target of $92. The note calls it fictional. [2]"
    assert verify_answer(answer, [TABLE, NOTE]).checked[0].verdict == "verified"


def test_a_neighbouring_citation_must_actually_contain_the_figure():
    answer = "The price target is $150. The note rates Uber Overweight [2]."
    assert verify_answer(answer, [TABLE, NOTE]).checked[0].verdict == "uncited"


def test_neighbours_in_another_paragraph_do_not_count():
    answer = "The price target is $92.\n\nThe note rates Uber Overweight [2]."
    assert verify_answer(answer, [TABLE, NOTE]).checked[0].verdict == "uncited"


def test_provenance_labels_are_not_claims_but_figures_still_are():
    answer = "Revenue was $17.5 billion [1]. This is a reported figure from the 10-K. [1]"
    assert [c.verdict for c in verify_answer(answer, [TABLE]).checked] == ["verified"]
    assert verify_answer("This is a reported figure of $19 billion [1].", [TABLE]).checked[0].verdict == "number_unverified"


def test_period_descriptors_are_not_figures():
    check = verify_answer("Harbor Point set a 12-month price target of $92 [2].", [NOTE]).checked[0]
    assert [f["claim"] for f in check.figures] == ["$92"]
