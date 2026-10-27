"""Check an answer against the sources it cites.

For each sentence of the answer:
- every citation [n] must refer to a source that was actually retrieved (else `fabricated`);
- every figure must appear in one of the sentence's cited sources, allowing for units and
  rounding, so "$17.5 billion" matches a table cell of 17,455 in millions (else
  `number_unverified`);
- the sentence's words must overlap its cited sources enough to be plausibly supported
  (else `weak_support`); an optional Jev check can judge support semantically;
- a sentence with a figure but no citation is `uncited`, unless a source cited by the
  adjacent sentence in the same paragraph prints that figure ("X was $92. ... [7]");
- a sentence that only labels provenance ("This is a reported figure.") is not a claim.

The checks are deterministic and cheap, so they run on every answer. They flag rather than
rewrite: the reader sees exactly which sentences to double-check.
"""

from __future__ import annotations

import logging
import math
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from tickmark.models import Source, Usage
from tickmark.text import content_words, split_sentences

log = logging.getLogger(__name__)

SUPPORT_THRESHOLD = 0.4
SUPPORT_THRESHOLD_WITH_FIGURES = 0.25  # matched figures are already strong evidence
CITATION = re.compile(r"\[(\d{1,3})\]")
NUMBER = re.compile(
    r"(?<![\w.])(?P<dollar>\$\s?)?\(?(?P<int>\d{1,3}(?:,\d{3})+|\d+)(?:\.(?P<dec>\d+))?\)?"
    r"\s?(?P<unit>%|percent\b|trillion\b|billion\b|million\b|thousand\b|bn\b|mn\b|[BMK]\b)?",
    re.IGNORECASE,
)
SCALES = {"thousand": 1e3, "k": 1e3, "million": 1e6, "mn": 1e6, "m": 1e6, "billion": 1e9, "bn": 1e9, "b": 1e9, "trillion": 1e12}
NO_CLAIM = re.compile(
    r"\b(could not find|couldn't find|not (available|covered|found|included)|no (information|sources?)|"
    r"cannot access|can't access|not permitted|do not have access|don't have access|unable to)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Figure:
    text: str
    value: float
    decimals: int
    scale: float  # 1 when no unit word was written
    percent: bool
    dollar: bool

    @property
    def is_year_like(self) -> bool:
        return (
            not self.percent
            and not self.dollar
            and self.scale == 1
            and self.decimals == 0
            and "," not in self.text
            and 1900 <= self.value <= 2100
        )


def extract_figures(text: str) -> list[Figure]:
    figures = []
    for m in NUMBER.finditer(text):
        unit = (m.group("unit") or "").lower()
        value = float(m.group("int").replace(",", "") + ("." + m.group("dec") if m.group("dec") else ""))
        figures.append(
            Figure(
                text=m.group(0).strip(),
                value=value,
                decimals=len(m.group("dec") or ""),
                scale=SCALES.get(unit, 1.0),
                percent=unit in {"%", "percent"},
                dollar=bool(m.group("dollar")),
            )
        )
    return figures


# "This is a reported figure from the 10-K." labels where the previous claim came from;
# it states nothing checkable of its own.
PROVENANCE = re.compile(
    r"^(this|that|these|those|it|both)\s+(is|are|was|were)\s+(an?\s+|the\s+)?"
    r"(reported|filed|house|internal|10-k|firm's|estimated?|forecast|fictional|audited)\b",
    re.IGNORECASE,
)
# "12-month target", "3-year plan": the number describes a period, it is not a figure.
COMPOUND = re.compile(r"\b\d+(?:\.\d+)?-(?=[A-Za-z])")
DATE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+\d{1,2}\b,?",
    re.IGNORECASE,
)


def _claim_figures(sentence: str) -> list[Figure]:
    """Figures in an answer sentence worth checking: not years, dates or small counts."""
    return [
        f
        for f in extract_figures(COMPOUND.sub(" ", DATE.sub(" ", CITATION.sub(" ", sentence))))
        if not f.is_year_like and not (f.scale == 1 and not f.percent and not f.dollar and f.decimals == 0 and f.value <= 10)
    ]


def figure_supported(claim: Figure, source_figures: list[Figure]) -> bool:
    """True if a source figure equals the claim within the claim's displayed precision.

    A source figure without a unit word may be in thousands, millions or billions (10-K
    tables state the unit once in a header), so each of those scales is tried.
    """
    target = claim.value * claim.scale
    tolerance = 0.5 * 10 ** (-claim.decimals) * claim.scale + 1e-9 * max(1.0, target)
    for fig in source_figures:
        if fig.percent != claim.percent:
            continue
        scales = [fig.scale] if (fig.scale != 1 or fig.percent) else [1.0, 1e3, 1e6, 1e9]
        if any(math.isclose(fig.value * s, target, abs_tol=tolerance) for s in scales):
            return True
    return False


def _stem(word: str) -> str:
    return word[:6]


def support_score(sentence: str, sources_text: str) -> float:
    claim = {_stem(w) for w in content_words(CITATION.sub(" ", sentence)) if not w[0].isdigit()}
    if not claim:
        return 1.0
    available = {_stem(w) for w in content_words(sources_text)}
    return len(claim & available) / len(claim)


@dataclass
class SentenceCheck:
    text: str
    verdict: str  # verified | number_unverified | weak_support | contradicted | fabricated | uncited | skipped
    citations: list[int] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    # one entry per figure in the claim: what it was matched to, for the claim ledger
    figures: list[dict] = field(default_factory=list)


@dataclass
class Verification:
    sentences: list[SentenceCheck]
    usage: Usage = field(default_factory=Usage)
    verifier: str = "lexical"

    @property
    def checked(self) -> list[SentenceCheck]:
        return [s for s in self.sentences if s.verdict != "skipped"]

    def summary(self) -> dict:
        checked = self.checked
        cited = [s for s in checked if s.citations]
        verified = [s for s in checked if s.verdict == "verified"]
        return {
            "verifier": self.verifier,
            "sentences_checked": len(checked),
            "verified": len(verified),
            "flagged": len(checked) - len(verified),
            "grounding_rate": round(len([s for s in cited if s.verdict == "verified"]) / len(cited), 3) if cited else None,
            "fabricated_citations": sum(1 for s in checked if s.verdict == "fabricated"),
            "unverified_figures": sum(1 for s in checked if s.verdict == "number_unverified"),
        }

    def to_dict(self) -> dict:
        return {
            "summary": self.summary(),
            "sentences": [
                {"text": s.text, "verdict": s.verdict, "citations": s.citations, "issues": s.issues, "figures": s.figures}
                for s in self.sentences
                if s.verdict != "skipped"
            ],
        }


def answer_blocks(answer: str) -> list[list[str]]:
    """Sentences of a Markdown answer, grouped by paragraph or list item.

    A citation that trails a full stop stays on its sentence.
    """
    blocks: list[list[str]] = []
    for line in answer.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = re.sub(r"^([-*•]|\d+\.)\s+", "", line).replace("**", "")
        out: list[str] = []
        for sentence in split_sentences(line):
            lead = re.match(r"^((?:\[\d{1,3}\]\s*)+)(.*)$", sentence)
            if lead and out:
                out[-1] = f"{out[-1]} {lead.group(1).strip()}"
                sentence = lead.group(2).strip()
            if sentence:
                out.append(sentence)
        if out:
            blocks.append(out)
    return blocks


def answer_sentences(answer: str) -> list[str]:
    return [sentence for block in answer_blocks(answer) for sentence in block]


def verify_answer(answer: str, sources: list[Source], semantic=None) -> Verification:
    """Run the deterministic checks; `semantic` optionally re-judges support (see JevSupportChecker)."""
    by_number = {s.number: s for s in sources}
    checks: list[SentenceCheck] = []
    for block in answer_blocks(answer):
        for i, sentence in enumerate(block):
            checks.append(
                _check_sentence(sentence, block[i - 1] if i else "", block[i + 1] if i + 1 < len(block) else "", by_number)
            )
    result = Verification(checks)
    if semantic is not None:
        semantic.refine(result, by_number)
    return result


def _check_sentence(sentence: str, before: str, after: str, by_number: dict[int, Source]) -> SentenceCheck:
    cited = sorted({int(n) for n in CITATION.findall(sentence)})
    figures = _claim_figures(sentence)
    if not figures and PROVENANCE.search(CITATION.sub("", sentence).strip()):
        return SentenceCheck(sentence, "skipped")
    if not cited:
        if not figures or NO_CLAIM.search(sentence):
            return SentenceCheck(sentence, "skipped")
        # "X was $92. The note calls it fictional. [7]": a figure is accepted when a source
        # cited by the sentence just before or after it, in the same paragraph, prints it.
        for neighbour in (after, before):
            near = [n for n in sorted({int(n) for n in CITATION.findall(neighbour)}) if n in by_number]
            matches = [_match(f, near, by_number) for f in figures] if near else []
            if matches and all(m["source"] is not None for m in matches):
                return SentenceCheck(sentence, "verified", near, [], matches)
        return SentenceCheck(sentence, "uncited", [], ["contains a figure but cites no source"])
    missing = [n for n in cited if n not in by_number]
    if missing:
        return SentenceCheck(sentence, "fabricated", cited, [f"[{n}] is not a retrieved source" for n in missing])
    cited_text = "\n".join(by_number[n].text for n in cited)
    context = "\n".join(f"{by_number[n].title} {by_number[n].locator}" for n in cited)
    matches = [_match(f, cited, by_number) for f in figures]
    bad = [m["claim"] for m in matches if m["source"] is None]
    if bad:
        return SentenceCheck(sentence, "number_unverified", cited, [f"{b} not found in cited sources" for b in bad], matches)
    score = support_score(sentence, f"{context}\n{cited_text}")
    if score < (SUPPORT_THRESHOLD_WITH_FIGURES if figures else SUPPORT_THRESHOLD):
        issue = f"only {score:.0%} of the claim's terms appear in its sources"
        return SentenceCheck(sentence, "weak_support", cited, [issue], matches)
    return SentenceCheck(sentence, "verified", cited, [], matches)


def _match(claim: Figure, cited: list[int], by_number: dict[int, Source]) -> dict:
    """Where a claimed figure was found: the cited source number and the figure as the source prints it."""
    for n in cited:
        for fig in extract_figures(by_number[n].text):
            if figure_supported(claim, [fig]):
                return {"claim": claim.text, "source": n, "found": fig.text}
    return {"claim": claim.text, "source": None, "found": None}


class JevSupportChecker:
    """Semantic support check with TypeSafe Jev: does the cited text support the claim?

    Runs only on sentences that passed the citation and figure checks. A `contradicts`
    verdict overrides the lexical result; `unrelated` downgrades to weak_support; a
    `supports` verdict upgrades weak lexical overlap (paraphrase) to verified.
    """

    MAX_SENTENCES = 12

    def __init__(self, model: str = "jev-latest", client=None):
        from typesafe_sdk import Choice, TypeSafeClient

        self.model = model
        self.client = client or TypeSafeClient(model=model, timeout=10.0)
        self.question = Choice(
            instructions="How do the passages in `sources` relate to the statement in `claim`?",
            criteria={
                "supports": "The passages state or directly imply the claim, including its figures.",
                "contradicts": "The passages state something that conflicts with the claim.",
                "unrelated": "The passages do not address the claim, or only partly cover it.",
            },
        )

    def _judge(self, check: SentenceCheck, by_number: dict[int, Source]):
        from typesafe_sdk import TypeSafeError

        state = {"claim": CITATION.sub("", check.text).strip(), "sources": [by_number[n].text[:3000] for n in check.citations]}
        try:
            response = self.client.system_one(state=state, questions={"support": self.question})
        except TypeSafeError as exc:
            log.warning("Jev support check failed: %s", exc)
            return check, None, Usage()
        from tickmark.llm.pricing import usage_for

        return check, response.answers["support"], usage_for(self.model, response.usage.input_tokens, 0)

    def refine(self, result: Verification, by_number: dict[int, Source]) -> None:
        candidates = [c for c in result.sentences if c.verdict in {"verified", "weak_support"}][: self.MAX_SENTENCES]
        result.verifier = "lexical+jev"
        with ThreadPoolExecutor(max_workers=6) as pool:
            for check, answer, usage in pool.map(lambda c: self._judge(c, by_number), candidates):
                result.usage = result.usage + usage
                if answer is None:
                    continue
                if answer.choice == "contradicts":
                    check.verdict = "contradicted"
                    check.issues.append(f"Jev: sources contradict the claim (confidence {answer.confidence:.2f})")
                elif answer.choice == "unrelated":
                    check.verdict = "weak_support"
                    check.issues.append(f"Jev: sources do not address the claim (confidence {answer.confidence:.2f})")
                elif answer.choice == "supports" and check.verdict == "weak_support":
                    check.verdict = "verified"
                    check.issues = []
