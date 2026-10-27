"""Text helpers: HTML to text, tokens, sentences and front matter."""

from __future__ import annotations

import html
import json
import re
from html.parser import HTMLParser

STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "it's",
        "of",
        "on",
        "or",
        "our",
        "over",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "to",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "about",
        "after",
        "also",
        "any",
        "between",
        "both",
        "during",
        "each",
        "more",
        "most",
        "other",
        "per",
        "such",
        "through",
        "under",
        "up",
        "vs",
        "versus",
    ]
)

_WORD = re.compile(r"[a-z0-9]+(?:[.'][a-z0-9]+)*")
# A sentence ends at . ! or ?, optionally followed by a closing quote or bracket.
_SENTENCE = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"”’)]))\s+(?=[A-Z0-9“\"(\[*-])")


def words(text: str) -> list[str]:
    return [w[:-2] if w.endswith("'s") else w for w in _WORD.findall(text.lower().replace("’", "'"))]


def content_words(text: str) -> list[str]:
    return [w for w in words(text) if w not in STOPWORDS and len(w) > 1]


def split_sentences(text: str) -> list[str]:
    parts: list[str] = []
    for block in text.split("\n"):
        block = block.strip()
        if block:
            parts.extend(s.strip() for s in _SENTENCE.split(block) if s.strip())
    return parts


def parse_front_matter(raw: str) -> tuple[dict, str]:
    """Split `---\\n{json}\\n---` front matter from a Markdown body."""
    if raw.startswith("---\n"):
        head, sep, body = raw[4:].partition("\n---\n")
        if sep:
            return json.loads(head), body.lstrip("\n")
    return {}, raw


_BLOCK_TAGS = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table", "section"}


class _TextExtractor(HTMLParser):
    """HTML to text that keeps paragraph breaks and turns table rows into `a | b | c`."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.row: list[str] | None = None
        self.cell: list[str] | None = None
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        style = (dict(attrs).get("style") or "").replace(" ", "")
        if tag in {"script", "style", "noscript", "svg"} or "display:none" in style:
            self.skip += 1
            return
        if tag == "tr":
            self.row = []
        elif tag in {"td", "th"} and self.row is not None:
            self.cell = []
        elif tag in _BLOCK_TAGS and self.row is None:
            self.out.append("\n")

    def handle_endtag(self, tag):
        if self.skip and tag in {"script", "style", "noscript", "svg", "div", "span"}:
            self.skip -= 1
            return
        if tag in {"td", "th"} and self.row is not None and self.cell is not None:
            text = " ".join("".join(self.cell).split())
            if text and text not in {"$", "%", ")"}:
                self.row.append(text)
            elif text in {"%", ")"} and self.row:
                self.row[-1] += text
            self.cell = None
        elif tag == "tr" and self.row is not None:
            if self.row:
                self.out.append("\n" + " | ".join(self.row) + "\n")
            self.row = None
        elif tag in _BLOCK_TAGS and self.row is None:
            self.out.append("\n")

    def handle_data(self, data):
        if self.skip:
            return
        if self.cell is not None:
            self.cell.append(data)
        elif self.row is None:
            self.out.append(data)


def html_to_text(raw: str) -> str:
    parser = _TextExtractor()
    parser.feed(raw)
    text = html.unescape("".join(parser.out)).replace("\xa0", " ")
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
