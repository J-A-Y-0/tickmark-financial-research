"""Split documents into retrievable chunks that carry a human-readable locator.

A locator is what a citation shows: "Item 7. MD&A › Results of Operations" for a filing,
"p. 4" for a PDF page, or a heading for Markdown. Tables are kept whole (or split into
row groups that repeat the header row) so a figure is never separated from its column
labels.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass

TARGET_CHARS = 1100
MIN_CHARS = 40  # drops page artefacts such as stray footnote markers
SECTION_SHORT = {"Item 7. Management's Discussion and Analysis": "Item 7. MD&A"}


@dataclass
class Chunk:
    ordinal: int
    locator: str
    text: str  # what the answer quotes and the citation shows


def _is_table(block: str) -> bool:
    return " | " in block


def _is_subheading(block: str) -> bool:
    return (
        "\n" not in block
        and len(block) < 90
        and not block.endswith((".", ":", ";", ","))
        and not _is_table(block)
        and block[:1].isupper()
    )


YEAR_CELL = re.compile(r"^(FY\s?)?(19|20)\d\d$")
_UNITS = re.compile(r"^\((in|In|dollars in) [^)]*(thousands|millions|billions)[^)]*\)$")


def is_header_row(row: str) -> bool:
    """A table row that labels columns with years, e.g. '2024 | 2023 | % Change'."""
    return sum(bool(YEAR_CELL.match(cell.strip())) for cell in row.split(" | ")) >= 2


def _carry_headers(blocks: list[str]) -> list[str]:
    """Give each table block the context it continues from: lead-in, column header and units.

    10-K tables often put '2024 | 2023 | % Change' and '(in millions)' above a sub-heading,
    so the rows that follow would otherwise lose which column is which year. Context is
    carried only when it belongs to the same table: the lead-in sentence ('... as a
    percentage of revenue:') must come just before the header row and the units line just
    after it, so one table's units never leak onto the next.
    """
    out: list[str] = []
    header = caption = units = None
    lead_in: tuple[str, int] | None = None
    since = 99
    for i, block in enumerate(blocks):
        if block.startswith("## "):
            header = caption = units = lead_in = None
        elif _is_table(block):
            first = block.split("\n", 1)[0]
            if is_header_row(first) or (" | " in first and is_header_row(first.split(" | ", 1)[1])):
                header, units, since = first, None, 0
                caption = lead_in[0] if lead_in and i - lead_in[1] <= 3 else None
            elif header and since <= 12:
                prefix = [p for p in (caption, header, units) if p and p not in block]
                block = "\n".join([*prefix, block])
        elif _UNITS.match(block):
            if header and since <= 2:
                units = block
        elif block.endswith(":") and len(block) < 300:
            lead_in = (block, i)
        out.append(block)
        since += 1
    return out


def _blocks(body: str) -> list[str]:
    """Paragraph blocks, with consecutive table rows merged into one block."""
    blocks: list[str] = []
    for raw in re.split(r"\n\s*\n", body):
        block = raw.strip()
        if not block:
            continue
        if _is_table(block) and blocks and _is_table(blocks[-1]):
            blocks[-1] += "\n" + block
        else:
            blocks.append(block)
    return _carry_headers(blocks)


def _split_table(table: str) -> list[str]:
    rows = table.split("\n")
    if len(table) <= TARGET_CHARS * 2:
        return [table]
    header, groups, current = rows[0], [], [rows[0]]
    for row in rows[1:]:
        if sum(len(r) for r in current) + len(row) > TARGET_CHARS * 2 and len(current) > 1:
            groups.append("\n".join(current))
            current = [header]
        current.append(row)
    groups.append("\n".join(current))
    return groups


def _split_long(text: str) -> list[str]:
    """Split an over-long paragraph at sentence boundaries."""
    if len(text) <= TARGET_CHARS * 2:
        return [text]
    pieces, current = [], ""
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if current and len(current) + len(sentence) > TARGET_CHARS:
            pieces.append(current.strip())
            current = ""
        current += sentence + " "
    if current.strip():
        pieces.append(current.strip())
    return pieces


def chunk_markdown(body: str, default_section: str = "") -> list[Chunk]:
    """Chunk Markdown whose sections start with `## ` and whose sub-sections are short lines."""
    chunks: list[Chunk] = []
    section, sub = default_section, ""
    buffer: list[str] = []
    counts: dict[str, int] = {}

    def locator() -> str:
        base = " › ".join(p for p in (SECTION_SHORT.get(section, section), sub) if p) or "Document"
        counts[base] = counts.get(base, 0) + 1
        return base if counts[base] == 1 else f"{base} (part {counts[base]})"

    def flush() -> None:
        text = "\n\n".join(buffer)
        if len(text) >= MIN_CHARS:
            chunks.append(Chunk(len(chunks), locator(), text))
        buffer.clear()

    for block in _blocks(body):
        if block.startswith("#"):
            flush()
            heading = block.lstrip("#").strip()
            if block.startswith("## ") or block.startswith("# "):
                section, sub = heading, ""
            else:
                sub = heading
            continue
        if _is_subheading(block):
            flush()
            sub = block
            continue
        pieces = _split_table(block) if _is_table(block) else _split_long(block)
        for piece in pieces:
            if buffer and sum(len(b) for b in buffer) + len(piece) > TARGET_CHARS:
                flush()
            buffer.append(piece)
    flush()
    if not chunks:  # a document shorter than one fragment is still a document
        text = "\n".join(line for line in body.splitlines() if not line.lstrip().startswith("#")).strip()
        if text:
            chunks.append(Chunk(0, SECTION_SHORT.get(section, section) or "Document", text))
    return chunks


def chunk_text(text: str) -> list[Chunk]:
    """Plain text: paragraphs grouped to the target size, located by part number."""
    chunks: list[Chunk] = []
    buffer: list[str] = []
    for block in _blocks(text):
        for piece in _split_long(block):
            if buffer and sum(len(b) for b in buffer) + len(piece) > TARGET_CHARS:
                chunks.append(Chunk(len(chunks), f"part {len(chunks) + 1}", "\n\n".join(buffer)))
                buffer = []
            buffer.append(piece)
    if buffer:
        chunks.append(Chunk(len(chunks), f"part {len(chunks) + 1}", "\n\n".join(buffer)))
    return chunks


def chunk_pdf(data: bytes) -> list[Chunk]:
    """One or more chunks per page, located by page number."""
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    chunks: list[Chunk] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = re.sub(r"[ \t]+", " ", page.extract_text() or "").strip()
        if not text:
            continue
        pieces = _split_long(re.sub(r"\n(?!\n)", " ", text)) if len(text) > TARGET_CHARS * 2 else [text]
        for part, piece in enumerate(pieces, start=1):
            locator = f"p. {page_number}" if len(pieces) == 1 else f"p. {page_number} (part {part})"
            chunks.append(Chunk(len(chunks), locator, piece))
    return chunks
