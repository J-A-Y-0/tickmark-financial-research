"""Download 10-K filings from SEC EDGAR and extract the sections Tickmark indexes.

Run: python data/build_filings.py

For each filing in FILINGS this finds the 10-K in the company's EDGAR submission
history, downloads the primary HTML document, converts it to text (tables become
pipe-separated rows), and keeps Item 1 (Business) and Item 7 (MD&A). The output is
one Markdown file per filing in data/filings/, with front matter recording the
source URL, accession number and a SHA-256 of the downloaded document.

SEC requires automated clients to identify themselves with a contact address, so
SEC_USER_AGENT must be set, e.g. "Jane Doe jane@example.com". Requests are
throttled well under SEC's 10 requests/second limit.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tickmark.text import html_to_text

OUT_DIR = Path(__file__).resolve().parent / "filings"

# (ticker, company, CIK, fiscal year)
FILINGS = [
    ("UBER", "Uber Technologies, Inc.", 1543151, 2021),
    ("UBER", "Uber Technologies, Inc.", 1543151, 2024),
    ("LYFT", "Lyft, Inc.", 1759509, 2024),
    ("DASH", "DoorDash, Inc.", 1792789, 2024),
]

SECTIONS = {
    "item-1": ("Item 1. Business", r"item\s*1\.?\s*business", r"item\s*1a\.?\s*risk\s*factors"),
    "item-7": (
        "Item 7. Management's Discussion and Analysis",
        r"item\s*7\.?\s*management",
        r"item\s*7a\.?\s*quantitative",
    ),
}


def _get(client: httpx.Client, url: str) -> httpx.Response:
    time.sleep(0.25)
    response = client.get(url)
    response.raise_for_status()
    return response


def find_10k(client: httpx.Client, cik: int, fiscal_year: int) -> dict:
    """The 10-K whose report date falls in the fiscal year (calendar fiscal years)."""
    data = _get(client, f"https://data.sec.gov/submissions/CIK{cik:010d}.json").json()
    recent = data["filings"]["recent"]
    for i, form in enumerate(recent["form"]):
        if form == "10-K" and recent["reportDate"][i].startswith(str(fiscal_year)):
            accession = recent["accessionNumber"][i]
            return {
                "accession": accession,
                "report_date": recent["reportDate"][i],
                "filing_date": recent["filingDate"][i],
                "url": (
                    f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}/{recent['primaryDocument'][i]}"
                ),
            }
    raise LookupError(f"No 10-K for CIK {cik} fiscal year {fiscal_year}")


def extract_section(text: str, start_pattern: str, end_pattern: str) -> str:
    """The longest span from a section heading to the next one.

    The table of contents also contains every heading, so the shortest match is the
    TOC entry; the body section is the longest.
    """
    best = ""
    starts = [m.start() for m in re.finditer(start_pattern, text, flags=re.I)]
    ends = [m.start() for m in re.finditer(end_pattern, text, flags=re.I)]
    for s in starts:
        following = [e for e in ends if e > s]
        if following and following[0] - s > len(best):
            best = text[s : following[0]]
    return best.strip()


def build_one(client: httpx.Client, ticker: str, company: str, cik: int, year: int) -> Path:
    meta = find_10k(client, cik, year)
    raw = _get(client, meta["url"]).text
    text = html_to_text(raw)
    parts = []
    for title, start, end in SECTIONS.values():
        body = extract_section(text, start, end)
        if len(body) < 2000:
            raise ValueError(f"{ticker} FY{year}: {title} not found ({len(body)} chars)")
        # drop the heading line itself; we write our own
        body = body.split("\n", 1)[1].strip() if "\n" in body else body
        parts.append(f"## {title}\n\n{body}")
    front = {
        "doc_id": f"{ticker.lower()}-10k-fy{year}",
        "title": f"{company} Form 10-K, fiscal year {year}",
        "company": company,
        "ticker": ticker,
        "fiscal_year": year,
        "form": "10-K",
        "accession": meta["accession"],
        "filing_date": meta["filing_date"],
        "source_url": meta["url"],
        "source_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "collection": "filings",
    }
    out = OUT_DIR / f"{front['doc_id']}.md"
    header = "---\n" + json.dumps(front, indent=2) + "\n---\n\n"
    out.write_text(header + "\n\n".join(parts) + "\n", encoding="utf-8")
    return out


def main() -> int:
    user_agent = os.environ.get("SEC_USER_AGENT")
    if not user_agent:
        print('Set SEC_USER_AGENT to "Your Name your@email" (required by SEC EDGAR).', file=sys.stderr)
        return 2
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with httpx.Client(headers={"User-Agent": user_agent}, timeout=60, follow_redirects=True) as client:
        for ticker, company, cik, year in FILINGS:
            path = build_one(client, ticker, company, cik, year)
            print(f"{path.name}: {path.stat().st_size // 1024} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
