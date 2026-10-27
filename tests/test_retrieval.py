import pytest

from tickmark.config import ROOT
from tickmark.retrieval.chunking import chunk_markdown, chunk_pdf, chunk_text
from tickmark.retrieval.embeddings import HashingEmbedder
from tickmark.retrieval.ingest import UploadWorker, build_corpus_index

ATLAS = "hpc-deal-atlas-2025-06"


def minimal_pdf(pages: list[str]) -> bytes:
    """A valid PDF with one line of Helvetica text per page (enough for pypdf to extract)."""
    objects = ["<< /Type /Catalog /Pages 2 0 R >>", None, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for text in pages:
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream.decode()}\nendstream")
        content_ref = len(objects)
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_ref} 0 R /Resources << /Font << /F1 3 0 R >> >> >>"
        )
        kids.append(f"{len(objects)} 0 R")
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
    out, offsets = b"%PDF-1.4\n", []
    for i, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{body}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    return out


class TestChunking:
    def test_sections_and_subheadings_become_locators(self):
        body = "## Item 7. Management's Discussion and Analysis\n\nOverview\n\n" + "Revenue grew strongly this year. " * 10
        chunks = chunk_markdown(body)
        assert chunks[0].locator == "Item 7. MD&A › Overview"

    def test_table_header_is_carried_to_continuation_rows(self):
        body = (
            "## Item 7. MD&A\n\n2024 | 2023 | % Change\n\n(in millions, except percentages)\n\n"
            "GAAP Financial Measures\n\nRevenue | 5,786.0 | 4,403.6 | 31%\n\nNet income (loss) | 22.8 | (340.3) | 107%"
        )
        table = next(c for c in chunk_markdown(body) if "Net income" in c.text)
        assert table.text.splitlines()[0] == "2024 | 2023 | % Change"
        assert "(in millions, except percentages)" in table.text

    def test_units_never_leak_onto_the_next_table(self):
        body = (
            "## Item 7. MD&A\n\n2024 | 2023\n\n(in millions)\n\nRevenue | 5,786.0 | 4,403.6\n\n"
            "The following table sets forth costs as a percentage of revenue:\n\nYear Ended December 31,\n\n"
            "2024 | 2023\n\nRevenue | 100.0% | 100.0%\n\nCosts and expenses\n\nCost of revenue | 57.7 | 57.8"
        )
        table = next(c for c in chunk_markdown(body) if "Cost of revenue" in c.text)
        assert "(in millions)" not in table.text
        assert table.text.startswith("The following table sets forth costs as a percentage of revenue:\n2024 | 2023")

    def test_header_is_not_carried_across_sections(self):
        body = "## A\n\n2024 | 2023\n\nRevenue | 1 | 2\n\n## B\n\nOther | 3 | 4 | and more words for length"
        other = next(c for c in chunk_markdown(body) if "Other" in c.text)
        assert "2024" not in other.text

    def test_duplicate_locators_are_numbered(self):
        body = "## S\n\nPart\n\n" + "\n\n".join("A paragraph about revenue growth. " * 30 for _ in range(3))
        locators = [c.locator for c in chunk_markdown(body)]
        assert len(locators) == len(set(locators))

    def test_plain_text_and_pdf(self):
        assert chunk_text("First paragraph with enough words.\n\nSecond paragraph here.")[0].locator == "part 1"
        pages = chunk_pdf(minimal_pdf(["Board deck page one about churn.", "Page two covers pricing."]))
        assert [p.locator for p in pages] == ["p. 1", "p. 2"]
        assert "churn" in pages[0].text


class TestSearch:
    def test_filters_by_company_and_fiscal_year(self, store):
        results = store.search("What was Uber's revenue in 2021?", "filings", "analyst")
        assert results and {r.doc_id for r in results} == {"uber-10k-fy2021"}

    def test_year_without_a_filing_falls_back_to_the_company(self, store):
        results = store.search("Uber free cash flow 2023", "filings", "analyst")
        assert {r.doc_id for r in results} == {"uber-10k-fy2024"}

    def test_comparison_retrieves_every_named_company(self, store):
        results = store.search("Compare Uber and Lyft Adjusted EBITDA in 2024", "filings", "analyst")
        assert {r.doc_id for r in results} == {"uber-10k-fy2024", "lyft-10k-fy2024"}

    def test_restricted_document_never_reaches_other_roles(self, store):
        for role in ("analyst", "consultant"):
            results = store.search("Project Atlas investment terms deal memo", "internal", role, k=20)
            assert ATLAS not in {r.doc_id for r in results}
        assert store.search("Project Atlas investment terms", "internal", "pm")[0].doc_id == ATLAS

    def test_citation_url_points_at_the_chunk(self, store):
        result = store.search("Lyft net income 2024", "filings", "analyst")[0]
        assert result.url.startswith("/documents/lyft-10k-fy2024#chunk-")

    def test_fts_syntax_in_queries_is_harmless(self, store):
        store.search('revenue" OR NEAR(uber AND "*', "filings", "analyst")


class TestIndexAndUploads:
    def test_index_build_is_idempotent(self, store):
        assert build_corpus_index(store, ROOT / "data") is False
        assert build_corpus_index(store, ROOT / "data", force=True) is True

    @pytest.mark.parametrize(
        "name, data",
        [
            ("board-notes.md", b"## Churn\n\nWe committed to reduce churn to 4 percent by Q4 across the board pack."),
            ("notes.txt", b"We committed to reduce churn to 4 percent by Q4 across the board pack."),
            ("deck.pdf", minimal_pdf(["We committed to reduce churn to 4 percent by Q4."])),
        ],
    )
    def test_upload_is_indexed_after_probe_and_respects_roles(self, store, name, data):
        worker = UploadWorker(store)
        doc_id = worker.submit(name, data, allowed_roles=["pm"], uploaded_by="pm")
        assert store.get_document(doc_id).status in {"pending", "parsing", "embedding", "indexed"}
        worker.wait()
        doc = store.get_document(doc_id)
        assert doc.status == "indexed", doc.error
        assert any(r.doc_id == doc_id for r in store.search("churn commitment", "internal", "pm"))
        assert not any(r.doc_id == doc_id for r in store.search("churn commitment", "internal", "analyst"))

    def test_unsupported_and_empty_uploads(self, store):
        worker = UploadWorker(store)
        with pytest.raises(ValueError):
            worker.submit("payload.exe", b"MZ", allowed_roles=["pm"], uploaded_by="pm")
        doc_id = worker.submit("empty.txt", b"   ", allowed_roles=["pm"], uploaded_by="pm")
        worker.wait()
        assert store.get_document(doc_id).status == "failed"

    def test_probe_detects_a_missing_chunk(self, store):
        assert store.probe(10**9) is False


def test_hashing_embedder_is_deterministic_and_normalised():
    a, b = HashingEmbedder().embed(["Uber revenue 2021", "Uber revenue 2021"])
    assert (a == b).all() and abs(float(a @ a) - 1) < 1e-5
