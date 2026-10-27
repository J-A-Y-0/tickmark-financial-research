"""The bundled corpus and access policy are internally consistent."""

import json

from tickmark.access import AccessPolicy
from tickmark.config import ROOT
from tickmark.retrieval.ingest import corpus_files
from tickmark.text import parse_front_matter

DATA = ROOT / "data"


def test_every_document_has_valid_front_matter():
    policy = AccessPolicy.load(DATA / "access.json")
    ids = []
    for path in corpus_files(DATA):
        meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
        assert {"doc_id", "title", "collection"} <= meta.keys(), path
        assert meta["collection"] in {"filings", "internal"}, path
        assert body.strip().startswith("## "), path
        for role in meta.get("allowed_roles") or []:
            assert role in policy.roles, (path, role)
        ids.append(meta["doc_id"])
    assert len(ids) == len(set(ids))


def test_filings_record_their_provenance():
    filings = [p for p in corpus_files(DATA) if p.parent.name == "filings"]
    assert len(filings) == 4
    for path in filings:
        meta, _ = parse_front_matter(path.read_text(encoding="utf-8"))
        assert meta["source_url"].startswith("https://www.sec.gov/Archives/edgar/data/")
        assert len(meta["source_sha256"]) == 64 and meta["form"] == "10-K"
        assert "allowed_roles" not in meta  # public filings are open to every role with filings access


def test_restricted_memo_is_pm_only():
    meta, _ = parse_front_matter((DATA / "knowledge" / "deal-memo-project-atlas.md").read_text(encoding="utf-8"))
    assert meta["allowed_roles"] == ["pm"]


def test_policy_users_and_roles():
    data = json.loads((DATA / "access.json").read_text(encoding="utf-8"))
    policy = AccessPolicy.load(DATA / "access.json")
    assert {u.role for u in policy.all_users()} == set(data["roles"])
