import json

import pytest
from fastapi.testclient import TestClient

from web.app import create_app


@pytest.fixture
def client(make_engine):
    with TestClient(create_app(make_engine())) as c:
        yield c


def sse_events(response) -> list[dict]:
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]


@pytest.mark.parametrize("path", ["/", "/library", "/activity", "/evals", "/documents/uber-10k-fy2021"])
def test_pages_render(client, path):
    response = client.get(path)
    assert response.status_code == 200
    assert "Tickmark" in response.text


def test_ask_streams_server_sent_events(client):
    response = client.post("/api/ask", json={"question": "What was Lyft's net income in 2024?"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = sse_events(response)
    assert events[0]["type"] == "start" and events[-1]["type"] == "done"
    assert "22.8" in "".join(e["text"] for e in events if e["type"] == "token")


def test_identity_from_cookie_and_header(client):
    client.post("/session", data={"user_id": "consultant", "next": "/"}, follow_redirects=False)
    events = sse_events(client.post("/api/ask", json={"question": "What is our price target on Uber?"}))
    assert events[-1]["status"] == "denied"
    events = sse_events(client.post("/api/ask", json={"question": "Uber revenue 2021?"}, headers={"X-User-Id": "nobody"}))
    assert events[-1]["status"] == "rejected"


def test_session_rejects_unknown_users_and_open_redirects(client):
    assert client.post("/session", data={"user_id": "nobody"}).status_code == 400
    response = client.post("/session", data={"user_id": "pm", "next": "//evil.example"}, follow_redirects=False)
    assert response.headers["location"] == "/"


def test_restricted_document_viewer_is_denied(client):
    assert "No access" in client.get("/documents/hpc-deal-atlas-2025-06").text
    client.post("/session", data={"user_id": "pm"})
    assert "Project Atlas" in client.get("/documents/hpc-deal-atlas-2025-06").text


def test_library_lists_only_readable_documents(client):
    ids = {d["doc_id"] for d in client.get("/api/documents").json()}
    assert "hpc-deal-atlas-2025-06" not in ids and "uber-10k-fy2021" in ids
    client.post("/session", data={"user_id": "consultant"})
    ids = {d["doc_id"] for d in client.get("/api/documents").json()}
    assert ids and all(not i.startswith("hpc-") for i in ids)


def test_upload_flow_and_permissions(client):
    response = client.post(
        "/api/documents",
        files={"file": ("q3.md", b"## Q3\n\nWe committed to cut churn to 4 percent.")},
        data={"visibility": "private"},
    )
    assert response.status_code == 202
    doc_id = response.json()["doc_id"]
    client.app.state.engine.uploads.wait()
    doc = next(d for d in client.get("/api/documents").json() if d["doc_id"] == doc_id)
    assert doc["status"] == "indexed" and doc["allowed_roles"] == ["analyst", "pm"]
    assert client.post("/api/documents", files={"file": ("x.exe", b"MZ")}).status_code == 415
    client.post("/session", data={"user_id": "consultant"})
    assert client.post("/api/documents", files={"file": ("a.txt", b"hello")}).status_code == 403


def test_activity_is_scoped_by_role(client):
    client.post("/api/ask", json={"question": "Uber revenue 2021?"})
    client.post("/session", data={"user_id": "consultant"})
    client.post("/api/ask", json={"question": "Lyft revenue 2024?"})
    assert {r["user_id"] for r in client.get("/api/activity").json()} == {"consultant"}
    client.post("/session", data={"user_id": "pm"})
    assert {r["user_id"] for r in client.get("/api/activity").json()} == {"analyst", "consultant"}


def test_config_estimate_health_and_missing_report(client):
    assert client.get("/api/config").json()["allowed_routes"] == ["filings", "internal", "web"]
    assert client.get("/api/estimate?depth=deep").json()["deep_daily_cap"] == 5
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/evals/report.json").status_code in {200, 404}


def test_stale_cookie_falls_back_to_the_default_access_level(client):
    client.cookies.set("tickmark_user", "someone-removed-from-the-policy")
    assert "Research analyst" in client.get("/").text
    events = sse_events(client.post("/api/ask", json={"question": "Uber revenue 2021?"}))
    assert events[-1]["status"] != "rejected"
