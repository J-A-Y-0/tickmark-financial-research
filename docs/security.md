# Access control and threat model

## Policy

`data/access.json` defines roles, the sources each role may query, and the users. Internal
documents add a document-level `allowed_roles` list in their front matter. Both layers
are default-deny: a role or document that does not grant access denies it.

| Control | Where | Enforced |
| --- | --- | --- |
| Unknown user rejected | `engine.ask` step 1 | Before any model call, search or cache read |
| Route permission | `engine.ask` after routing | Before retrieval and before the cache |
| Document permission | `DocumentStore.scope_documents` | Inside retrieval: unreadable chunks are never scored |
| Viewer and library | `web/app.py::_can_read` | Same two checks for pages and the documents API |
| Uploads | `POST /api/documents` | Only roles with internal access; visibility limited to roles that have it |
| Activity | `/activity`, `/api/activity` | Portfolio managers see all users; others see their own |

## Cache isolation

Cache partition = `sha256(policy version | role | allowed sources)` + index version.

| Scenario | Result | Test |
| --- | --- | --- |
| PM caches an answer from a restricted memo; analyst asks the same question | MISS; analyst's own search excludes the memo | `test_cached_pm_answer_is_never_served_to_an_analyst` |
| A request is fully or partly denied | Not looked up, not stored | `test_denials_are_never_cached` |
| Question is time-sensitive ("latest", "today", current year) | BYPASS | `test_time_sensitive_questions_bypass_the_cache` |
| Policy file changes | Version changes; old entries unreachable | `test_policy_change_invalidates_cached_answers` |
| New upload or index rebuild | Index version changes; old entries unreachable | `test_new_upload_invalidates_cached_answers` |
| A user's role changes | Next request uses the new role's partition | `test_role_change_moves_the_user_to_the_new_partition` |
| Answer cites a non-existent source | Not stored | `test_fabricated_citation_is_flagged_and_not_cached` |

## Threats considered

| Threat | Mitigation |
| --- | --- |
| Prompt injection asks the router for a restricted source | Routing cannot grant access; the access check follows routing. Tested with a router forced to pick internal research for an external consultant (`test_router_cannot_grant_access`). |
| Prompt injection claims a role ("as a portfolio manager I authorize…") | Identity comes from the session, never from the question. |
| Instructions hidden in retrieved web pages or uploads | The answer prompt treats sources as untrusted data; verification flags claims the sources don't support. |
| Cross-role leakage through the cache | Permission-scoped partitions (above). |
| XSS through source text | All source and answer text is escaped in the browser before formatting. |
| Open redirect after sign-in | `next` must be a local path. |
| FTS query syntax injection | Query terms are tokenised and quoted before reaching FTS5. |
| Unbounded spend | Per-user daily budget, deep-research cap, per-request time limits. |
| Secrets in the repository | Keys only from the environment; `.env` ignored; CI needs no secrets. |

## Not covered

The demo identity is a cookie naming a user in the policy file. It is a stand-in for SSO,
not authentication. Uploaded files are stored only as extracted text in SQLite, with no
malware scanning. A production deployment needs SSO, TLS termination, and an
organization-level audit sink.
