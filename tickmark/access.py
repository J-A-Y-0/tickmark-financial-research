"""Who may use which source.

Two layers, both default-deny:
1. Route permissions per role: may this role query SEC filings, internal research, the web?
2. Document permissions: an internal document's `allowed_roles` (enforced in retrieval).

The policy is versioned. Cache partitions include the version, so editing the policy
makes every previously cached answer unreachable instead of silently stale.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from tickmark.models import Route


@dataclass(frozen=True)
class User:
    user_id: str
    name: str
    role: str
    role_label: str


class AccessPolicy:
    def __init__(self, version: str, roles: dict[str, dict], users: dict[str, dict]):
        self.version = version
        self.roles = {
            name: {"label": r["label"], "routes": frozenset(Route(x) for x in r["routes"])} for name, r in roles.items()
        }
        for user_id, user in users.items():
            if user["role"] not in self.roles:
                raise ValueError(f"user {user_id!r} has unknown role {user['role']!r}")
        self.users = users

    @classmethod
    def load(cls, path: Path) -> AccessPolicy:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(data["policy_version"], data["roles"], data["users"])

    def user(self, user_id: str | None) -> User | None:
        record = self.users.get(user_id or "")
        if record is None:
            return None
        role = record["role"]
        label = self.roles[role]["label"]
        return User(user_id, record.get("name", label), role, label)

    def can_use(self, role: str, route: Route) -> bool:
        return route in self.roles.get(role, {}).get("routes", frozenset())

    def allowed_routes(self, role: str) -> list[Route]:
        return [r for r in Route if self.can_use(role, r)]

    def scope_key(self, role: str) -> str:
        """Cache partition for a role under this policy version."""
        routes = ",".join(sorted(r.value for r in self.allowed_routes(role)))
        return hashlib.sha256(f"{self.version}|{role}|{routes}".encode()).hexdigest()[:16]

    def all_users(self) -> list[User]:
        return [self.user(user_id) for user_id in self.users]
