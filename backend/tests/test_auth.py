"""Tenant authentication (G4).

Ownership used to come from an `x-owner-id` header, so any caller could claim any tenant: the data
was scoped but the boundary was self-declared. These tests pin the replacement — a hashed token that
resolves to an owner — and, just as importantly, that the single-user local studio still works
without one.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import rl_ide.api.server as server
from rl_ide.api.server import app
from rl_ide.state.auth_store import ApiTokenStore, get_token_store, hash_token


@pytest.fixture()
def token_store(tmp_path, monkeypatch):
    """A real token store on a scratch database, swapped into the running app."""
    store = ApiTokenStore(db_path=tmp_path / "tokens.db")
    monkeypatch.setattr(server, "get_token_store", lambda: store)
    monkeypatch.setattr(server, "_AUTH_REQUIRED_ENV", True)
    yield store
    monkeypatch.setattr(server, "_AUTH_REQUIRED_ENV", False)


def _issue(store: ApiTokenStore, owner: str = "acme") -> str:
    return store.create(owner_id=owner, label="test")["token"]


def test_tokens_are_stored_only_as_hashes(tmp_path):
    store = ApiTokenStore(db_path=tmp_path / "t.db")
    issued = store.create(owner_id="acme", label="ci")
    token = issued["token"]

    assert token.startswith("rlt_")
    assert len(token) > 30
    rows = store.list()
    assert rows and "token" not in rows[0]
    assert token not in str(rows)

    # Nothing in the database matches the plaintext.
    import sqlite3

    with sqlite3.connect(tmp_path / "t.db") as conn:
        dump = "\n".join(conn.iterdump())
    assert token not in dump
    assert hash_token(token) in dump


def test_verify_returns_the_owner_and_rejects_unknown_or_revoked(tmp_path):
    store = ApiTokenStore(db_path=tmp_path / "t.db")
    issued = store.create(owner_id="acme")
    assert store.verify(issued["token"]) == "acme"
    assert store.verify("rlt_not-a-real-token") is None
    assert store.verify("") is None

    assert store.revoke(issued["token_id"]) is True
    assert store.verify(issued["token"]) is None, "a revoked token must stop working immediately"
    assert store.revoke("tok_missing") is False


def test_verify_records_last_use(tmp_path):
    store = ApiTokenStore(db_path=tmp_path / "t.db")
    issued = store.create(owner_id="acme")
    assert store.list()[0]["last_used_at"] is None
    store.verify(issued["token"])
    assert store.list()[0]["last_used_at"] is not None


def test_requests_without_a_token_are_refused_when_auth_is_required(token_store):
    client = TestClient(app)
    resp = client.get("/api/v4/conversations")
    assert resp.status_code == 401
    assert "Authentication required" in resp.json()["detail"]
    assert resp.headers.get("WWW-Authenticate") == "Bearer"


def test_a_token_decides_the_tenant_and_the_owner_header_is_ignored(token_store):
    client = TestClient(app)
    acme = _issue(token_store, "acme")

    resp = client.get("/api/v4/whoami", headers={"Authorization": f"Bearer {acme}"})
    assert resp.status_code == 200
    assert resp.json() == {"owner_id": "acme", "authenticated": True}

    # Spoofing the old header must not change the tenant.
    spoofed = client.get(
        "/api/v4/whoami",
        headers={"Authorization": f"Bearer {acme}", "x-owner-id": "victim"},
    )
    assert spoofed.json()["owner_id"] == "acme"


def test_an_invalid_token_is_rejected(token_store):
    client = TestClient(app)
    _issue(token_store, "acme")
    resp = client.get("/api/v4/conversations", headers={"Authorization": "Bearer rlt_wrong"})
    assert resp.status_code == 401
    assert "Invalid" in resp.json()["detail"]


def test_data_created_by_one_tenant_is_invisible_to_another(token_store):
    client = TestClient(app)
    acme = _issue(token_store, "acme")
    beta = _issue(token_store, "beta")
    a = {"Authorization": f"Bearer {acme}"}
    b = {"Authorization": f"Bearer {beta}"}

    created = client.post("/api/v4/conversations", json={"title": "acme's agent"}, headers=a)
    assert created.status_code == 200
    conv_id = created.json()["conversation_id"]

    assert client.get(f"/api/v4/conversations/{conv_id}", headers=a).status_code == 200
    assert client.get(f"/api/v4/conversations/{conv_id}", headers=b).status_code == 404
    assert all(c["conversation_id"] != conv_id for c in client.get("/api/v4/conversations", headers=b).json())


def test_token_administration_requires_the_admin_token(token_store, monkeypatch):
    monkeypatch.setattr(server, "_ADMIN_TOKEN", "deploy-secret")
    client = TestClient(app)
    acme = _issue(token_store, "acme")

    # A tenant token cannot mint more tenants.
    assert client.post(
        "/api/v4/admin/tokens",
        json={"owner_id": "sneaky"},
        headers={"Authorization": f"Bearer {acme}"},
    ).status_code == 403

    minted = client.post(
        "/api/v4/admin/tokens",
        json={"owner_id": "gamma", "label": "pilot"},
        headers={"Authorization": "Bearer deploy-secret"},
    )
    assert minted.status_code == 200
    token = minted.json()["token"]
    assert token_store.verify(token) == "gamma"

    listed = client.get("/api/v4/admin/tokens", headers={"Authorization": "Bearer deploy-secret"})
    assert listed.status_code == 200
    assert any(row["owner_id"] == "gamma" for row in listed.json())
    assert all("token" not in row for row in listed.json())

    revoke = client.delete(
        f"/api/v4/admin/tokens/{minted.json()['token_id']}",
        headers={"Authorization": "Bearer deploy-secret"},
    )
    assert revoke.status_code == 200
    assert token_store.verify(token) is None


def test_local_single_user_mode_still_works_without_a_token(monkeypatch):
    """The desktop studio must not require credentials; that would break every existing user."""
    monkeypatch.setattr(server, "_AUTH_REQUIRED_ENV", False)
    monkeypatch.setattr(server, "get_token_store", lambda: (_ for _ in ()).throw(AssertionError("unused")))
    client = TestClient(app)
    resp = client.get("/api/v4/conversations")
    assert resp.status_code == 200
    assert client.get("/api/v4/whoami").json()["owner_id"] == "local"


def test_capabilities_reports_the_auth_mode(token_store):
    client = TestClient(app)
    acme = _issue(token_store, "acme")
    body = client.get("/api/v4/capabilities", headers={"Authorization": f"Bearer {acme}"}).json()
    assert body["auth"]["required"] is True
    assert body["auth"]["tenant_tokens"] >= 1
