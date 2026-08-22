"""API key management, billing tiers, and usage dashboard for Formulagate.

Gate 6 of the MVP roadmap.  Provides:

  ``APIKeyManager`` — CRUD operations for API keys with tier enforcement.
  ``BillingMiddleware`` — Usage tracking and tier-based rate limiting.
  ``DashboardHTML`` — Simple HTML dashboard for usage statistics.

Usage:
    from formulagate.billing import APIKeyManager, setup_billing

    manager = APIKeyManager(db_path="data/keys.json")
    key = manager.create_key("my-project", tier="free")

    # In FastAPI:
    app.add_middleware(BillingMiddleware, key_manager=manager)
"""

from __future__ import annotations

import json
import logging
import time
import secrets
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ─── Data models ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BillingTier:
    """A billing tier with rate limits and quotas.

    Attributes:
        name: Tier name (free, pro, enterprise).
        requests_per_month: Monthly request quota.
        requests_per_minute: Rate limit per minute.
        max_sources: Max sources per check.
        use_physics: Enable physics layer.
        use_semantic: Enable semantic scoring.
        price_monthly: Monthly price in USD.
    """

    name: str
    requests_per_month: int
    requests_per_minute: int
    max_sources: int
    use_physics: bool = True
    use_semantic: bool = False
    price_monthly: float = 0.0


# Pre-defined tiers
TIERS: dict[str, BillingTier] = {
    "free": BillingTier(
        name="free",
        requests_per_month=1000,
        requests_per_minute=10,
        max_sources=10,
        use_physics=True,
        use_semantic=False,
        price_monthly=0.0,
    ),
    "pro": BillingTier(
        name="pro",
        requests_per_month=100000,
        requests_per_minute=100,
        max_sources=100,
        use_physics=True,
        use_semantic=True,
        price_monthly=49.0,
    ),
    "enterprise": BillingTier(
        name="enterprise",
        requests_per_month=-1,  # unlimited
        requests_per_minute=1000,
        max_sources=-1,  # unlimited
        use_physics=True,
        use_semantic=True,
        price_monthly=0.0,  # custom pricing
    ),
}


@dataclass
class APIKey:
    """An API key with metadata and usage tracking."""

    key_id: str
    key_hash: str
    project_name: str
    tier: str = "free"
    is_active: bool = True
    created_at: str = ""
    last_used_at: str = ""
    requests_this_month: int = 0
    total_requests: int = 0

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "key_id": self.key_id,
            "project_name": self.project_name,
            "tier": self.tier,
            "is_active": self.is_active,
            "created_at": self.created_at,
            "last_used_at": self.last_used_at,
            "requests_this_month": self.requests_this_month,
            "total_requests": self.total_requests,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> APIKey:
        return cls(
            key_id=data["key_id"],
            key_hash=data["key_hash"],
            project_name=data["project_name"],
            tier=data.get("tier", "free"),
            is_active=data.get("is_active", True),
            created_at=data.get("created_at", ""),
            last_used_at=data.get("last_used_at", ""),
            requests_this_month=data.get("requests_this_month", 0),
            total_requests=data.get("total_requests", 0),
        )


# ─── API Key Manager ─────────────────────────────────────────────────────────


class APIKeyManager:
    """CRUD operations for API keys with tier enforcement.

    Args:
        db_path: Path to the JSON file storing keys.
    """

    def __init__(self, db_path: str | Path = "data/api_keys.json") -> None:
        self._db_path = Path(db_path)
        self._keys: dict[str, APIKey] = {}
        self._hash_to_key: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        """Load keys from disk."""
        if self._db_path.exists():
            try:
                data = json.loads(self._db_path.read_text(encoding="utf-8"))
                for item in data.get("keys", []):
                    key = APIKey.from_dict(item)
                    self._keys[key.key_id] = key
                    self._hash_to_key[key.key_hash] = key.key_id
                logger.info("Loaded %d API keys from %s", len(self._keys), self._db_path)
            except Exception as exc:
                logger.error("Failed to load API keys: %s", exc)
        else:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._save()

    def _save(self) -> None:
        """Persist keys to disk."""
        data = {
            "keys": [k.to_dict() for k in self._keys.values()],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._db_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def create_key(self, project_name: str, tier: str = "free") -> tuple[str, str]:
        """Create a new API key.

        Args:
            project_name: Name of the project using this key.
            tier: Billing tier (free, pro, enterprise).

        Returns:
            (key_id, full_key) — full_key should be shown once and stored securely.
        """
        if tier not in TIERS:
            raise ValueError(f"Unknown tier: {tier}. Available: {list(TIERS.keys())}")

        key_id = str(uuid.uuid4())
        full_key = f"fg_{secrets.token_hex(24)}"
        key_hash = secrets.hashlib.sha256(full_key.encode()).hexdigest()

        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            project_name=project_name,
            tier=tier,
        )

        self._keys[key_id] = api_key
        self._hash_to_key[key_hash] = key_id
        self._save()

        logger.info("Created API key for project '%s' (tier=%s)", project_name, tier)
        return key_id, full_key

    def validate_key(self, full_key: str) -> APIKey | None:
        """Validate an API key and return the associated APIKey.

        Args:
            full_key: The full API key string.

        Returns:
            APIKey if valid, None otherwise.
        """
        key_hash = secrets.hashlib.sha256(full_key.encode()).hexdigest()
        key_id = self._hash_to_key.get(key_hash)

        if not key_id:
            return None

        api_key = self._keys.get(key_id)
        if not api_key or not api_key.is_active:
            return None

        # Update usage
        now = datetime.now(timezone.utc).isoformat()
        api_key.last_used_at = now
        api_key.requests_this_month += 1
        api_key.total_requests += 1
        self._save()

        return api_key

    def revoke_key(self, key_id: str) -> bool:
        """Revoke an API key.

        Args:
            key_id: The key ID to revoke.

        Returns:
            True if key was found and revoked.
        """
        api_key = self._keys.get(key_id)
        if api_key:
            api_key.is_active = False
            self._save()
            logger.info("Revoked API key for project '%s'", api_key.project_name)
            return True
        return False

    def get_usage(self, key_id: str) -> dict[str, Any] | None:
        """Get usage statistics for an API key.

        Args:
            key_id: The key ID to get usage for.

        Returns:
            Usage dict or None if key not found.
        """
        api_key = self._keys.get(key_id)
        if not api_key:
            return None

        tier = TIERS.get(api_key.tier, TIERS["free"])
        remaining = max(0, tier.requests_per_month - api_key.requests_this_month) if tier.requests_per_month > 0 else -1

        return {
            "key_id": key_id,
            "project_name": api_key.project_name,
            "tier": api_key.tier,
            "requests_this_month": api_key.requests_this_month,
            "total_requests": api_key.total_requests,
            "monthly_limit": tier.requests_per_month,
            "remaining": remaining,
            "last_used_at": api_key.last_used_at,
            "created_at": api_key.created_at,
        }

    def list_keys(self) -> list[dict[str, Any]]:
        """List all API keys (without full key values)."""
        return [k.to_dict() for k in self._keys.values()]


# ─── Billing Middleware ──────────────────────────────────────────────────────


class BillingMiddleware:
    """Middleware that validates API keys and enforces tier limits.

    Args:
        key_manager: APIKeyManager instance.
        header_name: HTTP header containing the API key.
    """

    def __init__(
        self,
        key_manager: APIKeyManager,
        header_name: str = "x-api-key",
    ) -> None:
        self._key_manager = key_manager
        self._header_name = header_name

    def __call__(self, scope, receive, send) -> None:
        """ASGI interface."""
        from starlette.middleware.base import BaseHTTPMiddleware
        from starlette.requests import Request
        from starlette.responses import JSONResponse

        class _Middleware(BaseHTTPMiddleware):
            async def dispatch(self, request: Request, call_next):
                api_key = request.headers.get(self._header_name)

                if not api_key:
                    return JSONResponse(
                        status_code=401,
                        content={
                            "error": "missing_api_key",
                            "message": f"API key required. Send via {self._header_name} header.",
                        },
                    )

                validated = self._key_manager.validate_key(api_key)
                if not validated:
                    return JSONResponse(
                        status_code=403,
                        content={
                            "error": "invalid_api_key",
                            "message": "The provided API key is invalid or revoked.",
                        },
                    )

                tier = TIERS.get(validated.tier, TIERS["free"])
                if tier.requests_per_month > 0 and validated.requests_this_month >= tier.requests_per_month:
                    return JSONResponse(
                        status_code=429,
                        content={
                            "error": "quota_exceeded",
                            "message": f"Monthly quota exceeded for {validated.tier} tier.",
                            "limit": tier.requests_per_month,
                            "used": validated.requests_this_month,
                        },
                    )

                # Add user info to request state
                request.state.api_key = validated
                request.state.tier = tier

                response = await call_next(request)
                response.headers["X-Api-Key-Id"] = validated.key_id
                response.headers["X-Api-Key-Tier"] = validated.tier

                usage = self._key_manager.get_usage(validated.key_id)
                if usage:
                    response.headers["X-Requests-This-Month"] = str(usage["requests_this_month"])
                    response.headers["X-Requests-Remaining"] = str(usage["remaining"])

                return response

        _middleware = _Middleware(None)
        import asyncio

        async def _wrapped():
            from starlette.middleware.base import RequestResponseEndpoint
            from starlette.requests import Request

            # Create a simple request wrapper
            class _ScopeRequest:
                def __init__(self, scope):
                    self.scope = scope
                    self.state = type('State', (), {'api_key': None, 'tier': None})()

                @property
                def headers(self):
                    return _Headers(scope.get("headers", []))

            class _Headers:
                def __init__(self, headers):
                    self._headers = dict(h[0].decode() if isinstance(h[0], bytes) else h[0] for h in headers)

                def get(self, key, default=None):
                    return self._headers.get(key.lower(), default)

            req = _ScopeRequest(scope)

            api_key_val = req.headers.get(self._header_name)
            if not api_key_val:
                body = json.dumps({
                    "error": "missing_api_key",
                    "message": f"API key required. Send via {self._header_name} header.",
                }).encode()
                await send({
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [[b"content-type", b"application/json"]],
                })
                await send({"type": "http.response.body", "body": body})
                return

            validated = self._key_manager.validate_key(api_key_val)
            if not validated:
                body = json.dumps({
                    "error": "invalid_api_key",
                    "message": "The provided API key is invalid or revoked.",
                }).encode()
                await send({
                    "type": "http.response.start",
                    "status": 403,
                    "headers": [[b"content-type", b"application/json"]],
                })
                await send({"type": "http.response.body", "body": body})
                return

            tier = TIERS.get(validated.tier, TIERS["free"])
            if tier.requests_per_month > 0 and validated.requests_this_month >= tier.requests_per_month:
                body = json.dumps({
                    "error": "quota_exceeded",
                    "message": f"Monthly quota exceeded for {validated.tier} tier.",
                    "limit": tier.requests_per_month,
                    "used": validated.requests_this_month,
                }).encode()
                await send({
                    "type": "http.response.start",
                    "status": 429,
                    "headers": [[b"content-type", b"application/json"]],
                })
                await send({"type": "http.response.body", "body": body})
                return

            # Process request
            body = b""
            while True:
                msg = await receive()
                if msg["type"] == "http.disconnect":
                    break
                if msg.get("body"):
                    body += msg["body"]
                if not msg.get("more_body"):
                    break

            # Call next
            response_body = b""
            response_start = {}

            async def _send(event):
                nonlocal response_body, response_start
                if event["type"] == "http.response.start":
                    response_start = event
                elif event["type"] == "http.response.body":
                    if event.get("body"):
                        response_body += event["body"]

            await call_next(_receive_wrapper(scope, receive))
            # Simplified: just pass through with headers
            final_headers = response_start.get("headers", [])
            final_headers.append((b"x-api-key-tier", validated.tier.encode()))

            await send({
                "type": "http.response.start",
                "status": 200,
                "headers": final_headers,
            })
            await send({"type": "http.response.body", "body": response_body or b'{"status":"ok"}'})

        asyncio.create_task(_wrapped())


def _receive_wrapper(scope, receive):
    """Create a receive wrapper that returns empty body."""
    async def _receive():
        return {"type": "http.request", "body": b""}
    return _receive


# ─── Dashboard HTML ──────────────────────────────────────────────────────────


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Formulagate Dashboard</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; color: #333; }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        header {{ background: #1a237e; color: white; padding: 20px 0; margin-bottom: 30px; }}
        header .container {{ display: flex; justify-content: space-between; align-items: center; }}
        header h1 {{ font-size: 24px; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .stat-card {{ background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .stat-card h3 {{ color: #666; font-size: 14px; margin-bottom: 10px; }}
        .stat-card .value {{ font-size: 32px; font-weight: bold; color: #1a237e; }}
        .stat-card .label {{ font-size: 12px; color: #999; }}
        .section {{ background: white; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .section h2 {{ margin-bottom: 15px; color: #1a237e; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f8f9fa; font-weight: 600; color: #666; }}
        .badge {{ display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; }}
        .badge-free {{ background: #e3f2fd; color: #1565c0; }}
        .badge-pro {{ background: #fff3e0; color: #e65100; }}
        .badge-enterprise {{ background: #e8f5e9; color: #2e7d32; }}
        .badge-active {{ background: #e8f5e9; color: #2e7d32; }}
        .badge-inactive {{ background: #ffebee; color: #c62828; }}
        .progress-bar {{ background: #eee; border-radius: 4px; height: 8px; overflow: hidden; }}
        .progress-bar .fill {{ height: 100%; background: #1a237e; border-radius: 4px; transition: width 0.3s; }}
        .api-key {{ font-family: monospace; background: #f5f5f5; padding: 2px 6px; border-radius: 3px; }}
        button {{ padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer; font-size: 14px; }}
        .btn-primary {{ background: #1a237e; color: white; }}
        .btn-danger {{ background: #c62828; color: white; }}
        .btn:hover {{ opacity: 0.9; }}
        .modal {{ display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5); z-index: 1000; }}
        .modal.show {{ display: flex; align-items: center; justify-content: center; }}
        .modal-content {{ background: white; border-radius: 8px; padding: 30px; max-width: 500px; width: 90%; }}
        .modal-content h2 {{ margin-bottom: 20px; }}
        .form-group {{ margin-bottom: 15px; }}
        .form-group label {{ display: block; margin-bottom: 5px; font-weight: 600; }}
        .form-group input, .form-group select {{ width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 4px; }}
        .key-display {{ background: #f5f5f5; padding: 15px; border-radius: 4px; margin: 15px 0; word-break: break-all; font-family: monospace; }}
    </style>
</head>
<body>
    <header>
        <div class="container">
            <h1>Formulagate Dashboard</h1>
            <div>
                <button class="btn btn-primary" onclick="showCreateModal()">+ New API Key</button>
            </div>
        </div>
    </header>

    <div class="container">
        <div class="stats" id="stats">
            <div class="stat-card">
                <h3>Total API Keys</h3>
                <div class="value" id="total-keys">0</div>
            </div>
            <div class="stat-card">
                <h3>Total Requests</h3>
                <div class="value" id="total-requests">0</div>
            </div>
            <div class="stat-card">
                <h3>Active Keys</h3>
                <div class="value" id="active-keys">0</div>
            </div>
            <div class="stat-card">
                <h3>This Month</h3>
                <div class="value" id="this-month">0</div>
            </div>
        </div>

        <div class="section">
            <h2>API Keys</h2>
            <table>
                <thead>
                    <tr>
                        <th>Project</th>
                        <th>Tier</th>
                        <th>Status</th>
                        <th>Requests</th>
                        <th>Created</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody id="keys-table"></tbody>
            </table>
        </div>
    </div>

    <div class="modal" id="create-modal">
        <div class="modal-content">
            <h2>Create New API Key</h2>
            <div class="form-group">
                <label for="project-name">Project Name</label>
                <input type="text" id="project-name" placeholder="my-project">
            </div>
            <div class="form-group">
                <label for="tier-select">Tier</label>
                <select id="tier-select">
                    <option value="free">Free (1,000/month)</option>
                    <option value="pro">Pro (100,000/month) - $49</option>
                    <option value="enterprise">Enterprise (Unlimited)</option>
                </select>
            </div>
            <button class="btn btn-primary" onclick="createKey()">Create Key</button>
            <button class="btn" onclick="hideCreateModal()" style="background: #eee; margin-left: 10px;">Cancel</button>
            <div id="key-result" style="display: none; margin-top: 15px;">
                <p><strong>Your API Key (save this now!):</strong></p>
                <div class="key-display" id="new-key-display"></div>
            </div>
        </div>
    </div>

    <script>
        const API_BASE = window.location.origin;

        async function loadDashboard() {{
            try {{
                const res = await fetch(`${{API_BASE}}/admin/keys`);
                const data = await res.json();
                renderDashboard(data);
            }} catch (e) {{
                console.error('Failed to load dashboard:', e);
            }}
        }}

        function renderDashboard(data) {{
            document.getElementById('total-keys').textContent = data.total_keys || 0;
            document.getElementById('active-keys').textContent = data.active_keys || 0;
            document.getElementById('total-requests').textContent = data.total_requests || 0;
            document.getElementById('this-month').textContent = data.this_month || 0;

            const tbody = document.getElementById('keys-table');
            tbody.innerHTML = (data.keys || []).map(key => `
                <tr>
                    <td>${{key.project_name}}</td>
                    <td><span class="badge badge-${{key.tier}}">${{key.tier}}</span></td>
                    <td><span class="badge badge-${{key.is_active ? 'active' : 'inactive'}}">${{key.is_active ? 'Active' : 'Revoked'}}</span></td>
                    <td>${{key.total_requests.toLocaleString()}} (this month: ${{key.requests_this_month}})</td>
                    <td>${{new Date(key.created_at).toLocaleDateString()}}</td>
                    <td>
                        ${{!key.is_active ? '<span style="color:#999">Revoked</span>' :
                        `<button class="btn btn-danger" onclick="revokeKey('${{key.key_id}}')">Revoke</button>`}}
                    </td>
                </tr>
            `).join('');
        }}

        function showCreateModal() {{
            document.getElementById('create-modal').classList.add('show');
            document.getElementById('key-result').style.display = 'none';
        }}

        function hideCreateModal() {{
            document.getElementById('create-modal').classList.remove('show');
        }}

        async function createKey() {{
            const projectName = document.getElementById('project-name').value;
            const tier = document.getElementById('tier-select').value;

            if (!projectName) {{
                alert('Please enter a project name');
                return;
            }}

            try {{
                const res = await fetch(`${{API_BASE}}/admin/keys`, {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{project_name: projectName, tier}})
                }});
                const data = await res.json();
                document.getElementById('new-key-display').textContent = data.full_key;
                document.getElementById('key-result').style.display = 'block';
                loadDashboard();
            }} catch (e) {{
                alert('Failed to create key: ' + e.message);
            }}
        }}

        async function revokeKey(keyId) {{
            if (!confirm('Revoke this API key?')) return;
            try {{
                await fetch(`${{API_BASE}}/admin/keys/${{keyId}}`, {{method: 'DELETE'}});
                loadDashboard();
            }} catch (e) {{
                alert('Failed to revoke key: ' + e.message);
            }}
        }}

        loadDashboard();
    </script>
</body>
</html>"""


def setup_billing(
    app: Any,
    key_manager: APIKeyManager | None = None,
    db_path: str | Path = "data/api_keys.json",
) -> APIKeyManager:
    """Set up billing endpoints on a FastAPI app.

    Args:
        app: FastAPI application instance.
        key_manager: Optional existing APIKeyManager.
        db_path: Path to keys database.

    Returns:
        APIKeyManager instance.
    """
    if key_manager is None:
        key_manager = APIKeyManager(db_path=db_path)

    from fastapi import HTTPException

    @app.get("/admin/keys")
    async def list_keys():
        """List all API keys with usage stats."""
        keys = key_manager.list_keys()
        total_requests = sum(k["total_requests"] for k in keys)
        this_month = sum(k["requests_this_month"] for k in keys)
        active = sum(1 for k in keys if k["is_active"])

        return {
            "total_keys": len(keys),
            "active_keys": active,
            "total_requests": total_requests,
            "this_month": this_month,
            "keys": keys,
        }

    @app.post("/admin/keys")
    async def create_key_endpoint(body: dict[str, str]):
        """Create a new API key."""
        project_name = body.get("project_name", "unnamed")
        tier = body.get("tier", "free")

        try:
            key_id, full_key = key_manager.create_key(project_name, tier)
            return {
                "key_id": key_id,
                "full_key": full_key,
                "tier": tier,
                "message": "Save this key now — it will not be shown again!",
            }
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.delete("/admin/keys/{key_id}")
    async def revoke_key_endpoint(key_id: str):
        """Revoke an API key."""
        if not key_manager.revoke_key(key_id):
            raise HTTPException(404, "Key not found")
        return {"status": "ok", "message": "Key revoked"}

    @app.get("/admin/dashboard")
    async def dashboard():
        """Serve the HTML dashboard."""
        from fastapi.responses import HTMLResponse

        return HTMLResponse(content=DASHBOARD_HTML)

    @app.get("/admin/usage/{key_id}")
    async def usage_endpoint(key_id: str):
        """Get usage for a specific key."""
        usage = key_manager.get_usage(key_id)
        if not usage:
            raise HTTPException(404, "Key not found")
        return usage

    return key_manager
