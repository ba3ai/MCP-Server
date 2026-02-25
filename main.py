from __future__ import annotations

import os
import json
import logging
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.middleware.sessions import SessionMiddleware

from dotenv import load_dotenv

from app.db import init_db
from app.crypto import encrypt
from app.oauth_verify import verify_bearer_token
from app.qbo import exchange_code_for_tokens, build_intuit_auth_url
from app import db
from app.request_context import current_user
from app.ui import router as ui_router
from app.mcp_app import mcp
from fastapi import Request
load_dotenv()

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("qbo_mcp")

# NOTE: Avoid auto-redirects (307) caused by missing/extra trailing slashes.
# Some clients drop the Authorization header when following redirects.
@asynccontextmanager
async def lifespan(app_: FastAPI):
    # Initialize FastMCP Streamable HTTP session manager.
    # Without this, FastMCP raises: 'Task group is not initialized. Make sure to use run().' 
    async with mcp.session_manager.run():
        await init_db()
        yield


app = FastAPI(redirect_slashes=False, lifespan=lifespan)


app.add_middleware(
    SessionMiddleware,
    secret_key=os.environ.get("SESSION_SECRET", "change-me"),
    same_site="lax",
    https_only=True,
)

app.include_router(ui_router)


@app.get("/__debug")
def __debug(request: Request):
    return {
        "host": request.headers.get("host"),
        "x_forwarded_host": request.headers.get("x-forwarded-host"),
        "x_forwarded_proto": request.headers.get("x-forwarded-proto"),
        "render_git_commit": os.environ.get("RENDER_GIT_COMMIT"),
        "marker": "MCP-SERVER-DEBUG-2026-02-25",
    }

@app.get("/")
def root():
    return {"ok": True, "service": "QBO MCP Server (OAuth + UI)", "ui": "/ui", "mcp": "/mcp"}


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/intuit/connect")
def intuit_connect(state: str):
    return RedirectResponse(build_intuit_auth_url(state=state))


@app.get("/intuit/callback")
async def intuit_callback(code: str, realmId: str, state: str):
    token_resp = await exchange_code_for_tokens(code)
    access_token = token_resp["access_token"]
    refresh_token = token_resp["refresh_token"]
    expires_in = int(token_resp.get("expires_in", 3600))
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
    user_id = state

    await db.upsert_connection(
        user_id=user_id,
        realm_id=realmId,
        company_name=None,
        access_token_enc=encrypt(access_token),
        refresh_token_enc=encrypt(refresh_token),
        access_token_expires_at=expires_at,
    )
    return JSONResponse({"connected": True, "realmId": realmId, "user_id": user_id})


# ---------------------------------------------------------------------------
# OAuth discovery endpoints (required for ChatGPT Custom Connector OAuth mode)
# ---------------------------------------------------------------------------


def _normalized_issuer_from_env() -> Optional[str]:
    """Return issuer (with trailing slash if it's a domain-based issuer).

    Auth0 issuers typically include a trailing slash.
    """
    issuer = os.environ.get("OIDC_ISSUER")
    if issuer:
        # Preserve Auth0/OIDC conventions, but ensure a single trailing slash.
        return issuer.rstrip("/") + "/"

    dom = os.environ.get("OAUTH_ISSUER_DOMAIN")
    if dom:
        return f"https://{dom.rstrip('/')}/"

    return None


def _authorization_endpoint() -> Optional[str]:
    ep = os.environ.get("OIDC_AUTHORIZATION_ENDPOINT")
    if ep:
        return ep
    dom = os.environ.get("OAUTH_ISSUER_DOMAIN")
    if dom:
        return f"https://{dom.rstrip('/')}/authorize"
    return None


def _token_endpoint() -> Optional[str]:
    ep = os.environ.get("OIDC_TOKEN_ENDPOINT")
    if ep:
        return ep
    dom = os.environ.get("OAUTH_ISSUER_DOMAIN")
    if dom:
        return f"https://{dom.rstrip('/')}/oauth/token"
    return None


def _jwks_uri() -> Optional[str]:
    uri = os.environ.get("OIDC_JWKS_URI")
    if uri:
        return uri
    dom = os.environ.get("OAUTH_ISSUER_DOMAIN")
    if dom:
        return f"https://{dom.rstrip('/')}/.well-known/jwks.json"
    return None


def _registration_endpoint() -> Optional[str]:
    # If your IdP supports OIDC Dynamic Client Registration (DCR), set this.
    # Auth0's DCR endpoint is typically /oidc/register when enabled.
    ep = os.environ.get("OIDC_REGISTRATION_ENDPOINT")
    if ep:
        return ep
    dom = os.environ.get("OAUTH_ISSUER_DOMAIN")
    if dom:
        return f"https://{dom.rstrip('/')}/oidc/register"
    return None


def _supported_scopes() -> list[str]:
    # ChatGPT's OAuth client can be strict about requested vs. granted scopes.
    # Prefer standard OIDC scopes by default unless you explicitly configure
    # custom scopes in your IdP.
    raw = os.environ.get("OAUTH_SCOPES", "openid profile email offline_access")
    return [s for s in raw.replace(",", " ").split() if s]


def _public_base_url_from_request(request: Request) -> str:
    # Prefer explicit env for stability behind proxies.
    env_url = os.environ.get("PUBLIC_BASE_URL") or os.environ.get("BASE_URL") or os.environ.get("RENDER_EXTERNAL_URL")
    if env_url:
        return env_url.rstrip("/")

    # Fall back to request-derived.
    host = request.headers.get("x-forwarded-host") or request.headers.get("host")
    proto = request.headers.get("x-forwarded-proto") or request.url.scheme
    if host:
        return f"{proto}://{host}".rstrip("/")

    return ""


def _resource_url(request: Request) -> str:
    """Return the resource identifier (RFC 9728) for this MCP server.

    IMPORTANT: The `resource` MUST match the MCP server URL that the client
    configured (e.g. https://host/mcp). Many MCP clients validate this.
    """

    # If explicitly configured, trust the env value.
    resource = os.environ.get("OAUTH_RESOURCE")
    if resource:
        return resource.rstrip("/")

    base = _public_base_url_from_request(request).rstrip("/")
    # Our MCP endpoint is /mcp (served by the FastMCP HTTP transport).
    return f"{base}/mcp" if base else "/mcp"


@app.get("/.well-known/oauth-protected-resource")
@app.get("/.well-known/oauth-protected-resource/mcp")
def oauth_protected_resource(request: Request):
    """Protected Resource Metadata (RFC 9728).

    ChatGPT uses this to discover your authorization server(s) and scopes.
    """
    issuer = _normalized_issuer_from_env()

    return {
        "resource": _resource_url(request),
        "authorization_servers": [issuer] if issuer else [],
        "scopes_supported": _supported_scopes(),
        "bearer_methods_supported": ["header"],
        "resource_documentation": os.environ.get("RESOURCE_DOCUMENTATION"),
    }


@app.get("/.well-known/oauth-authorization-server")
def oauth_authorization_server():
    """OAuth Authorization Server Metadata (RFC 8414).

    Note: In a typical deployment, this lives on the authorization server domain.
    We expose it here as well because some MCP clients expect it on the same host.
    """
    return {
        "issuer": _normalized_issuer_from_env(),
        "authorization_endpoint": _authorization_endpoint(),
        "token_endpoint": _token_endpoint(),
        "registration_endpoint": _registration_endpoint(),
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code", "refresh_token"],
        "code_challenge_methods_supported": ["S256"],
        "scopes_supported": _supported_scopes(),
        "token_endpoint_auth_methods_supported": [
            # Common options; the real supported methods depend on your IdP.
            "client_secret_post",
            "client_secret_basic",
            "none",
        ],
    }


@app.get("/.well-known/openid-configuration")
def openid_configuration():
    return {
        "issuer": _normalized_issuer_from_env(),
        "authorization_endpoint": _authorization_endpoint(),
        "token_endpoint": _token_endpoint(),
        "registration_endpoint": _registration_endpoint(),
        "jwks_uri": _jwks_uri(),
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code", "refresh_token"],
        "subject_types_supported": ["public"],
        "id_token_signing_alg_values_supported": ["RS256"],
        "scopes_supported": ["openid", "profile", "email", "offline_access"],
    }


# ---------------------------------------------------------------------------
# MCP mount + OAuth enforcement (HTTP-level)
# ---------------------------------------------------------------------------


class MCPHttpOAuthWrapper:
    """ASGI wrapper that enforces OAuth for MCP requests.

    Key behavior for ChatGPT:
    - On 401, include `WWW-Authenticate: Bearer resource_metadata="..."` so the
      client can discover the protected resource metadata endpoint.

    See OpenAI Apps SDK docs for the expected challenge format.
    """

    def __init__(self, asgi_app: Any):
        self._app = asgi_app

    @staticmethod
    def _extract_jsonrpc_methods(body: bytes) -> list[str]:
        """Best-effort extraction of JSON-RPC method(s) from an MCP request body.

        Handles:
        - single JSON-RPC request objects
        - JSON-RPC batch requests (list of objects)
        - notifications (no "id")
        """
        if not body:
            return []
        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception:
            return []

        msgs = payload if isinstance(payload, list) else [payload]
        methods: list[str] = []
        for msg in msgs:
            if isinstance(msg, dict):
                m = msg.get("method")
                if m:
                    methods.append(str(m))
        return methods

    class _BodyBuffer:
        """Buffer the ASGI body so it can be read and then replayed."""

        def __init__(self, receive):
            self._receive = receive
            self._body: Optional[bytes] = None
            self._queue: Optional[list] = None

        async def body(self) -> bytes:
            if self._body is not None:
                return self._body
            chunks: list[bytes] = []
            more = True
            while more:
                message = await self._receive()
                if message.get("type") != "http.request":
                    continue
                chunks.append(message.get("body", b""))
                more = bool(message.get("more_body", False))
            self._body = b"".join(chunks)
            self._queue = [{"type": "http.request", "body": self._body, "more_body": False}]
            return self._body

        async def replay(self):
            # If we never read the body, just pass through.
            if self._queue is None:
                return await self._receive()
            if self._queue:
                return self._queue.pop(0)
            return {"type": "http.request", "body": b"", "more_body": False}

    def _challenge_headers(self, scope: Dict[str, Any]) -> Dict[str, str]:
        # Build an absolute resource metadata URL when possible.
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in (scope.get("headers") or [])}
        proto = headers.get("x-forwarded-proto") or scope.get("scheme") or "https"
        host = headers.get("x-forwarded-host") or headers.get("host")
        base = os.environ.get("PUBLIC_BASE_URL") or os.environ.get("BASE_URL") or os.environ.get("RENDER_EXTERNAL_URL")
        if base:
            base = base.rstrip("/")
        elif host:
            base = f"{proto}://{host}".rstrip("/")
        else:
            base = ""

        # Prefer root metadata URL. We also serve a path-qualified variant at
        # /.well-known/oauth-protected-resource/mcp for compatibility.
        resource_metadata_url = (
            f"{base}/.well-known/oauth-protected-resource" if base else "/.well-known/oauth-protected-resource"
        )
        scope_str = " ".join(_supported_scopes())

        www_auth = f'Bearer resource_metadata="{resource_metadata_url}"'
        if scope_str:
            www_auth += f', scope="{scope_str}"'

        return {"WWW-Authenticate": www_auth}

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        body_buf = self._BodyBuffer(receive)

        path = (scope.get("path") or "").rstrip("/")

        # Some clients probe MCP with GET /mcp. Return a simple 200 instead of forcing OAuth.
        if scope.get("method") == "GET" and (path == "" or path == "/"):
            resp = JSONResponse({"ok": True, "message": "MCP endpoint. Use POST for JSON-RPC."})
            await resp(scope, receive, send)
            return

        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in (scope.get("headers") or [])}
        auth = headers.get("authorization")

        # Some hosts fetch MCP metadata / tool lists for UI display without attaching an
        # access token. Tool discovery is safe to expose publicly.
        if allow_public_discovery and (not auth or not auth.lower().startswith("bearer ")):
            try:
                methods = self._extract_jsonrpc_methods(await body_buf.body())

                # Only allow a small, safe subset of MCP methods without OAuth.
                # IMPORTANT: Handle JSON-RPC batches safely (every method in the
                # batch must be allow-listed) to avoid auth bypass.
                public_methods = {"initialize", "tools/list", "resources/list", "prompts/list", "logging/setLevel", "ping"}

                def _is_public_method(m: str) -> bool:
                    return m in public_methods or m.startswith("notifications/")

                if methods and all(_is_public_method(m) for m in methods):
                    await self._app(scope, body_buf.replay, send)
                    return
            except Exception:
                # Fall back to normal auth enforcement.
                pass

        # The token's audience/resource MUST match the MCP URL the client uses.
        # Compute the expected resource from the request headers.
        expected_resource = None
        try:
            # Reuse logic from _challenge_headers.
            hdrs = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in (scope.get("headers") or [])}
            proto = hdrs.get("x-forwarded-proto") or scope.get("scheme") or "https"
            host = hdrs.get("x-forwarded-host") or hdrs.get("host")
            base = os.environ.get("PUBLIC_BASE_URL") or os.environ.get("BASE_URL") or os.environ.get("RENDER_EXTERNAL_URL")
            if base:
                base = base.rstrip("/")
            elif host:
                base = f"{proto}://{host}".rstrip("/")
            else:
                base = ""
            expected_resource = os.environ.get("OAUTH_RESOURCE") or (f"{base}/mcp" if base else None)
        except Exception:
            expected_resource = os.environ.get("OAUTH_RESOURCE")

        try:
            claims = await verify_bearer_token(auth, audience=expected_resource)
        except PermissionError as e:
            logger.info("MCP auth rejected: path=%s expected_audience=%s reason=%s", path, expected_resource, str(e))
            resp = JSONResponse(
                {"error": "unauthorized", "error_description": str(e)},
                status_code=401,
                headers=self._challenge_headers(scope),
            )
            await resp(scope, receive, send)
            return
        except Exception as e:
            logger.exception("Unexpected OAuth verification error")
            resp = JSONResponse({"error": f"OAuth verify error: {e}"}, status_code=500)
            await resp(scope, receive, send)
            return

        # Attach user context for tool handlers.
        token = current_user.set({"sub": claims.get("sub"), "email": claims.get("email")})
        try:
            await self._app(scope, body_buf.replay, send)
        finally:
            # Avoid leaking identity across requests.
            current_user.reset(token)


# ---------------------------------------------------------------------------
# MCP transport mounting
#
# IMPORTANT: FastMCP's HTTP transport already exposes the /mcp (and /sse) paths.
# Mount it at the root and let FastAPI's explicit routes above take precedence.
# This avoids /mcp -> /mcp/ redirects and "double /mcp" path issues.
# ---------------------------------------------------------------------------

app.mount("/mcp", MCPHttpOAuthWrapper(mcp.streamable_http_app()))
