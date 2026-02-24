from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta
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

load_dotenv()

app = FastAPI()

app.add_middleware(
    SessionMiddleware,
    secret_key=os.environ.get("SESSION_SECRET", "change-me"),
    same_site="lax",
    https_only=True,
)

app.include_router(ui_router)


@app.on_event("startup")
async def _startup() -> None:
    await init_db()


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
    # MCP examples often use read/write. Adjust to your needs.
    raw = os.environ.get("OAUTH_SCOPES", "read write")
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


@app.get("/.well-known/oauth-protected-resource")
def oauth_protected_resource(request: Request):
    """Protected Resource Metadata (RFC 9728).

    ChatGPT uses this to discover your authorization server(s) and scopes.
    """
    resource = os.environ.get("OAUTH_RESOURCE")
    if not resource:
        resource = _public_base_url_from_request(request)

    issuer = _normalized_issuer_from_env()

    return {
        "resource": resource,
        "authorization_servers": [issuer] if issuer else [],
        "scopes_supported": _supported_scopes(),
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

        resource_metadata_url = f"{base}/.well-known/oauth-protected-resource" if base else "/.well-known/oauth-protected-resource"
        scope_str = " ".join(_supported_scopes())

        www_auth = f'Bearer resource_metadata="{resource_metadata_url}"'
        if scope_str:
            www_auth += f', scope="{scope_str}"'

        return {"WWW-Authenticate": www_auth}

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        method = (scope.get("method") or "").upper()

        # Some clients probe with GET/HEAD.
        if method in ("GET", "HEAD"):
            resp = JSONResponse(
                {"ok": True, "service": "QBO MCP Server (OAuth + UI)", "auth": "oauth", "ui": "/ui"},
                status_code=200,
            )
            await resp(scope, receive, send)
            return

        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in (scope.get("headers") or [])}
        auth = headers.get("authorization")

        try:
            claims = await verify_bearer_token(auth)
        except PermissionError as e:
            resp = JSONResponse(
                {"error": "unauthorized", "error_description": str(e)},
                status_code=401,
                headers=self._challenge_headers(scope),
            )
            await resp(scope, receive, send)
            return
        except Exception as e:
            resp = JSONResponse({"error": f"OAuth verify error: {e}"}, status_code=500)
            await resp(scope, receive, send)
            return

        # Attach user context for tool handlers.
        current_user.set({"sub": claims.get("sub"), "email": claims.get("email")})
        await self._app(scope, receive, send)


app.mount("/mcp", MCPHttpOAuthWrapper(mcp.streamable_http_app()))
