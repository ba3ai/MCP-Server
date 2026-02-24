from __future__ import annotations

import os
from typing import Any, Dict, Optional

import httpx
from dotenv import load_dotenv
from jose import jwt
from jose.exceptions import JWTError

load_dotenv()

# Auth0-style issuer is usually the tenant domain (with a trailing slash).
ISSUER_DOMAIN = os.environ.get("OAUTH_ISSUER_DOMAIN")
ISSUER = f"https://{ISSUER_DOMAIN.rstrip('/')}/" if ISSUER_DOMAIN else None

# Fallback audience. In production, pass the *expected* audience from the
# request context (see main.py) so it matches the MCP URL that the client uses.
DEFAULT_AUDIENCE = os.environ.get("OAUTH_AUDIENCE") or os.environ.get("OAUTH_RESOURCE")

ALGORITHMS = [a.strip() for a in os.environ.get("OAUTH_ALGORITHMS", "RS256").split(",") if a.strip()]

_jwks_cache: Optional[Dict[str, Any]] = None


async def _get_jwks() -> Dict[str, Any]:
    global _jwks_cache
    if _jwks_cache:
        return _jwks_cache
    if not ISSUER:
        raise RuntimeError("OAUTH_ISSUER_DOMAIN is not set")

    url = f"{ISSUER}.well-known/jwks.json"
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url)
        r.raise_for_status()
        _jwks_cache = r.json()
        return _jwks_cache


def _norm_url(u: str) -> str:
    # Normalize only the trailing slash to avoid common audience mismatches.
    return u.rstrip("/")


async def verify_bearer_token(auth_header: Optional[str], *, audience: Optional[str] = None) -> Dict[str, Any]:
    """Verify an Authorization header containing a Bearer JWT.

    Important: MCP clients validate that the token is intended for the MCP
    server (audience / resource indicator). For ChatGPT connectors this often
    needs to match the *exact* MCP URL (e.g. https://host/mcp).
    """

    expected_audience = audience or DEFAULT_AUDIENCE
    if not expected_audience:
        raise RuntimeError("OAUTH_AUDIENCE (or OAUTH_RESOURCE) is not set")

    if not auth_header or not auth_header.lower().startswith("bearer "):
        raise PermissionError("Missing Bearer token")

    token = auth_header.split(" ", 1)[1].strip()

    # Fast-path: Auth0 may return opaque access tokens (or JWE tokens) when the
    # request doesn't indicate a resource/audience. MCP clients (including
    # ChatGPT) generally expect a signed JWT access token for resource servers.
    # JWTs have exactly 2 '.' separators.
    if token.count(".") != 2:
        raise PermissionError(
            "Access token does not look like a signed JWT (it may be opaque or JWE-encrypted). "
            "If you're using Auth0, create an API with identifier equal to your MCP resource URL "
            "(e.g. https://<host>/mcp) and enable Auth0's Resource Parameter Compatibility Profile (Auth for MCP), "
            "or set a Default Audience so Auth0 returns a JWT access token."
        )

    jwks = await _get_jwks()

    try:
        unverified_header = jwt.get_unverified_header(token)
    except JWTError as e:
        # Common failure if Auth0 returns opaque tokens or encrypted JWE tokens.
        raise PermissionError(
            "Access token is not a JWT. If using Auth0, ensure your API identifier matches your MCP resource URL "
            "and enable the Resource Parameter Compatibility Profile (Auth for MCP), or set a Default Audience "
            "(so Auth0 returns a signed JWT access token instead of an opaque/JWE token)."
        ) from e

    kid = unverified_header.get("kid")
    try:
        key = next(k for k in jwks.get("keys", []) if k.get("kid") == kid)
    except StopIteration as e:
        raise PermissionError("Invalid token: unknown key id (kid)") from e

    # Decode with signature + issuer validation, but do audience validation
    # manually so we can tolerate minor URL formatting differences.
    try:
        claims = jwt.decode(
            token,
            key,
            algorithms=ALGORITHMS,
            issuer=ISSUER,
            options={"verify_aud": False},
        )
    except JWTError as e:
        raise PermissionError(f"Invalid token: {e}") from e

    token_aud = claims.get("aud")
    aud_list = token_aud if isinstance(token_aud, list) else [token_aud]
    aud_norms = {_norm_url(a) for a in aud_list if isinstance(a, str)}
    if _norm_url(expected_audience) not in aud_norms:
        raise PermissionError(
            f"Invalid token audience. Expected '{expected_audience}' (normalized '{_norm_url(expected_audience)}'), "
            f"got {token_aud!r}"
        )

    return claims
