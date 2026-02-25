import os
import base64
from urllib.parse import urlencode
from typing import Any, Dict, Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

# ---------------------------
# Intuit OAuth (for QBO)
# ---------------------------

def _token_url() -> str:
    return "https://oauth.platform.intuit.com/oauth2/v1/tokens/bearer"


def _auth_base_url() -> str:
    return "https://appcenter.intuit.com/connect/oauth2"


def _basic_auth_header() -> str:
    cid = os.environ["INTUIT_CLIENT_ID"]
    sec = os.environ["INTUIT_CLIENT_SECRET"]
    token = base64.b64encode(f"{cid}:{sec}".encode()).decode()
    return f"Basic {token}"


def build_intuit_auth_url(state: str) -> str:
    """Return the user-facing Intuit OAuth connect URL."""
    client_id = os.environ["INTUIT_CLIENT_ID"]
    redirect_uri = os.environ["INTUIT_REDIRECT_URI"]
    scope = os.environ.get("INTUIT_SCOPE", "com.intuit.quickbooks.accounting")
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": scope,
        "state": state,
    }
    return f"{_auth_base_url()}?{urlencode(params)}"


async def exchange_code_for_tokens(code: str) -> dict:
    """Exchange an Intuit authorization code for tokens."""
    redirect_uri = os.environ["INTUIT_REDIRECT_URI"]
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            _token_url(),
            headers={"Authorization": _basic_auth_header(), "Accept": "application/json"},
            data={"grant_type": "authorization_code", "code": code, "redirect_uri": redirect_uri},
        )
        resp.raise_for_status()
        return resp.json()


async def refresh_access_token(refresh_token: str) -> dict:
    """Refresh an Intuit QBO access token."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            _token_url(),
            headers={"Authorization": _basic_auth_header(), "Accept": "application/json"},
            data={"grant_type": "refresh_token", "refresh_token": refresh_token},
        )
        resp.raise_for_status()
        return resp.json()


# ---------------------------
# QBO Accounting API helpers
# ---------------------------

def _minorversion() -> str:
    # Intuit minorversion changes over time; keep configurable.
    return os.environ.get("QBO_MINORVERSION", "75")


def _qbo_env_is_sandbox() -> bool:
    env = (os.environ.get("QBO_ENV") or os.environ.get("INTUIT_ENVIRONMENT") or "production").lower()
    return env in ("sandbox", "development", "dev", "test")


def _qbo_api_base_url(*, sandbox: Optional[bool] = None) -> str:
    """Return the QBO API base URL.

    Intuit uses a different hostname for sandbox vs production.
    """
    use_sandbox = _qbo_env_is_sandbox() if sandbox is None else sandbox
    return "https://sandbox-quickbooks.api.intuit.com" if use_sandbox else "https://quickbooks.api.intuit.com"


async def qbo_request(
    method: str,
    *,
    realm_id: str,
    access_token: str,
    path: str,
    params: Optional[Dict[str, Any]] = None,
    json_body: Optional[Dict[str, Any]] = None,
    sandbox: Optional[bool] = None,
) -> Dict[str, Any]:
    """Low-level QBO request helper.

    Args:
        method: HTTP method
        realm_id: QBO company realm ID
        access_token: Intuit access token
        path: Path under /v3/company/{realm_id}, e.g. '/invoice'
        params: Query string parameters
        json_body: JSON request body (for POST)
        sandbox: Force sandbox/prod hostname override.
    """
    base = _qbo_api_base_url(sandbox=sandbox)
    if not path.startswith("/"):
        path = "/" + path

    url = f"{base}/v3/company/{realm_id}{path}"

    qparams: Dict[str, Any] = dict(params or {})
    # Minorversion is generally safe for most endpoints.
    qparams.setdefault("minorversion", _minorversion())

    headers: Dict[str, str] = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
    }
    if method.upper() in ("POST", "PUT", "PATCH"):
        headers["Content-Type"] = "application/json"

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.request(method.upper(), url, headers=headers, params=qparams, json=json_body)

    # Helpful error payloads for debugging
    if resp.status_code >= 400:
        try:
            err = resp.json()
        except Exception:
            err = resp.text
        raise httpx.HTTPStatusError(
            f"QBO API error {resp.status_code} for {method.upper()} {url}: {err}",
            request=resp.request,
            response=resp,
        )

    if resp.status_code == 204:
        return {"ok": True, "status_code": 204}

    ctype = (resp.headers.get("content-type") or "").lower()
    if "application/json" in ctype:
        return resp.json()

    return {"ok": True, "status_code": resp.status_code, "content_type": ctype, "text": resp.text}


async def qbo_query(realm_id: str, access_token: str, sql: str, *, sandbox: Optional[bool] = None) -> dict:
    """Run an Intuit Query Language (IQL) SQL-like query."""
    return await qbo_request(
        "GET",
        realm_id=realm_id,
        access_token=access_token,
        path="/query",
        params={"query": sql},
        sandbox=sandbox,
    )


async def qbo_create_entity(
    realm_id: str,
    access_token: str,
    entity: str,
    payload: Dict[str, Any],
    *,
    sandbox: Optional[bool] = None,
) -> Dict[str, Any]:
    return await qbo_request(
        "POST",
        realm_id=realm_id,
        access_token=access_token,
        path=f"/{entity}",
        json_body=payload,
        sandbox=sandbox,
    )


async def qbo_get_entity(
    realm_id: str,
    access_token: str,
    entity: str,
    entity_id: str,
    *,
    sandbox: Optional[bool] = None,
) -> Dict[str, Any]:
    return await qbo_request(
        "GET",
        realm_id=realm_id,
        access_token=access_token,
        path=f"/{entity}/{entity_id}",
        sandbox=sandbox,
    )


async def qbo_update_entity(
    realm_id: str,
    access_token: str,
    entity: str,
    payload: Dict[str, Any],
    *,
    sparse: bool = False,
    sandbox: Optional[bool] = None,
) -> Dict[str, Any]:
    body = dict(payload)
    if sparse:
        body.setdefault("sparse", True)
    return await qbo_request(
        "POST",
        realm_id=realm_id,
        access_token=access_token,
        path=f"/{entity}",
        json_body=body,
        sandbox=sandbox,
    )


async def qbo_operation(
    realm_id: str,
    access_token: str,
    entity: str,
    operation: str,
    payload: Dict[str, Any],
    *,
    sandbox: Optional[bool] = None,
) -> Dict[str, Any]:
    """POST /{entity}?operation={operation}"""
    return await qbo_request(
        "POST",
        realm_id=realm_id,
        access_token=access_token,
        path=f"/{entity}",
        params={"operation": operation},
        json_body=payload,
        sandbox=sandbox,
    )


async def qbo_send_transaction(
    realm_id: str,
    access_token: str,
    entity: str,
    entity_id: str,
    *,
    send_to: Optional[str] = None,
    sandbox: Optional[bool] = None,
) -> Dict[str, Any]:
    """Send a transaction (Invoice/Estimate) via QBO email endpoint."""
    params: Dict[str, Any] = {}
    if send_to:
        params["sendTo"] = send_to
    return await qbo_request(
        "POST",
        realm_id=realm_id,
        access_token=access_token,
        path=f"/{entity}/{entity_id}/send",
        params=params,
        sandbox=sandbox,
    )


async def qbo_get_report(
    realm_id: str,
    access_token: str,
    report_name: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    sandbox: Optional[bool] = None,
) -> Dict[str, Any]:
    """Fetch a QBO report."""
    return await qbo_request(
        "GET",
        realm_id=realm_id,
        access_token=access_token,
        path=f"/reports/{report_name}",
        params=params or {},
        sandbox=sandbox,
    )
