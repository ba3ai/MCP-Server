from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from app import db
from app.crypto import decrypt, encrypt
from app.qbo import (
    refresh_access_token,
    qbo_query,
    qbo_create_entity,
    qbo_get_entity,
    qbo_update_entity,
    qbo_operation,
    qbo_send_transaction,
    qbo_get_report,
)


async def _resolve_realm_id(user_id: str, realm_id: Optional[str]) -> str:
    """Resolve an optional realm_id to a concrete realm_id.

    If realm_id is None, use the most recently updated connection.
    """
    if realm_id:
        return realm_id

    companies = await db.list_connections(user_id)
    if not companies:
        raise ValueError(
            "No QuickBooks companies connected for this user. "
            "Run the connect tool first (qbo_connect_company) and complete the Intuit OAuth flow."
        )
    return companies[0]["realm_id"]


async def _get_valid_access_token(user_id: str, realm_id: str) -> str:
    conn = await db.get_connection(user_id, realm_id)
    access_enc = conn.get("access_token_enc")
    refresh_enc = conn["refresh_token_enc"]
    exp = conn.get("access_token_expires_at")

    access_token = decrypt(access_enc) if access_enc else None
    refresh_token = decrypt(refresh_enc)

    # Refresh if missing or expiring soon
    if (not access_token) or (not exp) or (exp <= datetime.now(timezone.utc) + timedelta(seconds=30)):
        token_resp = await refresh_access_token(refresh_token)
        access_token = token_resp["access_token"]
        new_refresh = token_resp.get("refresh_token", refresh_token)
        expires_in = int(token_resp.get("expires_in", 3600))
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
        await db.upsert_connection(
            user_id=user_id,
            realm_id=realm_id,
            company_name=conn.get("company_name"),
            access_token_enc=encrypt(access_token),
            refresh_token_enc=encrypt(new_refresh),
            access_token_expires_at=expires_at,
        )
    return access_token


# ----------------------
# Query helpers
# ----------------------


async def query_company(user_id: str, realm_id: str, sql: str, *, sandbox: Optional[bool] = None) -> Dict[str, Any]:
    token = await _get_valid_access_token(user_id, realm_id)
    data = await qbo_query(realm_id, token, sql, sandbox=sandbox)
    return {"realm_id": realm_id, "data": data}


async def query_all(user_id: str, sql: str, limit_per_company: int = 20, *, sandbox: Optional[bool] = None) -> Dict[str, Any]:
    companies = await db.list_connections(user_id)
    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for c in companies:
        realm_id = c["realm_id"]
        try:
            token = await _get_valid_access_token(user_id, realm_id)
            data = await qbo_query(realm_id, token, sql, sandbox=sandbox)
            results.append({"realm_id": realm_id, "company_name": c.get("company_name"), "data": data})
        except Exception as e:
            errors.append({"realm_id": realm_id, "error": str(e)})
    return {"sql": sql, "limit_per_company": limit_per_company, "results": results, "errors": errors}


# ----------------------
# CRUD / Reports helpers
# ----------------------


async def create_entity(
    user_id: str,
    realm_id: Optional[str],
    *,
    entity: str,
    payload: Dict[str, Any],
    sandbox: Optional[bool] = None,
) -> Dict[str, Any]:
    rid = await _resolve_realm_id(user_id, realm_id)
    token = await _get_valid_access_token(user_id, rid)
    data = await qbo_create_entity(rid, token, entity, payload, sandbox=sandbox)
    return {"realm_id": rid, "entity": entity, "data": data}


async def get_entity(
    user_id: str,
    realm_id: Optional[str],
    *,
    entity: str,
    entity_id: str,
    sandbox: Optional[bool] = None,
) -> Dict[str, Any]:
    rid = await _resolve_realm_id(user_id, realm_id)
    token = await _get_valid_access_token(user_id, rid)
    data = await qbo_get_entity(rid, token, entity, entity_id, sandbox=sandbox)
    return {"realm_id": rid, "entity": entity, "id": entity_id, "data": data}


async def update_entity(
    user_id: str,
    realm_id: Optional[str],
    *,
    entity: str,
    payload: Dict[str, Any],
    sparse: bool = False,
    sandbox: Optional[bool] = None,
) -> Dict[str, Any]:
    rid = await _resolve_realm_id(user_id, realm_id)
    token = await _get_valid_access_token(user_id, rid)
    data = await qbo_update_entity(rid, token, entity, payload, sparse=sparse, sandbox=sandbox)
    return {"realm_id": rid, "entity": entity, "sparse": sparse, "data": data}


async def operate_entity(
    user_id: str,
    realm_id: Optional[str],
    *,
    entity: str,
    operation: str,
    payload: Dict[str, Any],
    sandbox: Optional[bool] = None,
) -> Dict[str, Any]:
    rid = await _resolve_realm_id(user_id, realm_id)
    token = await _get_valid_access_token(user_id, rid)
    data = await qbo_operation(rid, token, entity, operation, payload, sandbox=sandbox)
    return {"realm_id": rid, "entity": entity, "operation": operation, "data": data}


async def send_transaction(
    user_id: str,
    realm_id: Optional[str],
    *,
    entity: str,
    entity_id: str,
    send_to: Optional[str] = None,
    sandbox: Optional[bool] = None,
) -> Dict[str, Any]:
    rid = await _resolve_realm_id(user_id, realm_id)
    token = await _get_valid_access_token(user_id, rid)
    data = await qbo_send_transaction(rid, token, entity, entity_id, send_to=send_to, sandbox=sandbox)
    return {"realm_id": rid, "entity": entity, "id": entity_id, "send_to": send_to, "data": data}


async def get_report(
    user_id: str,
    realm_id: Optional[str],
    *,
    report_name: str,
    params: Optional[Dict[str, Any]] = None,
    sandbox: Optional[bool] = None,
) -> Dict[str, Any]:
    rid = await _resolve_realm_id(user_id, realm_id)
    token = await _get_valid_access_token(user_id, rid)
    data = await qbo_get_report(rid, token, report_name, params=params or {}, sandbox=sandbox)
    return {"realm_id": rid, "report": report_name, "params": params or {}, "data": data}


def _build_select_sql(
    entity: str,
    *,
    where: Optional[str] = None,
    start_position: int = 1,
    max_results: int = 50,
    order_by: Optional[str] = None,
) -> str:
    """Build a simple IQL SELECT statement."""
    sql = f"SELECT * FROM {entity}"
    if where:
        sql += f" WHERE {where}"
    if order_by:
        sql += f" ORDERBY {order_by}"
    sql += f" STARTPOSITION {int(start_position)} MAXRESULTS {int(max_results)}"
    return sql


async def search_entity(
    user_id: str,
    realm_id: Optional[str],
    *,
    entity: str,
    where: Optional[str] = None,
    start_position: int = 1,
    max_results: int = 50,
    order_by: Optional[str] = None,
    sandbox: Optional[bool] = None,
) -> Dict[str, Any]:
    sql = _build_select_sql(
        entity,
        where=where,
        start_position=start_position,
        max_results=max_results,
        order_by=order_by,
    )
    return await query_company(user_id, await _resolve_realm_id(user_id, realm_id), sql, sandbox=sandbox)
