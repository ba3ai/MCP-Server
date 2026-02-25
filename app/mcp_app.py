from __future__ import annotations

from typing import Any, Dict, Optional

from mcp.server.fastmcp import FastMCP

from app.request_context import current_user
from app.qbo import build_intuit_auth_url
from app import db
from app.service import (
    query_company,
    query_all,
    create_entity,
    get_entity,
    update_entity,
    operate_entity,
    send_transaction,
    get_report,
    search_entity,
)
from dotenv import load_dotenv

load_dotenv()

# NOTE: ChatGPT Apps / Custom Connectors require *stateless* HTTP mode for the
# HTTP/SSE transport expected by the client.
mcp = FastMCP("QBO MCP Server (OAuth + UI)", stateless_http=True)


def _user_id() -> str:
    u = current_user.get() or {}
    return (u.get("email") or u.get("sub") or "unknown_user").strip()


# ----------------------
# Existing helper tools
# ----------------------


@mcp.tool(description="Return an Intuit OAuth connect URL for the current user. Open it to connect another QBO company.")
async def qbo_connect_company() -> Dict[str, Any]:
    uid = _user_id()
    return {"user_id": uid, "connect_url": build_intuit_auth_url(state=uid)}


@mcp.tool(description="List all connected QBO companies for the current user.")
async def qbo_list_companies() -> Dict[str, Any]:
    uid = _user_id()
    return {"user_id": uid, "companies": await db.list_connections(uid)}


@mcp.tool(description="Run a QBO Query (SQL-like) for a specific company (realm_id).")
async def qbo_query_company(realm_id: str, sql: str) -> Dict[str, Any]:
    uid = _user_id()
    return await query_company(uid, realm_id, sql)


@mcp.tool(description="Run a QBO Query (SQL-like) across all connected companies.")
async def qbo_query_all(sql: str, limit_per_company: int = 20) -> Dict[str, Any]:
    uid = _user_id()
    return await query_all(uid, sql, limit_per_company)


# ----------------------
# QuickBooks action tools (ChatGPT Custom Connector)
# ----------------------
# NOTE: These tool names are intentionally hyphenated to match expected action IDs.


# --- Reports ---


@mcp.tool(
    name="quickbooks-create-ap-aging-report",
    description="Creates an AP aging report in QuickBooks Online (Report: APAgingDetail).",
    annotations={"readOnlyHint": True},
)
async def quickbooks_create_ap_aging_report(
    realm_id: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    uid = _user_id()
    return await get_report(uid, realm_id, report_name="APAgingDetail", params=params or {})


@mcp.tool(
    name="quickbooks-create-pl-report",
    description="Creates a Profit and Loss report in QuickBooks Online (Report: ProfitAndLoss).",
    annotations={"readOnlyHint": True},
)
async def quickbooks_create_pl_report(
    realm_id: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    uid = _user_id()
    return await get_report(uid, realm_id, report_name="ProfitAndLoss", params=params or {})


@mcp.tool(
    name="quickbooks-get-balance-sheet-report",
    description="Retrieves the Balance Sheet report from QuickBooks Online (Report: BalanceSheet).",
    annotations={"readOnlyHint": True},
)
async def quickbooks_get_balance_sheet_report(
    realm_id: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    uid = _user_id()
    return await get_report(uid, realm_id, report_name="BalanceSheet", params=params or {})


@mcp.tool(
    name="quickbooks-get-cash-flow-report",
    description="Retrieves the Cash Flow report from QuickBooks Online (Report: CashFlow).",
    annotations={"readOnlyHint": True},
)
async def quickbooks_get_cash_flow_report(
    realm_id: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    uid = _user_id()
    return await get_report(uid, realm_id, report_name="CashFlow", params=params or {})


# --- Create ---


@mcp.tool(name="quickbooks-create-bill", description="Creates a bill.")
async def quickbooks_create_bill(realm_id: Optional[str] = None, bill: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    uid = _user_id()
    if bill is None:
        raise ValueError("Missing required parameter: bill")
    return await create_entity(uid, realm_id, entity="bill", payload=bill)


@mcp.tool(name="quickbooks-create-customer", description="Creates a customer.")
async def quickbooks_create_customer(
    realm_id: Optional[str] = None,
    customer: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    uid = _user_id()
    if customer is None:
        raise ValueError("Missing required parameter: customer")
    return await create_entity(uid, realm_id, entity="customer", payload=customer)


@mcp.tool(name="quickbooks-create-estimate", description="Creates an estimate.")
async def quickbooks_create_estimate(realm_id: Optional[str] = None, estimate: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    uid = _user_id()
    if estimate is None:
        raise ValueError("Missing required parameter: estimate")
    return await create_entity(uid, realm_id, entity="estimate", payload=estimate)


@mcp.tool(name="quickbooks-create-invoice", description="Creates an invoice.")
async def quickbooks_create_invoice(realm_id: Optional[str] = None, invoice: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    uid = _user_id()
    if invoice is None:
        raise ValueError("Missing required parameter: invoice")
    return await create_entity(uid, realm_id, entity="invoice", payload=invoice)


@mcp.tool(name="quickbooks-create-payment", description="Creates a payment.")
async def quickbooks_create_payment(realm_id: Optional[str] = None, payment: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    uid = _user_id()
    if payment is None:
        raise ValueError("Missing required parameter: payment")
    return await create_entity(uid, realm_id, entity="payment", payload=payment)


@mcp.tool(name="quickbooks-create-purchase", description="Creates a new purchase.")
async def quickbooks_create_purchase(realm_id: Optional[str] = None, purchase: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    uid = _user_id()
    if purchase is None:
        raise ValueError("Missing required parameter: purchase")
    return await create_entity(uid, realm_id, entity="purchase", payload=purchase)


@mcp.tool(name="quickbooks-create-purchase-order", description="Creates a purchase order.")
async def quickbooks_create_purchase_order(
    realm_id: Optional[str] = None,
    purchase_order: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    uid = _user_id()
    if purchase_order is None:
        raise ValueError("Missing required parameter: purchase_order")
    return await create_entity(uid, realm_id, entity="purchaseorder", payload=purchase_order)


@mcp.tool(name="quickbooks-create-sales-receipt", description="Creates a sales receipt.")
async def quickbooks_create_sales_receipt(
    realm_id: Optional[str] = None,
    sales_receipt: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    uid = _user_id()
    if sales_receipt is None:
        raise ValueError("Missing required parameter: sales_receipt")
    return await create_entity(uid, realm_id, entity="salesreceipt", payload=sales_receipt)


# --- Get ---


@mcp.tool(name="quickbooks-get-bill", description="Returns info about a bill.", annotations={"readOnlyHint": True})
async def quickbooks_get_bill(realm_id: Optional[str] = None, bill_id: str = "") -> Dict[str, Any]:
    uid = _user_id()
    if not bill_id:
        raise ValueError("Missing required parameter: bill_id")
    return await get_entity(uid, realm_id, entity="bill", entity_id=bill_id)


@mcp.tool(name="quickbooks-get-customer", description="Returns info about a customer.", annotations={"readOnlyHint": True})
async def quickbooks_get_customer(realm_id: Optional[str] = None, customer_id: str = "") -> Dict[str, Any]:
    uid = _user_id()
    if not customer_id:
        raise ValueError("Missing required parameter: customer_id")
    return await get_entity(uid, realm_id, entity="customer", entity_id=customer_id)


@mcp.tool(name="quickbooks-get-invoice", description="Returns info about an invoice.", annotations={"readOnlyHint": True})
async def quickbooks_get_invoice(realm_id: Optional[str] = None, invoice_id: str = "") -> Dict[str, Any]:
    uid = _user_id()
    if not invoice_id:
        raise ValueError("Missing required parameter: invoice_id")
    return await get_entity(uid, realm_id, entity="invoice", entity_id=invoice_id)


@mcp.tool(name="quickbooks-get-payment", description="Returns info about a payment.", annotations={"readOnlyHint": True})
async def quickbooks_get_payment(realm_id: Optional[str] = None, payment_id: str = "") -> Dict[str, Any]:
    uid = _user_id()
    if not payment_id:
        raise ValueError("Missing required parameter: payment_id")
    return await get_entity(uid, realm_id, entity="payment", entity_id=payment_id)


@mcp.tool(name="quickbooks-get-purchase", description="Returns info about a purchase.", annotations={"readOnlyHint": True})
async def quickbooks_get_purchase(realm_id: Optional[str] = None, purchase_id: str = "") -> Dict[str, Any]:
    uid = _user_id()
    if not purchase_id:
        raise ValueError("Missing required parameter: purchase_id")
    return await get_entity(uid, realm_id, entity="purchase", entity_id=purchase_id)


@mcp.tool(
    name="quickbooks-get-purchase-order",
    description="Returns details about a purchase order.",
    annotations={"readOnlyHint": True},
)
async def quickbooks_get_purchase_order(realm_id: Optional[str] = None, purchase_order_id: str = "") -> Dict[str, Any]:
    uid = _user_id()
    if not purchase_order_id:
        raise ValueError("Missing required parameter: purchase_order_id")
    return await get_entity(uid, realm_id, entity="purchaseorder", entity_id=purchase_order_id)


@mcp.tool(
    name="quickbooks-get-sales-receipt",
    description="Returns details about a sales receipt.",
    annotations={"readOnlyHint": True},
)
async def quickbooks_get_sales_receipt(realm_id: Optional[str] = None, sales_receipt_id: str = "") -> Dict[str, Any]:
    uid = _user_id()
    if not sales_receipt_id:
        raise ValueError("Missing required parameter: sales_receipt_id")
    return await get_entity(uid, realm_id, entity="salesreceipt", entity_id=sales_receipt_id)


@mcp.tool(
    name="quickbooks-get-time-activity",
    description="Returns info about a time activity.",
    annotations={"readOnlyHint": True},
)
async def quickbooks_get_time_activity(realm_id: Optional[str] = None, time_activity_id: str = "") -> Dict[str, Any]:
    uid = _user_id()
    if not time_activity_id:
        raise ValueError("Missing required parameter: time_activity_id")
    return await get_entity(uid, realm_id, entity="timeactivity", entity_id=time_activity_id)


@mcp.tool(
    name="quickbooks-get-my-company",
    description="Gets info about the connected company (CompanyInfo).",
    annotations={"readOnlyHint": True},
)
async def quickbooks_get_my_company(realm_id: Optional[str] = None) -> Dict[str, Any]:
    uid = _user_id()
    # CompanyInfo often has Id=1. Query is safer.
    rid = realm_id
    if not rid:
        companies = await db.list_connections(uid)
        if not companies:
            raise ValueError("No QuickBooks companies connected for this user")
        rid = companies[0]["realm_id"]
    return await query_company(uid, rid, "SELECT * FROM CompanyInfo")


# --- Update / Sparse update / Void / Delete ---


@mcp.tool(name="quickbooks-update-customer", description="Updates a customer.")
async def quickbooks_update_customer(realm_id: Optional[str] = None, customer: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    uid = _user_id()
    if customer is None:
        raise ValueError("Missing required parameter: customer")
    return await update_entity(uid, realm_id, entity="customer", payload=customer)


@mcp.tool(name="quickbooks-update-estimate", description="Updates an estimate.")
async def quickbooks_update_estimate(realm_id: Optional[str] = None, estimate: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    uid = _user_id()
    if estimate is None:
        raise ValueError("Missing required parameter: estimate")
    return await update_entity(uid, realm_id, entity="estimate", payload=estimate)


@mcp.tool(name="quickbooks-update-invoice", description="Updates an invoice.")
async def quickbooks_update_invoice(realm_id: Optional[str] = None, invoice: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    uid = _user_id()
    if invoice is None:
        raise ValueError("Missing required parameter: invoice")
    return await update_entity(uid, realm_id, entity="invoice", payload=invoice)


@mcp.tool(
    name="quickbooks-sparse-update-invoice",
    description="Sparse update an invoice (only provided fields are changed).",
)
async def quickbooks_sparse_update_invoice(realm_id: Optional[str] = None, invoice: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    uid = _user_id()
    if invoice is None:
        raise ValueError("Missing required parameter: invoice")
    return await update_entity(uid, realm_id, entity="invoice", payload=invoice, sparse=True)


@mcp.tool(name="quickbooks-update-item", description="Updates an item.")
async def quickbooks_update_item(realm_id: Optional[str] = None, item: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    uid = _user_id()
    if item is None:
        raise ValueError("Missing required parameter: item")
    return await update_entity(uid, realm_id, entity="item", payload=item)


@mcp.tool(name="quickbooks-delete-purchase", description="Delete a specific purchase.")
async def quickbooks_delete_purchase(
    realm_id: Optional[str] = None,
    purchase: Optional[Dict[str, Any]] = None,
    purchase_id: Optional[str] = None,
    sync_token: Optional[str] = None,
) -> Dict[str, Any]:
    uid = _user_id()
    body = dict(purchase or {})
    if not body.get("Id"):
        if not purchase_id:
            raise ValueError("Provide purchase.Id in purchase OR provide purchase_id")
        body["Id"] = purchase_id
    if not body.get("SyncToken"):
        if sync_token is None:
            raise ValueError("Provide purchase.SyncToken in purchase OR provide sync_token")
        body["SyncToken"] = sync_token
    return await operate_entity(uid, realm_id, entity="purchase", operation="delete", payload=body)


@mcp.tool(name="quickbooks-void-invoice", description="Voids an invoice.")
async def quickbooks_void_invoice(
    realm_id: Optional[str] = None,
    invoice: Optional[Dict[str, Any]] = None,
    invoice_id: Optional[str] = None,
    sync_token: Optional[str] = None,
) -> Dict[str, Any]:
    uid = _user_id()
    body = dict(invoice or {})
    if not body.get("Id"):
        if not invoice_id:
            raise ValueError("Provide invoice.Id in invoice OR provide invoice_id")
        body["Id"] = invoice_id
    if not body.get("SyncToken"):
        if sync_token is None:
            raise ValueError("Provide invoice.SyncToken in invoice OR provide sync_token")
        body["SyncToken"] = sync_token
    return await operate_entity(uid, realm_id, entity="invoice", operation="void", payload=body)


# --- Send (email) ---


@mcp.tool(name="quickbooks-send-estimate", description="Sends an estimate by email.")
async def quickbooks_send_estimate(
    realm_id: Optional[str] = None,
    estimate_id: str = "",
    send_to: Optional[str] = None,
) -> Dict[str, Any]:
    uid = _user_id()
    if not estimate_id:
        raise ValueError("Missing required parameter: estimate_id")
    return await send_transaction(uid, realm_id, entity="estimate", entity_id=estimate_id, send_to=send_to)


@mcp.tool(name="quickbooks-send-invoice", description="Sends an invoice by email.")
async def quickbooks_send_invoice(
    realm_id: Optional[str] = None,
    invoice_id: str = "",
    send_to: Optional[str] = None,
) -> Dict[str, Any]:
    uid = _user_id()
    if not invoice_id:
        raise ValueError("Missing required parameter: invoice_id")
    return await send_transaction(uid, realm_id, entity="invoice", entity_id=invoice_id, send_to=send_to)


# --- Search / Query ---


@mcp.tool(name="quickbooks-search-query", description="Performs a search query against a QuickBooks entity using Intuit Query Language (IQL).", annotations={"readOnlyHint": True})
async def quickbooks_search_query(realm_id: Optional[str] = None, sql: str = "") -> Dict[str, Any]:
    uid = _user_id()
    if not sql:
        raise ValueError("Missing required parameter: sql")
    # allow passing full SELECT ...
    # NOTE: QBO IQL is case-sensitive for entity names (e.g., Invoice, Customer).
    #       We keep it user-supplied for maximum flexibility.
    if realm_id is None:
        # choose most recent
        companies = await db.list_connections(uid)
        if not companies:
            raise ValueError("No connected companies")
        realm_id = companies[0]["realm_id"]
    return await query_company(uid, realm_id, sql)


@mcp.tool(name="quickbooks-search-accounts", description="Search for accounts.", annotations={"readOnlyHint": True})
async def quickbooks_search_accounts(
    realm_id: Optional[str] = None,
    where: Optional[str] = None,
    start_position: int = 1,
    max_results: int = 50,
) -> Dict[str, Any]:
    uid = _user_id()
    return await search_entity(uid, realm_id, entity="Account", where=where, start_position=start_position, max_results=max_results)


@mcp.tool(name="quickbooks-search-customers", description="Searches for customers.", annotations={"readOnlyHint": True})
async def quickbooks_search_customers(
    realm_id: Optional[str] = None,
    where: Optional[str] = None,
    start_position: int = 1,
    max_results: int = 50,
) -> Dict[str, Any]:
    uid = _user_id()
    return await search_entity(uid, realm_id, entity="Customer", where=where, start_position=start_position, max_results=max_results)


@mcp.tool(name="quickbooks-search-invoices", description="Searches for invoices.", annotations={"readOnlyHint": True})
async def quickbooks_search_invoices(
    realm_id: Optional[str] = None,
    where: Optional[str] = None,
    start_position: int = 1,
    max_results: int = 50,
) -> Dict[str, Any]:
    uid = _user_id()
    return await search_entity(uid, realm_id, entity="Invoice", where=where, start_position=start_position, max_results=max_results)


@mcp.tool(
    name="quickbooks-sandbox-search-invoices",
    description="Searches for invoices against the sandbox hostname (requires sandbox Intuit tokens).",
    annotations={"readOnlyHint": True},
)
async def quickbooks_sandbox_search_invoices(
    realm_id: Optional[str] = None,
    where: Optional[str] = None,
    start_position: int = 1,
    max_results: int = 50,
) -> Dict[str, Any]:
    uid = _user_id()
    return await search_entity(
        uid,
        realm_id,
        entity="Invoice",
        where=where,
        start_position=start_position,
        max_results=max_results,
        sandbox=True,
    )


@mcp.tool(name="quickbooks-search-items", description="Searches for items.", annotations={"readOnlyHint": True})
async def quickbooks_search_items(
    realm_id: Optional[str] = None,
    where: Optional[str] = None,
    start_position: int = 1,
    max_results: int = 50,
) -> Dict[str, Any]:
    uid = _user_id()
    return await search_entity(uid, realm_id, entity="Item", where=where, start_position=start_position, max_results=max_results)


@mcp.tool(name="quickbooks-search-products", description="Search for products (Items where Type is Inventory or NonInventory).", annotations={"readOnlyHint": True})
async def quickbooks_search_products(
    realm_id: Optional[str] = None,
    where: Optional[str] = None,
    start_position: int = 1,
    max_results: int = 50,
) -> Dict[str, Any]:
    uid = _user_id()
    base_where = "(Type = 'Inventory' OR Type = 'NonInventory')"
    combined = f"({base_where}) AND ({where})" if where else base_where
    return await search_entity(uid, realm_id, entity="Item", where=combined, start_position=start_position, max_results=max_results)


@mcp.tool(name="quickbooks-search-services", description="Search for services (Items where Type is Service).", annotations={"readOnlyHint": True})
async def quickbooks_search_services(
    realm_id: Optional[str] = None,
    where: Optional[str] = None,
    start_position: int = 1,
    max_results: int = 50,
) -> Dict[str, Any]:
    uid = _user_id()
    base_where = "Type = 'Service'"
    combined = f"({base_where}) AND ({where})" if where else base_where
    return await search_entity(uid, realm_id, entity="Item", where=combined, start_position=start_position, max_results=max_results)


@mcp.tool(name="quickbooks-search-purchases", description="Searches for purchases.", annotations={"readOnlyHint": True})
async def quickbooks_search_purchases(
    realm_id: Optional[str] = None,
    where: Optional[str] = None,
    start_position: int = 1,
    max_results: int = 50,
) -> Dict[str, Any]:
    uid = _user_id()
    return await search_entity(uid, realm_id, entity="Purchase", where=where, start_position=start_position, max_results=max_results)


@mcp.tool(name="quickbooks-search-time-activities", description="Searches for time activities.", annotations={"readOnlyHint": True})
async def quickbooks_search_time_activities(
    realm_id: Optional[str] = None,
    where: Optional[str] = None,
    start_position: int = 1,
    max_results: int = 50,
) -> Dict[str, Any]:
    uid = _user_id()
    return await search_entity(uid, realm_id, entity="TimeActivity", where=where, start_position=start_position, max_results=max_results)


@mcp.tool(name="quickbooks-search-vendors", description="Searches for vendors.", annotations={"readOnlyHint": True})
async def quickbooks_search_vendors(
    realm_id: Optional[str] = None,
    where: Optional[str] = None,
    start_position: int = 1,
    max_results: int = 50,
) -> Dict[str, Any]:
    uid = _user_id()
    return await search_entity(uid, realm_id, entity="Vendor", where=where, start_position=start_position, max_results=max_results)
