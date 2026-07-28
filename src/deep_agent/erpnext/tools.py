"""LangChain async tools for interacting with ERPNext with integrated security policy and audit logging."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool

from deep_agent.erpnext.client import ERPNextClient, ERPNextError
from deep_agent.erpnext.security import ERPNextSecurityError, ERPNextSecurityPolicy


def _get_client() -> ERPNextClient:
    return ERPNextClient()


@tool
async def erpnext_list_documents(
    doctype: str,
    filters: str | dict[str, Any] | list[Any] | None = None,
    fields: list[str] | None = None,
    limit: int = 20,
    order_by: str | None = None,
) -> str:
    """Fetch a list of documents from ERPNext.

    Allowed DocTypes: Customer, Lead, Opportunity, Quotation, Sales Order, Sales Invoice, Payment Entry, Account, Journal Entry.

    Args:
        doctype: The ERPNext DocType name (e.g. "Customer", "Sales Invoice", "Lead").
        filters: Optional filters as dict or JSON string. E.g. {"status": "Unpaid"}.
        fields: Optional list of fields to retrieve (e.g. ["name", "customer_name", "status"]).
        limit: Max number of records to return (default 20).
        order_by: Field to order by (e.g. "creation desc").
    """
    try:
        clean_doctype = ERPNextSecurityPolicy.validate_doctype(doctype)
        clean_filters = ERPNextSecurityPolicy.sanitize_filters(filters)
    except ERPNextSecurityError as sec_err:
        return f"Security Violation: {sec_err}"

    client = _get_client()
    parsed_filters = clean_filters
    if isinstance(clean_filters, str) and clean_filters.strip():
        try:
            parsed_filters = json.loads(clean_filters)
        except Exception:
            pass

    try:
        results = await client.get_list(
            doctype=clean_doctype,
            filters=parsed_filters,
            fields=fields,
            limit_page_length=limit,
            order_by=order_by,
        )
        return json.dumps(results, indent=2, default=str)
    except ERPNextError as exc:
        return f"Error listing {clean_doctype}: {exc}"
    except Exception as exc:
        return f"Unexpected error listing {clean_doctype}: {exc}"


@tool
async def erpnext_get_document(doctype: str, name: str) -> str:
    """Fetch complete details of a specific ERPNext document.

    Args:
        doctype: The ERPNext DocType (e.g. "Sales Invoice", "Customer").
        name: The unique ID or name of the document (e.g. "ACC-SINV-2026-00001" or "CUST-0001").
    """
    try:
        clean_doctype = ERPNextSecurityPolicy.validate_doctype(doctype)
    except ERPNextSecurityError as sec_err:
        return f"Security Violation: {sec_err}"

    client = _get_client()
    try:
        doc = await client.get_doc(doctype=clean_doctype, name=name)
        return json.dumps(doc, indent=2, default=str)
    except ERPNextError as exc:
        return f"Error fetching {clean_doctype} '{name}': {exc}"
    except Exception as exc:
        return f"Unexpected error fetching {clean_doctype} '{name}': {exc}"


@tool
async def erpnext_create_document(doctype: str, data: str | dict[str, Any]) -> str:
    """Create a new document in ERPNext.

    Args:
        doctype: The ERPNext DocType (e.g. "Customer", "Lead", "Quotation", "Sales Invoice").
        data: A dict or JSON string with the field values for the document.
    """
    try:
        clean_doctype = ERPNextSecurityPolicy.validate_doctype(doctype)
    except ERPNextSecurityError as sec_err:
        ERPNextSecurityPolicy.log_audit_event("create", doctype, data=data, success=False, error=str(sec_err))
        return f"Security Violation: {sec_err}"

    client = _get_client()
    payload = data
    if isinstance(data, str):
        try:
            payload = json.loads(data)
        except Exception as exc:
            ERPNextSecurityPolicy.log_audit_event("create", clean_doctype, data=data, success=False, error=str(exc))
            return f"Invalid JSON data for creation: {exc}"

    try:
        doc = await client.create_doc(doctype=clean_doctype, data=payload)
        doc_name = doc.get("name") if isinstance(doc, dict) else None
        ERPNextSecurityPolicy.log_audit_event("create", clean_doctype, doc_name=doc_name, data=payload, success=True)
        return json.dumps(doc, indent=2, default=str)
    except ERPNextError as exc:
        ERPNextSecurityPolicy.log_audit_event("create", clean_doctype, data=payload, success=False, error=str(exc))
        return f"Error creating {clean_doctype}: {exc}"
    except Exception as exc:
        ERPNextSecurityPolicy.log_audit_event("create", clean_doctype, data=payload, success=False, error=str(exc))
        return f"Unexpected error creating {clean_doctype}: {exc}"


@tool
async def erpnext_update_document(doctype: str, name: str, data: str | dict[str, Any]) -> str:
    """Update fields on an existing draft or editable ERPNext document.

    Args:
        doctype: The ERPNext DocType.
        name: The document ID/name.
        data: A dict or JSON string with updated field values.
    """
    try:
        clean_doctype = ERPNextSecurityPolicy.validate_doctype(doctype)
    except ERPNextSecurityError as sec_err:
        ERPNextSecurityPolicy.log_audit_event("update", doctype, doc_name=name, data=data, success=False, error=str(sec_err))
        return f"Security Violation: {sec_err}"

    client = _get_client()
    payload = data
    if isinstance(data, str):
        try:
            payload = json.loads(data)
        except Exception as exc:
            ERPNextSecurityPolicy.log_audit_event("update", clean_doctype, doc_name=name, data=data, success=False, error=str(exc))
            return f"Invalid JSON data for update: {exc}"

    try:
        doc = await client.update_doc(doctype=clean_doctype, name=name, data=payload)
        ERPNextSecurityPolicy.log_audit_event("update", clean_doctype, doc_name=name, data=payload, success=True)
        return json.dumps(doc, indent=2, default=str)
    except ERPNextError as exc:
        ERPNextSecurityPolicy.log_audit_event("update", clean_doctype, doc_name=name, data=payload, success=False, error=str(exc))
        return f"Error updating {clean_doctype} '{name}': {exc}"
    except Exception as exc:
        ERPNextSecurityPolicy.log_audit_event("update", clean_doctype, doc_name=name, data=payload, success=False, error=str(exc))
        return f"Unexpected error updating {clean_doctype} '{name}': {exc}"


@tool
async def erpnext_submit_document(doctype: str, name: str) -> str:
    """Submit a document in ERPNext (sets docstatus=1, converting draft to official/submitted).

    Args:
        doctype: The ERPNext DocType (e.g. "Sales Invoice", "Journal Entry", "Quotation").
        name: The document ID/name.
    """
    try:
        clean_doctype = ERPNextSecurityPolicy.validate_doctype(doctype)
    except ERPNextSecurityError as sec_err:
        ERPNextSecurityPolicy.log_audit_event("submit", doctype, doc_name=name, success=False, error=str(sec_err))
        return f"Security Violation: {sec_err}"

    client = _get_client()
    try:
        doc = await client.submit_doc(doctype=clean_doctype, name=name)
        ERPNextSecurityPolicy.log_audit_event("submit", clean_doctype, doc_name=name, success=True)
        return json.dumps(doc, indent=2, default=str)
    except ERPNextError as exc:
        ERPNextSecurityPolicy.log_audit_event("submit", clean_doctype, doc_name=name, success=False, error=str(exc))
        return f"Error submitting {clean_doctype} '{name}': {exc}"
    except Exception as exc:
        ERPNextSecurityPolicy.log_audit_event("submit", clean_doctype, doc_name=name, success=False, error=str(exc))
        return f"Unexpected error submitting {clean_doctype} '{name}': {exc}"


ALL_ERPNEXT_TOOLS = [
    erpnext_list_documents,
    erpnext_get_document,
    erpnext_create_document,
    erpnext_update_document,
    erpnext_submit_document,
]
