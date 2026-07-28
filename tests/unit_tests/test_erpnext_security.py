"""Unit tests for ERPNext security policy, validation, and audit logging."""

from __future__ import annotations

import pytest

from deep_agent.erpnext.security import ERPNextSecurityError, ERPNextSecurityPolicy
from deep_agent.erpnext.tools import (
    erpnext_create_document,
    erpnext_get_document,
    erpnext_list_documents,
)


def test_doctype_allowlist_allowed():
    assert ERPNextSecurityPolicy.is_doctype_allowed("Customer") is True
    assert ERPNextSecurityPolicy.is_doctype_allowed("Sales Invoice") is True
    assert ERPNextSecurityPolicy.is_doctype_allowed("Journal Entry") is True
    assert ERPNextSecurityPolicy.is_doctype_allowed("Lead") is True


def test_doctype_allowlist_prohibited():
    assert ERPNextSecurityPolicy.is_doctype_allowed("User") is False
    assert ERPNextSecurityPolicy.is_doctype_allowed("DocType") is False
    assert ERPNextSecurityPolicy.is_doctype_allowed("System Settings") is False
    assert ERPNextSecurityPolicy.is_doctype_allowed("Access Log") is False


def test_validate_doctype_raises_error():
    with pytest.raises(ERPNextSecurityError, match="Access Denied: DocType 'User' is not allowed"):
        ERPNextSecurityPolicy.validate_doctype("User")


def test_sanitize_filters_prohibits_sql_injection():
    with pytest.raises(ERPNextSecurityError, match="forbidden keyword 'select' detected"):
        ERPNextSecurityPolicy.sanitize_filters("SELECT * FROM tabUser")


@pytest.mark.asyncio
async def test_tool_blocks_prohibited_doctype():
    res = await erpnext_get_document.ainvoke({"doctype": "UserPermission", "name": "test"})
    assert "Security Violation" in res
    assert "Access Denied" in res


@pytest.mark.asyncio
async def test_tool_blocks_prohibited_list_doctype():
    res = await erpnext_list_documents.ainvoke({"doctype": "System Settings"})
    assert "Security Violation" in res
    assert "Access Denied" in res


@pytest.mark.asyncio
async def test_tool_blocks_prohibited_create_doctype():
    res = await erpnext_create_document.ainvoke({"doctype": "User", "data": '{"email": "hacker@test.com"}'})
    assert "Security Violation" in res
    assert "Access Denied" in res
