"""Unit tests for ERPNext email client and tool integration."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from deep_agent.erpnext.client import ERPNextClient
from deep_agent.erpnext.tools import erpnext_send_email


@pytest.fixture
def erpnext_client():
    return ERPNextClient(
        url="https://test.erpnext.com",
        api_key="test_key",
        api_secret="test_secret",
    )


@pytest.mark.asyncio
async def test_client_send_email(erpnext_client):
    with patch.object(ERPNextClient, "run_method", new_callable=AsyncMock) as mock_method:
        mock_method.return_value = {"status": "queued"}
        res = await erpnext_client.send_email(
            recipients="client@company.com",
            subject="Invoice Notification",
            message="Your invoice is ready.",
            reference_doctype="Sales Invoice",
            reference_name="ACC-SINV-001",
        )
        assert res["status"] == "queued"
        mock_method.assert_called_once_with(
            "frappe.sendmail",
            args={
                "recipients": "client@company.com",
                "subject": "Invoice Notification",
                "message": "Your invoice is ready.",
                "reference_doctype": "Sales Invoice",
                "reference_name": "ACC-SINV-001",
            },
        )


@pytest.mark.asyncio
async def test_tool_send_email_success():
    with patch.object(ERPNextClient, "send_email", new_callable=AsyncMock) as mock_send:
        mock_send.return_value = {"status": "ok"}
        res_str = await erpnext_send_email.ainvoke(
            {
                "recipients": "test@domain.com",
                "subject": "Test Subject",
                "message": "<p>Hello</p>",
            }
        )
        assert "queued successfully" in res_str
        assert "status" in res_str


@pytest.mark.asyncio
async def test_tool_send_email_prohibited_reference_doctype():
    res_str = await erpnext_send_email.ainvoke(
        {
            "recipients": "test@domain.com",
            "subject": "Test",
            "message": "Hello",
            "reference_doctype": "User",
            "reference_name": "Administrator",
        }
    )
    assert "Security Violation" in res_str
    assert "Access Denied" in res_str
