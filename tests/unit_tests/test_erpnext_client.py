"""Unit tests for ERPNextClient and ERPNext tools."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deep_agent.erpnext.client import ERPNextClient
from deep_agent.erpnext.tools import (
    erpnext_create_document,
    erpnext_get_document,
    erpnext_list_documents,
    erpnext_submit_document,
    erpnext_update_document,
)


@pytest.fixture
def erpnext_client():
    return ERPNextClient(
        url="https://test.erpnext.com",
        api_key="test_key",
        api_secret="test_secret",
    )


@pytest.mark.asyncio
async def test_client_init_error():
    with patch.dict("os.environ", {"ERPNEXT_URL": ""}, clear=True):
        with pytest.raises(ValueError, match="ERPNEXT_URL must be provided"):
            ERPNextClient(url="")


@pytest.mark.asyncio
async def test_get_list(erpnext_client):
    mock_data = [{"name": "CUST-0001", "customer_name": "ACME Corp"}]
    mock_response = MagicMock()
    mock_response.is_success = True
    mock_response.status_code = 200
    mock_response.json.return_value = {"data": mock_data}

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response

        res = await erpnext_client.get_list("Customer", filters={"status": "Active"})
        assert res == mock_data
        mock_get.assert_called_once()


@pytest.mark.asyncio
async def test_get_doc(erpnext_client):
    mock_doc = {"name": "ACC-SINV-001", "grand_total": 1500.0, "status": "Unpaid"}
    mock_response = MagicMock()
    mock_response.is_success = True
    mock_response.status_code = 200
    mock_response.json.return_value = {"data": mock_doc}

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_response

        doc = await erpnext_client.get_doc("Sales Invoice", "ACC-SINV-001")
        assert doc["name"] == "ACC-SINV-001"
        assert doc["grand_total"] == 1500.0


@pytest.mark.asyncio
async def test_create_doc(erpnext_client):
    input_data = {"customer": "CUST-0001", "items": [{"item_code": "ITEM-1", "qty": 2}]}
    mock_response_data = {"name": "ACC-SINV-002", **input_data}
    mock_response = MagicMock()
    mock_response.is_success = True
    mock_response.status_code = 200
    mock_response.json.return_value = {"data": mock_response_data}

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        res = await erpnext_client.create_doc("Sales Invoice", input_data)
        assert res["name"] == "ACC-SINV-002"


@pytest.mark.asyncio
async def test_tools_list_documents():
    mock_list = [{"name": "LEAD-0001", "lead_name": "John Doe"}]
    with patch.object(ERPNextClient, "get_list", new_callable=AsyncMock) as mock_method:
        mock_method.return_value = mock_list
        res_str = await erpnext_list_documents.ainvoke(
            {"doctype": "Lead", "filters": '{"status": "Open"}'}
        )
        data = json.loads(res_str)
        assert len(data) == 1
        assert data[0]["name"] == "LEAD-0001"


@pytest.mark.asyncio
async def test_tools_get_document():
    mock_doc = {"name": "CUST-0001", "customer_name": "ACME Corp"}
    with patch.object(ERPNextClient, "get_doc", new_callable=AsyncMock) as mock_method:
        mock_method.return_value = mock_doc
        res_str = await erpnext_get_document.ainvoke(
            {"doctype": "Customer", "name": "CUST-0001"}
        )
        data = json.loads(res_str)
        assert data["name"] == "CUST-0001"


@pytest.mark.asyncio
async def test_tools_create_document():
    mock_created = {"name": "QUOT-0001", "docstatus": 0}
    with patch.object(ERPNextClient, "create_doc", new_callable=AsyncMock) as mock_method:
        mock_method.return_value = mock_created
        res_str = await erpnext_create_document.ainvoke(
            {"doctype": "Quotation", "data": '{"party_name": "ACME"}'}
        )
        data = json.loads(res_str)
        assert data["name"] == "QUOT-0001"


@pytest.mark.asyncio
async def test_tools_update_document():
    mock_updated = {"name": "QUOT-0001", "remarks": "Updated"}
    with patch.object(ERPNextClient, "update_doc", new_callable=AsyncMock) as mock_method:
        mock_method.return_value = mock_updated
        res_str = await erpnext_update_document.ainvoke(
            {"doctype": "Quotation", "name": "QUOT-0001", "data": '{"remarks": "Updated"}'}
        )
        data = json.loads(res_str)
        assert data["remarks"] == "Updated"


@pytest.mark.asyncio
async def test_tools_submit_document():
    mock_submitted = {"name": "ACC-SINV-001", "docstatus": 1}
    with patch.object(ERPNextClient, "submit_doc", new_callable=AsyncMock) as mock_method:
        mock_method.return_value = mock_submitted
        res_str = await erpnext_submit_document.ainvoke(
            {"doctype": "Sales Invoice", "name": "ACC-SINV-001"}
        )
        data = json.loads(res_str)
        assert data["docstatus"] == 1
