"""Async client for Frappe / ERPNext REST API."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx


class ERPNextError(Exception):
    """Base exception for ERPNext API errors."""

    def __init__(self, message: str, status_code: int | None = None, response_data: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data


class ERPNextClient:
    """Async client to interact with ERPNext via REST API."""

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        api_secret: str | None = None,
        timeout: float = 30.0,
    ):
        self.url = (url or os.getenv("ERPNEXT_URL", "")).rstrip("/")
        self.api_key = api_key or os.getenv("ERPNEXT_API_KEY", "")
        self.api_secret = api_secret or os.getenv("ERPNEXT_API_SECRET", "")
        self.timeout = timeout

        if not self.url:
            raise ValueError("ERPNEXT_URL must be provided or set in environment.")

    def _get_headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if self.api_key and self.api_secret:
            headers["Authorization"] = f"token {self.api_key}:{self.api_secret}"
        return headers

    def _handle_response(self, response: httpx.Response) -> Any:
        try:
            data = response.json()
        except Exception:
            data = response.text

        if response.is_success:
            if isinstance(data, dict) and "data" in data:
                return data["data"]
            if isinstance(data, dict) and "message" in data:
                return data["message"]
            return data

        error_msg = f"ERPNext API Error [{response.status_code}]: {response.reason_phrase}"
        if isinstance(data, dict):
            if "exception" in data:
                error_msg += f" - {data['exception']}"
            elif "_server_messages" in data:
                try:
                    msgs = json.loads(data["_server_messages"])
                    error_msg += f" - {', '.join(msgs)}"
                except Exception:
                    error_msg += f" - {data['_server_messages']}"
            elif "message" in data:
                error_msg += f" - {data['message']}"

        raise ERPNextError(error_msg, status_code=response.status_code, response_data=data)

    async def get_list(
        self,
        doctype: str,
        filters: dict[str, Any] | list | None = None,
        fields: list[str] | None = None,
        limit_page_length: int = 20,
        order_by: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch a list of documents for a given DocType."""
        endpoint = f"{self.url}/api/resource/{doctype}"
        params: dict[str, Any] = {"limit_page_length": limit_page_length}

        if filters:
            params["filters"] = json.dumps(filters)
        if fields:
            params["fields"] = json.dumps(fields)
        if order_by:
            params["order_by"] = order_by

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            res = await client.get(endpoint, headers=self._get_headers(), params=params)
            return self._handle_response(res)

    async def get_doc(self, doctype: str, name: str) -> dict[str, Any]:
        """Fetch a single document by DocType and name."""
        endpoint = f"{self.url}/api/resource/{doctype}/{name}"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            res = await client.get(endpoint, headers=self._get_headers())
            return self._handle_response(res)

    async def create_doc(self, doctype: str, data: dict[str, Any]) -> dict[str, Any]:
        """Create a new document for a given DocType."""
        endpoint = f"{self.url}/api/resource/{doctype}"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            res = await client.post(endpoint, headers=self._get_headers(), json=data)
            return self._handle_response(res)

    async def update_doc(self, doctype: str, name: str, data: dict[str, Any]) -> dict[str, Any]:
        """Update an existing document."""
        endpoint = f"{self.url}/api/resource/{doctype}/{name}"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            res = await client.put(endpoint, headers=self._get_headers(), json=data)
            return self._handle_response(res)

    async def submit_doc(self, doctype: str, name: str) -> dict[str, Any]:
        """Submit a document (set docstatus=1)."""
        return await self.update_doc(doctype, name, {"docstatus": 1})

    async def cancel_doc(self, doctype: str, name: str) -> dict[str, Any]:
        """Cancel a document (set docstatus=2)."""
        return await self.update_doc(doctype, name, {"docstatus": 2})

    async def run_method(self, method: str, args: dict[str, Any] | None = None) -> Any:
        """Call an RPC method in Frappe/ERPNext."""
        endpoint = f"{self.url}/api/method/{method}"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            res = await client.post(endpoint, headers=self._get_headers(), json=args or {})
            return self._handle_response(res)

    async def send_email(
        self,
        recipients: str | list[str],
        subject: str,
        message: str,
        reference_doctype: str | None = None,
        reference_name: str | None = None,
    ) -> Any:
        """Send an email using Frappe/ERPNext's sendmail RPC method."""
        recipients_str = recipients if isinstance(recipients, str) else ", ".join(recipients)
        args: dict[str, Any] = {
            "recipients": recipients_str,
            "subject": subject,
            "message": message,
        }
        if reference_doctype:
            args["reference_doctype"] = reference_doctype
        if reference_name:
            args["reference_name"] = reference_name

        return await self.run_method("frappe.sendmail", args=args)
