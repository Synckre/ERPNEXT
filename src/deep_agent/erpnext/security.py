"""Security, validation, and audit logging module for ERPNext integration."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("deep_agent.erpnext.security")

# Allowlist of permitted DocTypes per functional domain
DOMAIN_ALLOWLISTS: dict[str, set[str]] = {
    "crm": {"Lead", "Customer", "Opportunity", "Quotation", "Issue", "Address", "Contact", "Communication", "Email Queue"},
    "invoicing": {"Sales Order", "Sales Invoice", "Payment Entry", "Customer", "Item", "Communication", "Email Queue"},
    "accounting": {
        "Account",
        "Journal Entry",
        "Payment Entry",
        "Payment Reconciliation",
        "GL Entry",
        "Communication",
        "Email Queue",
    },
    "notifications": {"Communication", "Email Queue", "Notification"},
}

# Global union of all permitted DocTypes
GLOBAL_ALLOWLIST: set[str] = set().union(*DOMAIN_ALLOWLISTS.values())


class ERPNextSecurityError(Exception):
    """Exception raised when an ERPNext security policy is violated."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class ERPNextSecurityPolicy:
    """Security policy manager for ERPNext operations."""

    @staticmethod
    def is_doctype_allowed(doctype: str, domain: str | None = None) -> bool:
        """Check if a DocType is permitted globally or for a specific domain."""
        clean_doctype = doctype.strip()
        if domain and domain.lower() in DOMAIN_ALLOWLISTS:
            return clean_doctype in DOMAIN_ALLOWLISTS[domain.lower()]
        return clean_doctype in GLOBAL_ALLOWLIST

    @staticmethod
    def validate_doctype(doctype: str, domain: str | None = None) -> str:
        """Validate that a DocType is allowed, raising an error if prohibited.

        Returns the cleaned DocType name.
        """
        clean_doctype = doctype.strip()
        if not ERPNextSecurityPolicy.is_doctype_allowed(clean_doctype, domain=domain):
            domain_str = f" in domain '{domain}'" if domain else ""
            raise ERPNextSecurityError(
                f"Access Denied: DocType '{clean_doctype}' is not allowed{domain_str}. "
                "Only standard CRM, Invoicing, and Accounting DocTypes are accessible."
            )
        return clean_doctype

    @staticmethod
    def sanitize_filters(filters: Any) -> Any:
        """Basic sanitization check for query filters to prevent malformed or illegal inputs."""
        if filters is None:
            return None

        if isinstance(filters, str):
            # Block suspicious characters or SQL keyword patterns if raw SQL string attempted
            forbidden_keywords = ["select ", "union ", "drop ", "insert ", "delete ", "exec ", "--", ";"]
            filters_lower = filters.lower()
            for kw in forbidden_keywords:
                if kw in filters_lower:
                    raise ERPNextSecurityError(f"Invalid filter expression: forbidden keyword '{kw.strip()}' detected.")

        return filters

    @staticmethod
    def log_audit_event(
        action: str,
        doctype: str,
        doc_name: str | None = None,
        data: Any = None,
        success: bool = True,
        error: str | None = None,
    ) -> None:
        """Log a structured audit event for all mutation operations."""
        event = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "action": action,
            "doctype": doctype,
            "document_name": doc_name,
            "success": success,
            "data_summary": str(data)[:200] if data else None,
            "error": error,
        }
        log_message = f"[AUDIT_EVENT] {json.dumps(event)}"
        if success:
            logger.info(log_message)
        else:
            logger.warning(log_message)
