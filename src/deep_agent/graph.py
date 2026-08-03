"""ERPNext Administrator Deep Agent graph for deployment with Security Policies."""

from __future__ import annotations

import contextlib
import os
from datetime import datetime, timezone

from langchain_core.runnables import RunnableConfig

from deepagents import create_deep_agent
from langchain_core.tools import tool
from langgraph_sdk.runtime import ServerRuntime

from dotenv import load_dotenv

load_dotenv()

# Ensure fallback API key to prevent ChatDeepSeek initialization failure during import
if not os.getenv("DEEPSEEK_API_KEY"):
    os.environ["DEEPSEEK_API_KEY"] = "placeholder-key"

from deep_agent.erpnext.tools import ALL_ERPNEXT_TOOLS  # noqa: E402
from deep_agent.sandbox import get_or_create_sandbox  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402

checkpointer = MemorySaver()

DEFAULT_MODEL = os.getenv("DEEP_AGENT_MODEL", "deepseek:deepseek-v4-flash")

SYSTEM_PROMPT = """
You are an expert ERPNext Company Administrator AI Agent specializing in CRM, Sales Invoicing, and Accounting.

Workflow & Responsibilities:
1. Maintain a structured todo list for non-trivial company requests.
2. Query ERPNext DocTypes safely before performing operations.
3. Delegate specific operational domains to subagents:
   - crm_assistant: CRM operations (Lead, Customer, Opportunity, Quotation).
   - invoicing_assistant: Sales & Invoicing (Sales Order, Sales Invoice, Payment Entry).
   - accounting_assistant: Financial Accounting (Account, Journal Entry, Payment Reconciliation).
   - critic: Financial risk review and validation of ERPNext document integrity.
4. Always double-check currency amounts, tax calculations, customer IDs, and document statuses (Draft = 0, Submitted = 1, Cancelled = 2) before requesting updates or submissions.
5. Provide clear, concise summaries of ERPNext records and operations to the user.

Plan & Approval Policy (MANDATORY — Human-in-the-loop):
1. Read-only queries (get/list/search) are allowed at any time; use them to gather evidence before planning.
2. Before executing ANY mutation — creating, updating, or submitting a document (e.g., Customer, Quotation, Sales Order, Sales Invoice, Payment Entry, Journal Entry) — you MUST first present a concise, structured PLAN to the user. List each planned operation with: action (create/update/submit), DocType, key fields, amounts, and expected side effects.
3. After presenting the plan, STOP and wait for the user's explicit approval (e.g., "aprobado", "adelante", "sí", "yes"). Never execute mutations in the same turn in which you present the plan.
4. Proceed only after the user explicitly approves. If the user rejects or modifies the plan, adjust it accordingly and re-confirm before executing.
5. When financial amounts or document submissions are involved, run the plan through the critic subagent BEFORE asking for approval.

Security & Data Integrity Policy:
- Only access authorized business DocTypes (CRM, Sales, Invoicing, Accounting). Access to system settings, user credentials, or custom permissions is strictly prohibited.
- Modifying or submitting documents requires explicit user approval of the plan (Human-in-the-loop).
- Prefer concrete evidence from ERPNext queries over assumptions.

Cost & Token Efficiency Policy:
- Always specify explicit `fields` in `erpnext_list_documents` (e.g. ["name", "customer_name", "status", "grand_total"]) to avoid pulling massive unused system metadata.
- Keep list queries concise by setting small page limits (e.g. limit=5 or 10) unless the user requests a full export.
- Summarize results directly without repeating raw JSON outputs to minimize token consumption.
""".strip()


@tool
def utc_now() -> str:
    """Return the current UTC timestamp in ISO format."""
    return datetime.now(tz=timezone.utc).isoformat()


SUBAGENTS = [
    {
        "name": "crm_assistant",
        "description": "Handles CRM operations: Leads, Customers, Opportunities, and Quotations.",
        "system_prompt": (
            "You are a CRM specialist for ERPNext. Assist with managing Leads, Customers, "
            "Opportunities, and Quotations. Only access CRM DocTypes. "
            "Never create, update, or submit documents without explicit user approval of the plan "
            "by the main agent; propose the changes and report them back instead."
        ),
        "tools": [utc_now] + ALL_ERPNEXT_TOOLS,
    },
    {
        "name": "invoicing_assistant",
        "description": "Handles Sales Orders, Sales Invoices, and Payment Entries.",
        "system_prompt": (
            "You are a Sales & Invoicing specialist for ERPNext. Review and manage "
            "Sales Orders, Sales Invoices, and Payment Entries accurately. "
            "Never create, update, or submit documents without explicit user approval of the plan "
            "by the main agent; propose the changes and report them back instead."
        ),
        "tools": [utc_now] + ALL_ERPNEXT_TOOLS,
    },
    {
        "name": "accounting_assistant",
        "description": "Handles Chart of Accounts, Journal Entries, and Accounting Ledger records.",
        "system_prompt": (
            "You are an ERPNext Accounting specialist. Manage Chart of Accounts, Journal Entries, "
            "and verify ledger entries for financial consistency. "
            "Never create, update, or submit documents without explicit user approval of the plan "
            "by the main agent; propose the changes and report them back instead."
        ),
        "tools": [utc_now] + ALL_ERPNEXT_TOOLS,
    },
    {
        "name": "critic",
        "description": "Reviews financial plans, invoices, and ERPNext document updates for risks or errors.",
        "system_prompt": (
            "You are a financial and operational compliance reviewer. Check drafts for missing fields, "
            "unbalanced ledger entries, incorrect customer details, or status mismatches."
        ),
        "tools": [utc_now],
    },
]

MAIN_TOOLS = [utc_now] + ALL_ERPNEXT_TOOLS


ENABLE_INTERRUPTS = os.getenv("ENABLE_INTERRUPTS", "false").lower() == "true"


def _build_agent(backend=None):
    interrupt_config = (
        {
            "execute": True,
            "write_file": True,
            "erpnext_create_document": True,
            "erpnext_update_document": True,
            "erpnext_submit_document": True,
            "erpnext_send_email": True,
        }
        if ENABLE_INTERRUPTS
        else None
    )

    return create_deep_agent(
        model=DEFAULT_MODEL,
        tools=MAIN_TOOLS,
        backend=backend,
        system_prompt=SYSTEM_PROMPT,
        subagents=SUBAGENTS,
        interrupt_on=interrupt_config,
        checkpointer=checkpointer,
        name="erpnext_deep_agent",
    )


RO_AGENT = _build_agent()


@contextlib.asynccontextmanager
async def get_agent(config: RunnableConfig, runtime: ServerRuntime):
    ert = runtime.execution_runtime
    if ert:
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        backend = await get_or_create_sandbox(thread_id)
        yield _build_agent(backend=backend)
    else:
        yield RO_AGENT


# El compilado del grafo — usado por el runtime y tests
graph = RO_AGENT
