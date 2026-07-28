import os
import pytest

# Ensure dummy environment variables for unit tests
os.environ.setdefault("DEEPSEEK_API_KEY", "mock-deepseek-key")
os.environ.setdefault("ERPNEXT_URL", "https://mock.erpnext.com")
os.environ.setdefault("ERPNEXT_API_KEY", "mock-api-key")
os.environ.setdefault("ERPNEXT_API_SECRET", "mock-api-secret")


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    return "asyncio"
