"""Test fixtures."""

import pytest
from src.storage.database import Database


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    import asyncio
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def db():
    """Initialize test database."""
    # Use test database URL
    import os
    os.environ["DATABASE_URL"] = "postgresql://hybridrag:hybridrag@localhost:5432/hybridrag_test"

    await Database.init_db()
    yield Database
    await Database.close()
