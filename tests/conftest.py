import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "live: marks tests that require live API keys (GROQ_API_KEY or GEMINI_API_KEY); "
        "deselect with -k 'not live'",
    )
