"""Shared pytest fixtures for the Unwrap test suite."""
import pytest
from pathlib import Path

from backend.config import Settings
from backend.matcher import KnowledgeBase
from backend.models import UserProfile


@pytest.fixture(scope="session")
def kb() -> KnowledgeBase:
    """Fixture providing a loaded KnowledgeBase instance."""
    data_path = Path(__file__).resolve().parent.parent / "data" / "ingredients.json"
    return KnowledgeBase(data_path=data_path)


@pytest.fixture
def default_config() -> Settings:
    """Fixture providing default application settings."""
    return Settings(
        FUZZY_MATCH_THRESHOLD=92,
        LLM_PICK_ENABLED=False,
    )


@pytest.fixture
def milk_allergy_profile() -> UserProfile:
    return UserProfile(allergens=["milk"])


@pytest.fixture
def soy_allergy_profile() -> UserProfile:
    return UserProfile(allergens=["soy"])


@pytest.fixture
def multi_allergy_profile() -> UserProfile:
    return UserProfile(allergens=["milk", "soy", "peanut", "wheat"])


@pytest.fixture
def clear_profile() -> UserProfile:
    return UserProfile(allergens=["fish", "shellfish"])
