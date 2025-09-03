"""
Shared pytest fixtures for the behaviour cloning benchmarks test suite.
"""
import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Generator
import pytest
from unittest.mock import Mock, MagicMock


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """
    Create a temporary directory that is automatically cleaned up after the test.
    
    Returns:
        Path: Path to the temporary directory
    """
    temp_path = Path(tempfile.mkdtemp())
    try:
        yield temp_path
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def temp_file(temp_dir: Path) -> Path:
    """
    Create a temporary file within a temporary directory.
    
    Args:
        temp_dir: The temporary directory fixture
        
    Returns:
        Path: Path to the temporary file
    """
    temp_file = temp_dir / "test_file.txt"
    temp_file.write_text("test content")
    return temp_file


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """
    Mock configuration dictionary for testing.
    
    Returns:
        Dict: Mock configuration with common test values
    """
    return {
        "model": {
            "name": "test_model",
            "hidden_dim": 128,
            "num_layers": 2,
        },
        "training": {
            "batch_size": 32,
            "learning_rate": 0.001,
            "num_epochs": 10,
        },
        "data": {
            "sequence_length": 100,
            "normalize": True,
        },
        "environment": {
            "name": "test_env",
            "max_steps": 1000,
        }
    }


@pytest.fixture
def mock_device():
    """
    Mock device for testing (CPU/CUDA agnostic).
    
    Returns:
        str: Mock device string
    """
    return "cpu"


@pytest.fixture
def mock_agent():
    """
    Mock agent for testing agent functionality.
    
    Returns:
        Mock: Mock agent with common methods
    """
    agent = MagicMock()
    agent.predict.return_value = [0.5, 0.3, 0.2]  # Mock action
    agent.train.return_value = {"loss": 0.1, "accuracy": 0.95}
    agent.save.return_value = True
    agent.load.return_value = True
    return agent


@pytest.fixture
def mock_dataset():
    """
    Mock dataset for testing data loading functionality.
    
    Returns:
        Mock: Mock dataset with common methods
    """
    dataset = MagicMock()
    dataset.__len__.return_value = 100
    dataset.__getitem__.return_value = {
        "observations": [1.0, 2.0, 3.0],
        "actions": [0.1, 0.2, 0.3],
        "rewards": 1.0
    }
    return dataset


@pytest.fixture
def mock_environment():
    """
    Mock environment for testing simulation functionality.
    
    Returns:
        Mock: Mock environment with gym-like interface
    """
    env = MagicMock()
    env.reset.return_value = [0.0, 0.0, 0.0]  # Mock observation
    env.step.return_value = ([0.1, 0.1, 0.1], 1.0, False, {})  # obs, reward, done, info
    env.close.return_value = None
    return env


@pytest.fixture
def sample_observation():
    """
    Sample observation data for testing.
    
    Returns:
        list: Sample observation vector
    """
    return [0.1, 0.2, 0.3, 0.4, 0.5]


@pytest.fixture
def sample_action():
    """
    Sample action data for testing.
    
    Returns:
        list: Sample action vector
    """
    return [0.0, 1.0, 0.0]


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """
    Directory containing test data files.
    
    Returns:
        Path: Path to test data directory
    """
    return Path(__file__).parent / "data"


@pytest.fixture
def clean_environment():
    """
    Clean environment variables for testing.
    Saves and restores environment state.
    """
    old_environ = dict(os.environ)
    # Clear potentially interfering environment variables
    test_vars_to_clear = [
        "CUDA_VISIBLE_DEVICES",
        "PYTHONPATH",
    ]
    
    for var in test_vars_to_clear:
        os.environ.pop(var, None)
    
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old_environ)


@pytest.fixture
def disable_gpu():
    """
    Force CPU-only mode for testing.
    """
    old_environ = dict(os.environ)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old_environ)


# Pytest configuration hooks
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location."""
    for item in items:
        # Mark tests in unit/ directory as unit tests
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        # Mark tests in integration/ directory as integration tests
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)