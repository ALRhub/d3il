"""
Sample unit test to demonstrate unit test structure.
"""
import pytest


@pytest.mark.unit
class TestSampleUnit:
    """Sample unit test class."""
    
    def test_basic_assertion(self):
        """Basic unit test."""
        assert 1 + 1 == 2
        
    def test_with_fixture(self, mock_config):
        """Unit test using a fixture."""
        assert mock_config["model"]["name"] == "test_model"
        assert mock_config["training"]["batch_size"] == 32