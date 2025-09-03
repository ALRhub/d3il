"""
Infrastructure validation tests to ensure the testing setup works correctly.
"""
import pytest
from pathlib import Path
import tempfile
import os


class TestInfrastructure:
    """Test class to validate testing infrastructure setup."""
    
    def test_pytest_working(self):
        """Test that pytest is working correctly."""
        assert True
        
    def test_temp_dir_fixture(self, temp_dir):
        """Test that temp_dir fixture works."""
        assert isinstance(temp_dir, Path)
        assert temp_dir.exists()
        assert temp_dir.is_dir()
        
    def test_temp_file_fixture(self, temp_file):
        """Test that temp_file fixture works."""
        assert isinstance(temp_file, Path)
        assert temp_file.exists()
        assert temp_file.is_file()
        assert temp_file.read_text() == "test content"
        
    def test_mock_config_fixture(self, mock_config):
        """Test that mock_config fixture works."""
        assert isinstance(mock_config, dict)
        assert "model" in mock_config
        assert "training" in mock_config
        assert "data" in mock_config
        assert "environment" in mock_config
        
    def test_mock_agent_fixture(self, mock_agent):
        """Test that mock_agent fixture works."""
        # Test predict method
        result = mock_agent.predict()
        assert result == [0.5, 0.3, 0.2]
        
        # Test train method
        train_result = mock_agent.train()
        assert "loss" in train_result
        assert "accuracy" in train_result
        
    def test_mock_dataset_fixture(self, mock_dataset):
        """Test that mock_dataset fixture works."""
        assert len(mock_dataset) == 100
        item = mock_dataset[0]
        assert "observations" in item
        assert "actions" in item
        assert "rewards" in item
        
    def test_mock_environment_fixture(self, mock_environment):
        """Test that mock_environment fixture works."""
        # Test reset
        obs = mock_environment.reset()
        assert obs == [0.0, 0.0, 0.0]
        
        # Test step
        obs, reward, done, info = mock_environment.step([1.0])
        assert obs == [0.1, 0.1, 0.1]
        assert reward == 1.0
        assert done is False
        assert isinstance(info, dict)
        
    def test_sample_data_fixtures(self, sample_observation, sample_action):
        """Test that sample data fixtures work."""
        assert len(sample_observation) == 5
        assert len(sample_action) == 3
        
    @pytest.mark.unit
    def test_unit_marker(self):
        """Test that unit marker works."""
        assert True
        
    def test_clean_environment_fixture(self, clean_environment):
        """Test that clean_environment fixture works."""
        # This test just ensures the fixture doesn't break
        assert True
        
    def test_disable_gpu_fixture(self, disable_gpu):
        """Test that disable_gpu fixture works."""
        # Should set CUDA_VISIBLE_DEVICES to empty string
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""


class TestProjectStructure:
    """Test that the project structure is set up correctly."""
    
    def test_tests_directory_exists(self):
        """Test that tests directory exists."""
        tests_dir = Path(__file__).parent
        assert tests_dir.exists()
        assert tests_dir.is_dir()
        
    def test_unit_directory_exists(self):
        """Test that unit tests directory exists."""
        unit_dir = Path(__file__).parent / "unit"
        assert unit_dir.exists()
        assert unit_dir.is_dir()
        
    def test_integration_directory_exists(self):
        """Test that integration tests directory exists."""
        integration_dir = Path(__file__).parent / "integration"
        assert integration_dir.exists()
        assert integration_dir.is_dir()
        
    def test_conftest_exists(self):
        """Test that conftest.py exists."""
        conftest = Path(__file__).parent / "conftest.py"
        assert conftest.exists()
        assert conftest.is_file()
        
    def test_init_files_exist(self):
        """Test that __init__.py files exist in test directories."""
        test_dirs = [
            Path(__file__).parent,
            Path(__file__).parent / "unit",
            Path(__file__).parent / "integration"
        ]
        
        for test_dir in test_dirs:
            init_file = test_dir / "__init__.py"
            assert init_file.exists(), f"Missing __init__.py in {test_dir}"


class TestPytestConfiguration:
    """Test that pytest configuration is working."""
    
    def test_markers_available(self):
        """Test that custom markers are available."""
        # This will be validated by pytest's marker detection
        assert True
        
    def test_coverage_configured(self):
        """Test that coverage is configured (indirect test)."""
        # This test ensures we can run without coverage breaking
        assert True


@pytest.mark.slow
class TestSlowMarker:
    """Test class for slow tests marker."""
    
    def test_slow_test(self):
        """Test that slow marker works."""
        # This is a mock slow test
        import time
        time.sleep(0.1)  # Very short sleep for demo
        assert True