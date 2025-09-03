"""
Sample integration test to demonstrate integration test structure.
"""
import pytest


@pytest.mark.integration  
class TestSampleIntegration:
    """Sample integration test class."""
    
    def test_basic_integration(self):
        """Basic integration test."""
        assert True
        
    def test_with_multiple_fixtures(self, mock_agent, mock_environment):
        """Integration test using multiple fixtures."""
        # Simulate agent interacting with environment
        obs = mock_environment.reset()
        action = mock_agent.predict(obs)
        next_obs, reward, done, info = mock_environment.step(action)
        
        assert obs == [0.0, 0.0, 0.0]
        assert action == [0.5, 0.3, 0.2]
        assert next_obs == [0.1, 0.1, 0.1]
        assert reward == 1.0