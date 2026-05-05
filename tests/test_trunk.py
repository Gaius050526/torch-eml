import json
from unittest.mock import MagicMock, patch
import torch
from torch_eml.trunk import LLMTrunk


FEATURES = [
    {"name": "market_size", "description": "log10 of TAM"},
    {"name": "team_exp", "description": "years of experience"},
    {"name": "growth", "description": "QoQ growth rate"},
]


class TestLLMTrunkInit:
    def test_stores_features(self):
        trunk = LLMTrunk(provider="anthropic", model="test", features=FEATURES)
        assert len(trunk.features) == 3

    def test_feature_names(self):
        trunk = LLMTrunk(provider="anthropic", model="test", features=FEATURES)
        assert trunk.feature_names == ["market_size", "team_exp", "growth"]


class TestLLMTrunkExtract:
    @patch("torch_eml.trunk._call_anthropic")
    def test_returns_tensor(self, mock_call):
        mock_call.return_value = json.dumps(
            {"market_size": 4.2, "team_exp": 12.0, "growth": 0.85}
        )
        trunk = LLMTrunk(provider="anthropic", model="test", features=FEATURES)
        result = trunk.extract("Some pitch text")
        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 3)

    @patch("torch_eml.trunk._call_anthropic")
    def test_correct_values(self, mock_call):
        mock_call.return_value = json.dumps(
            {"market_size": 4.2, "team_exp": 12.0, "growth": 0.85}
        )
        trunk = LLMTrunk(provider="anthropic", model="test", features=FEATURES)
        result = trunk.extract("Some pitch text")
        expected = torch.tensor([[4.2, 12.0, 0.85]])
        torch.testing.assert_close(result, expected)

    @patch("torch_eml.trunk._call_anthropic")
    def test_missing_feature_raises(self, mock_call):
        mock_call.return_value = json.dumps(
            {"market_size": 4.2, "team_exp": 12.0}
        )
        trunk = LLMTrunk(provider="anthropic", model="test", features=FEATURES)
        try:
            trunk.extract("Some pitch text")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "growth" in str(e)

    @patch("torch_eml.trunk._call_anthropic")
    def test_non_numeric_raises(self, mock_call):
        mock_call.return_value = json.dumps(
            {"market_size": "big", "team_exp": 12.0, "growth": 0.85}
        )
        trunk = LLMTrunk(provider="anthropic", model="test", features=FEATURES)
        try:
            trunk.extract("Some pitch text")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "market_size" in str(e)

    @patch("torch_eml.trunk._call_anthropic")
    def test_malformed_json_retries(self, mock_call):
        mock_call.side_effect = [
            "not json at all",
            json.dumps({"market_size": 4.2, "team_exp": 12.0, "growth": 0.85}),
        ]
        trunk = LLMTrunk(provider="anthropic", model="test", features=FEATURES)
        result = trunk.extract("Some pitch text")
        assert result.shape == (1, 3)
        assert mock_call.call_count == 2


class TestLLMTrunkOpenAI:
    @patch("torch_eml.trunk._call_openai")
    def test_openai_provider(self, mock_call):
        mock_call.return_value = json.dumps(
            {"market_size": 4.2, "team_exp": 12.0, "growth": 0.85}
        )
        trunk = LLMTrunk(provider="openai", model="gpt-4o", features=FEATURES)
        result = trunk.extract("Some pitch text")
        assert result.shape == (1, 3)
