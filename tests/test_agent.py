"""
Unit tests for the mini ADK agent.

These deliberately test the plain-Python tool function and the agent's
static configuration only — no live Gemini calls. That keeps the CI job
fast, free, and independent of API keys/quota. Add an integration test
(marked with @pytest.mark.integration) later if you want a real call.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from main_agent.agent import root_agent, get_weather  # noqa: E402


def test_get_weather_known_city():
    result = get_weather("Chennai")
    assert result["status"] == "success"
    assert "hot" in result["report"].lower()


def test_get_weather_is_case_insensitive():
    result = get_weather("BANGALORE")
    assert result["status"] == "success"


def test_get_weather_unknown_city_returns_error():
    result = get_weather("Atlantis")
    assert result["status"] == "error"
    assert "no weather data" in result["error_message"].lower()


def test_agent_is_configured_correctly():
    assert root_agent.name == "mini_weather_agent"
    assert root_agent.model == "gemini-2.0-flash"
    tool_names = [t.__name__ for t in root_agent.tools]
    assert "get_weather" in tool_names


def test_agent_has_an_instruction():
    assert root_agent.instruction and len(root_agent.instruction) > 10
