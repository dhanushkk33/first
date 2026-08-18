"""
Mini ADK starter agent.

Keep this small on purpose — one Gemini-backed agent with a single tool.
Add more tools / sub-agents once the CI/CD skeleton is proven out.
"""

import os
from google.adk.agents import Agent


def get_weather(city: str) -> dict:
    """Retrieves the current weather report for a specified city.

    Args:
        city (str): The name of the city (e.g. "Chennai").

    Returns:
        dict: {"status": "success", "report": str} or
              {"status": "error", "error_message": str}
    """
    mock_weather_db = {
        "chennai": "It's hot and humid, around 34°C.",
        "bangalore": "Pleasant, 24°C with light clouds.",
        "mumbai": "Warm and humid, 31°C, chance of rain.",
    }
    key = city.lower().strip()
    if key in mock_weather_db:
        return {"status": "success", "report": mock_weather_db[key]}
    return {
        "status": "error",
        "error_message": f"No weather data available for '{city}'.",
    }


# GOOGLE_API_KEY (Gemini) is read from the environment at runtime.
# Locally: put it in src/main_agent/.env (gitignored).
# In CI/CD: injected as a GitHub Actions secret, passed to Databricks as a
# serving-endpoint environment variable — see deploy/deploy_to_databricks.py
root_agent = Agent(
    name="mini_weather_agent",
    model="gemini-2.0-flash",
    description="A small starter agent that answers weather questions for a city.",
    instruction=(
        "You are a concise, helpful assistant. "
        "Whenever the user asks about weather in a city, call the get_weather tool. "
        "If the tool returns an error, apologize briefly and say the city isn't supported yet. "
        "Keep every answer to 1-2 sentences."
    ),
    tools=[get_weather],
)
