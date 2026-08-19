import os
import sys
import time

# 🛑 THE MASTER FIX: Stop MLflow from spying on our environment variables!
os.environ["MLFLOW_RECORD_ENV_VARS_IN_MODEL_LOGGING"] = "false"

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from mlflow.models.signature import infer_signature

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from main_agent.agent import root_agent  # noqa: E402


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


DATABRICKS_HOST = _require_env("DATABRICKS_HOST")
DATABRICKS_TOKEN = _require_env("DATABRICKS_TOKEN")
GOOGLE_API_KEY = _require_env("GOOGLE_API_KEY")

UC_CATALOG = os.environ.get("UC_CATALOG", "main")
UC_SCHEMA = os.environ.get("UC_SCHEMA", "agents")
UC_MODEL_NAME = os.environ.get("UC_MODEL_NAME", "mini_weather_agent")
ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "mini-weather-agent-endpoint")
GIT_SHA = os.environ.get("GITHUB_SHA", "local")
UC_MODEL_FQN = f"{UC_CATALOG}.{UC_SCHEMA}.{UC_MODEL_NAME}"

os.environ["DATABRICKS_HOST"] = DATABRICKS_HOST
os.environ["DATABRICKS_TOKEN"] = DATABRICKS_TOKEN


class _ADKAgentWrapper(mlflow.pyfunc.PythonModel):
    """Wraps the ADK agent as an MLflow pyfunc model so Databricks can serve it."""

    def load_context(self, context):
        from google.adk.runners import InMemoryRunner
        os.environ.setdefault("GOOGLE_API_KEY", GOOGLE_API_KEY)
        self.runner = InMemoryRunner(agent=root_agent)

    def predict(self, context, model_input):
        import asyncio
        from google.genai import types 

        if hasattr(model_input, "to_dict"):  
            prompts = model_input["prompt"].tolist()
        elif isinstance(model_input, dict):
            prompts = model_input.get("prompt", [str(model_input)])
        else:
            prompts = [str(model_input)]

        async def _ask(prompt: str) -> str:
            session = await self.runner.session_service.create_session(
                app_name=self.runner.app_name, user_id="ci_cd_user"
            )
            final_text = ""
            
            msg = types.Content(
                role="user", 
                parts=[types.Part(text=prompt)]
            )

            async for event in self.runner.run_async(
                user_id="ci_cd_user",
                session_id=session.id,
                new_message=msg, 
            ):
                if hasattr(event, 'content') and event.content and hasattr(event.content, 'parts'):
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            final_text += part.text
            return final_text

        return [asyncio.run(_ask(p)) for p in prompts]


def log_and_register_model() -> str:
    """Logs the agent to MLflow and registers it in Unity Catalog."""
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment("/Shared/mini_weather_agent_logs")

    input_example = {"prompt": "What is the weather in Chennai?"}
    output_example = ["It's hot and humid, around 34°C."]
    signature = infer_signature(input_example, output_example)

    with mlflow.start_run(run_name=f"mini_weather_agent_{GIT_SHA[:7]}") as run:
        mlflow.set_tag("git_sha", GIT_SHA)
        mlflow.set_tag("source", "github-actions-ci-cd")

        conda_env = {
        "name": "mlflow-env",
        "channels": ["conda-forge"],
        "dependencies": [
            "python=3.11",
            "pip<=26.2.1",
            {
                "pip": [
                    "google-adk>=2.6.0",
                    "mlflow",
                    "databricks-sdk",
                    "google-genai"
                ]
            }
        ]
    }

        logged = mlflow.pyfunc.log_model(
            artifact_path="agent",
            python_model=_ADKAgentWrapper(),
            registered_model_name=UC_MODEL_FQN,
            signature=signature,
            input_example=input_example,
            classmethodode_paths=[os.path.join(os.path.dirname(__file__), "..", "src", "main_agent")],
            conda_env=conda_env,
        )
        print(f"Logged model in run {run.info.run_id}, version {logged.registered_model_version}")

    return logged.registered_model_version


def deploy_endpoint(model_version: str) -> None:
    """Creates or updates the Databricks endpoint."""
    client = WorkspaceClient(host=DATABRICKS_HOST, token=DATABRICKS_TOKEN)

    served_entities = [
        ServedEntityInput(
            entity_name=UC_MODEL_FQN,
            entity_version=model_version,
            workload_size="Small",
            scale_to_zero_enabled=True,
            environment_vars={"GOOGLE_API_KEY": "{{secrets/agent_secrets/google_api_key}}"},
        )
    ]

    existing_names = [e.name for e in client.serving_endpoints.list()]

    if ENDPOINT_NAME in existing_names:
        print(f"Endpoint '{ENDPOINT_NAME}' exists — updating to model version {model_version}")
        client.serving_endpoints.update_config(
            name=ENDPOINT_NAME, served_entities=served_entities
        )
    else:
        print(f"Creating new endpoint '{ENDPOINT_NAME}' with model version {model_version}")
        client.serving_endpoints.create(
            name=ENDPOINT_NAME,
            config=EndpointCoreConfigInput(
                name=ENDPOINT_NAME,  # 🛑 THE FIX: The SDK now demands the name here too!
                served_entities=served_entities
            ),
        )
    print("Waiting for endpoint to become ready...")
    for _ in range(120): 
        state = client.serving_endpoints.get(ENDPOINT_NAME).state
        print(f"  ready={state.ready} config_update={state.config_update}")
        if str(state.ready) == "EndpointStateReady.READY":
            break
        time.sleep(15)
    else:
        raise TimeoutError("Endpoint did not become ready in time.")

    print(f"✅ '{ENDPOINT_NAME}' is live at {DATABRICKS_HOST}/serving-endpoints/{ENDPOINT_NAME}/invocations")


if __name__ == "__main__":
    version = log_and_register_model()
    deploy_endpoint(version)







# """
# CI/CD deploy step: packages the ADK agent with MLflow, registers it to
# Unity Catalog, and creates/updates a Databricks Model Serving endpoint.

# Required environment variables (set as GitHub Actions repo secrets and
# passed in via cicd.yml — never hardcode these):

#   DATABRICKS_HOST   e.g. https://adb-xxxxxxxxxxxxxxxx.xx.azuredatabricks.net
#   DATABRICKS_TOKEN  Databricks PAT with model registry + serving perms
#   GOOGLE_API_KEY    Gemini API key the deployed agent uses at inference time

# Optional (sensible defaults below):
#   UC_CATALOG, UC_SCHEMA, UC_MODEL_NAME, ENDPOINT_NAME, GITHUB_SHA
# """

# import os
# import sys
# import time
# from google.adk.messages import UserMessage
# import mlflow
# from databricks.sdk import WorkspaceClient
# from databricks.sdk.service.serving import (
#     EndpointCoreConfigInput,
#     ServedEntityInput,
# )



# sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
# from main_agent.agent import root_agent  # noqa: E402


# def _require_env(name: str) -> str:
#     value = os.environ.get(name)
#     if not value:
#         raise RuntimeError(f"Missing required environment variable: {name}")
#     return value


# DATABRICKS_HOST = _require_env("DATABRICKS_HOST")
# DATABRICKS_TOKEN = _require_env("DATABRICKS_TOKEN")
# GOOGLE_API_KEY = _require_env("GOOGLE_API_KEY")

# UC_CATALOG = os.environ.get("UC_CATALOG", "main")
# UC_SCHEMA = os.environ.get("UC_SCHEMA", "agents")
# UC_MODEL_NAME = os.environ.get("UC_MODEL_NAME", "mini_weather_agent")
# ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "mini-weather-agent-endpoint")
# GIT_SHA = os.environ.get("GITHUB_SHA", "local")

# UC_MODEL_FQN = f"{UC_CATALOG}.{UC_SCHEMA}.{UC_MODEL_NAME}"

# # mlflow/databricks-sdk auth via env vars (standard names, no code changes needed)
# os.environ["DATABRICKS_HOST"] = DATABRICKS_HOST
# os.environ["DATABRICKS_TOKEN"] = DATABRICKS_TOKEN


# # class _ADKAgentWrapper(mlflow.pyfunc.PythonModel):
# #     """Wraps the ADK agent as an MLflow pyfunc model so Databricks can serve it."""

# #     def load_context(self, context):
# #         from google.adk.runners import InMemoryRunner

# #         os.environ.setdefault("GOOGLE_API_KEY", GOOGLE_API_KEY)
# #         self.runner = InMemoryRunner(agent=root_agent)

# #     def predict(self, context, model_input):
# #         import asyncio

# #         if hasattr(model_input, "to_dict"):  # pandas DataFrame
# #             prompts = model_input["prompt"].tolist()
# #         elif isinstance(model_input, dict):
# #             prompts = model_input.get("prompt", [str(model_input)])
# #         else:
# #             prompts = [str(model_input)]

# #         async def _ask(prompt: str) -> str:
# #             session = await self.runner.session_service.create_session(
# #                 app_name=self.runner.app_name, user_id="ci_cd_user"
# #             )
# #             final_text = ""
# #             async for event in self.runner.run_async(
# #                 user_id="ci_cd_user",
# #                 session_id=session.id,
# #                 new_message=prompt,
# #             ):
# #                 if event.is_final_response() and event.content:
# #                     final_text = event.content.parts[0].text
# #             return final_text

# #         return [asyncio.run(_ask(p)) for p in prompts]


# # def log_and_register_model() -> str:
# #     """Logs the agent to MLflow and registers it in Unity Catalog. Returns the new model version."""
# #     mlflow.set_registry_uri("databricks-uc")
# #     mlflow.set_tracking_uri("databricks")

# #     mlflow.set_experiment("/Shared/mini_weather_agent_logs")

# #     with mlflow.start_run(run_name=f"mini_weather_agent_{GIT_SHA[:7]}") as run:
# #         mlflow.set_tag("git_sha", GIT_SHA)
# #         mlflow.set_tag("source", "github-actions-ci-cd")

# #         logged = mlflow.pyfunc.log_model(
# #             artifact_path="agent",
# #             python_model=_ADKAgentWrapper(),
# #             registered_model_name=UC_MODEL_FQN,
# #             pip_requirements=[
# #                 "google-adk>=2.6.0",
# #                 "mlflow",
# #                 "databricks-sdk",
# #             ],
# #         )
# #         print(f"Logged model in run {run.info.run_id}, version {logged.registered_model_version}")

# #     return logged.registered_model_version

# class _ADKAgentWrapper(mlflow.pyfunc.PythonModel):
#     """Wraps the ADK agent as an MLflow pyfunc model so Databricks can serve it."""

#     def load_context(self, context):
#         from google.adk.runners import InMemoryRunner
#         os.environ.setdefault("GOOGLE_API_KEY", GOOGLE_API_KEY)
#         self.runner = InMemoryRunner(agent=root_agent)

#     def predict(self, context, model_input):
#         import asyncio

#         if hasattr(model_input, "to_dict"):
#             prompts = model_input["prompt"].tolist()
#         elif isinstance(model_input, dict):
#             prompts = model_input.get("prompt", [str(model_input)])
#         else:
#             prompts = [str(model_input)]

#         async def _ask(prompt: str) -> str:
#             session = await self.runner.session_service.create_session(
#                 app_name=self.runner.app_name, user_id="ci_cd_user"
#             )
#             final_text = ""
            
#             # BUG FIX: Wrap the prompt in UserMessage for ADK 2.0!
#             async for event in self.runner.run_async(
#                 user_id="ci_cd_user",
#                 session_id=session.id,
#                 new_message=UserMessage(prompt), 
#             ):
#                 if hasattr(event, 'content') and event.content and hasattr(event.content, 'parts'):
#                     for part in event.content.parts:
#                         if hasattr(part, 'text') and part.text:
#                             final_text += part.text
#             return final_text

#         return [asyncio.run(_ask(p)) for p in prompts]

# from mlflow.models.signature import infer_signature

# def log_and_register_model() -> str:
#     """Logs the agent to MLflow and registers it in Unity Catalog."""
#     mlflow.set_registry_uri("databricks-uc")
#     mlflow.set_tracking_uri("databricks")
#     mlflow.set_experiment("/Shared/mini_weather_agent_logs")

#     # Give MLflow a dummy input so it can build the Signature automatically
#     input_example = {"prompt": "What is the weather in Chennai?"}

#     with mlflow.start_run(run_name=f"mini_weather_agent_{GIT_SHA[:7]}") as run:
#         mlflow.set_tag("git_sha", GIT_SHA)
#         mlflow.set_tag("source", "github-actions-ci-cd")

#         logged = mlflow.pyfunc.log_model(
#             artifact_path="agent",
#             python_model=_ADKAgentWrapper(),
#             registered_model_name=UC_MODEL_FQN,
#             input_example=input_example,  # <--- THE FIX
#             pip_requirements=[
#                 "google-adk>=2.6.0",
#                 "mlflow",
#                 "databricks-sdk",
#             ],
#         )
#         print(f"Logged model in run {run.info.run_id}, version {logged.registered_model_version}")

#     return logged.registered_model_version

# def deploy_endpoint(model_version: str) -> None:
#     """Creates the serving endpoint if new, otherwise rolls it forward to the new model version."""
#     client = WorkspaceClient(host=DATABRICKS_HOST, token=DATABRICKS_TOKEN)

#     served_entities = [
#         ServedEntityInput(
#             entity_name=UC_MODEL_FQN,
#             entity_version=model_version,
#             workload_size="Small",
#             scale_to_zero_enabled=True,
#             # Reference a Databricks secret scope rather than injecting the raw key.
#             # Create it once: databricks secrets create-scope agent_secrets
#             #                 databricks secrets put-secret agent_secrets google_api_key
#             environment_vars={"GOOGLE_API_KEY": "{{secrets/agent_secrets/google_api_key}}"},
#         )
#     ]

#     existing_names = [e.name for e in client.serving_endpoints.list()]

#     if ENDPOINT_NAME in existing_names:
#         print(f"Endpoint '{ENDPOINT_NAME}' exists — updating to model version {model_version}")
#         client.serving_endpoints.update_config(
#             name=ENDPOINT_NAME, served_entities=served_entities
#         )
#     else:
#         print(f"Creating new endpoint '{ENDPOINT_NAME}' with model version {model_version}")
#         client.serving_endpoints.create(
#             name=ENDPOINT_NAME,
#             config=EndpointCoreConfigInput(served_entities=served_entities),
#         )

#     print("Waiting for endpoint to become ready...")
#     for _ in range(40):  # ~10 min max
#         state = client.serving_endpoints.get(ENDPOINT_NAME).state
#         print(f"  ready={state.ready} config_update={state.config_update}")
#         if str(state.ready) == "EndpointStateReady.READY":
#             break
#         time.sleep(15)
#     else:
#         raise TimeoutError("Endpoint did not become ready in time.")

#     print(f"✅ '{ENDPOINT_NAME}' is live at {DATABRICKS_HOST}/serving-endpoints/{ENDPOINT_NAME}/invocations")


# if __name__ == "__main__":
#     version = log_and_register_model()
#     deploy_endpoint(version)
