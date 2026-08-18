# enterprise_adk_pipeline (mini)

A minimal CI/CD pipeline: a Google ADK agent gets tested and deployed to a
Databricks Model Serving endpoint automatically on every push to `main`.

```
push to GitHub → GitHub Actions runs pytest → on success, deploy job
logs the agent with MLflow → registers to Unity Catalog → creates/updates
a Databricks Serving endpoint
```

## 1. Local setup

```bash
cd enterprise_adk_pipeline
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp src/main_agent/.env.example src/main_agent/.env   # then fill in your real key
pytest tests/ -v
```

## 2. GitHub repo secrets (Settings → Secrets and variables → Actions)

| Secret name        | Value                                                |
|---------------------|-------------------------------------------------------|
| `DATABRICKS_HOST`   | e.g. `https://adb-xxxxxxxxxxxxxxxx.xx.azuredatabricks.net` |
| `DATABRICKS_TOKEN`  | Databricks personal access token (Unity Catalog + Serving perms) |
| `GOOGLE_API_KEY`    | Your Gemini API key                                   |

## 3. Databricks one-time setup

```bash
databricks secrets create-scope agent_secrets
databricks secrets put-secret agent_secrets google_api_key
```
The serving endpoint reads the key from this secret scope rather than the
raw env var, so it's never stored in plaintext on the endpoint config.

## 4. Ship it

```bash
git push origin main
```
Watch the run under the **Actions** tab. On success, check
`{DATABRICKS_HOST}/serving-endpoints/mini-weather-agent-endpoint`.

## Growing this later
- Swap `get_weather` for real tools / add sub-agents.
- Add an `integration` pytest marker that hits live Gemini, run it only on
  a manual `workflow_dispatch` trigger (keeps normal PR runs free/fast).
- Split `deploy` into `staging` and `prod` jobs gated by environments.
