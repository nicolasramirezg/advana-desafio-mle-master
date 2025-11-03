# Challenge – Software Engineer (ML & LLMs)

This document describes the process I followed to complete the Software Engineer (ML & LLMs) challenge.
The objective was to productionize a Data Scientist's flight-delay model using proper engineering practices clean code, containerization, and automated CI/CD.

I focused on:
- Reproducibility and deterministic builds using uv instead of traditional Makefiles
- FastAPI + Cloud Run for scalable serving
- GitHub Actions for full CI/CD automation

## Part I — Model Operationalization

### 1. Refactoring and Structure

I converted the DS's notebook (training.ipynb) into a modular Python class (DelayModel) with methods:
- `preprocess()`
- `fit()`
- `predict()`

Each was refactored to follow PEP 8, include docstrings, and handle datetime and feature logic consistently.

### 2. Fixes and Stability Improvements

I fixed bugs in:
- Datetime parsing and timezone consistency
- Feature derivation (high_season, period_day, min_diff)
- Prevention of data leakage during preprocessing

### 3. Model Choice

I selected Logistic Regression as the production model because it offered:
- Interpretability and low inference latency
- Stable convergence and good recall/precision for this binary classification task
- Suitability for lightweight APIs

### 4. Dependency and Environment Management with uv

Instead of relying on Make commands, I adopted uv for dependency management and execution.

Reasons:
- Speed and determinism — uv uses a lockfile (uv.lock) to guarantee identical environments across local, Docker, and CI
- Simplicity — single-line environment recreation (uv sync --frozen) instead of manual virtual-env setup
- Lightweight reproducibility in cloud and CI pipelines

Example local run:
```bash
uv run pytest tests/model --cov=challenge --cov-report term
```
This replaced make model-test while achieving the same coverage validation.

## Part II — API Deployment with FastAPI (api.py)

### 1. Implementation

I implemented a /predict endpoint using FastAPI, exposing the trained model.
The endpoint accepts a JSON list of flights and returns predicted delays.

Example request:
```json
{"flights": [{"OPERA": "Aerolineas Argentinas", "TIPOVUELO": "N", "MES": 3}]}
```

Good practices applied:
- Pydantic for validation and clear input contracts
- Lazy model loading to optimize cold-start times
- Clean separation between request handling and ML logic

### 2. Local Validation

Ran locally with:
```bash
uv run uvicorn challenge.api:app --reload
```

and tested with:
```bash
curl -X POST "http://localhost:8080/predict" \
     -H "Content-Type: application/json" \
     -d '{"flights":[{"OPERA":"LATAM","TIPOVUELO":"N","MES":3}]}'
```

All responses matched expected predictions.

### 3. Automated Testing

Instead of make api-test, I used:
```bash
uv run pytest tests/api --cov=challenge --cov-report term
```
All tests passed successfully.

## Part III — Cloud Deployment (GCP + Cloud Run)

### 1. Containerization

I built a lightweight Docker image leveraging uv for reproducible installs:

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install uv && uv sync --frozen --all-extras
CMD ["uv", "run", "uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8080"]
```

Built and verified locally with:
```bash
docker build -t delay-api .
docker run -p 8080:8080 delay-api
```

### 2. Cloud Run Deployment

I deployed to Google Cloud Run using:
```bash
gcloud builds submit --tag gcr.io/flight-api-nicolramirez/delay-api
gcloud run deploy delay-api \
  --image gcr.io/flight-api-nicolramirez/delay-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

Successfully deployed at:
https://delay-api-447905421089.us-central1.run.app

### 3. Stress Testing

To validate scalability:
```bash
uv run locust -f tests/stress/test_stress_api.py \
  --host https://delay-api-447905421089.us-central1.run.app
```

The service sustained concurrent requests with stable latency.

## Part IV — CI/CD Implementation (GitHub Actions + GCP)

### 1. Continuous Integration (ci.yml)

The CI workflow:
- Checks out the repo
- Installs Python 3.12 and uv
- Syncs dependencies via uv.lock
- Ensures data/data.csv exists for tests
- Runs model + API tests with coverage

This provides deterministic testing for every push to develop or feature/*.

### 2. Continuous Deployment (cd.yml)

Triggered on merges into main, it:
- Authenticates with a GCP Service Account (GCP_SA_KEY secret)
- Builds and pushes the Docker image using Cloud Build
- Deploys the latest revision to Cloud Run

### 3. Validation

After merging:
- CI confirmed all tests passed
- CD automatically deployed the new container
- Verified API availability on the public endpoint

| Component      | Description                                                                                              |
| -------------- | -------------------------------------------------------------------------------------------------------- |
| **Model**      | Logistic Regression model refactored and tested                                                          |
| **API**        | FastAPI endpoint `/predict` for model inference                                                          |
| **Deployment** | [https://delay-api-447905421089.us-central1.run.app](https://delay-api-447905421089.us-central1.run.app) |
| **CI/CD**      | GitHub Actions workflows for automated testing and deployment                                            |
| **Repository** | [https://github.com/nicolramirez/advana-desafio-mle-master](#)                                           |