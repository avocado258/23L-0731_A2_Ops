# VisionGuard

[![CI](https://github.com/avocado258/23L-0731_A2_Ops/actions/workflows/ci.yml/badge.svg)](https://github.com/avocado258/23L-0731_A2_Ops/actions/workflows/ci.yml)

VisionGuard is an end-to-end MLOps project that classifies an uploaded image as **AI-generated** or **real**. It covers the full lifecycle: data auditing and versioning, model training with experiment tracking, a model registry, a Flask web application with a database audit trail, containerisation, continuous integration, and a publicly hosted model API.

- **Author:** `Muhammad Hassaan Arshad` (roll number 23L-0731)
- **GitHub:** https://github.com/avocado258/23L-0731_A2_Ops
- **DagsHub (experiments, registry, data):** https://dagshub.com/avocado258/23L-0731_A2_Ops
- **Live model API:** `https://two3l-0731-a2-ops.onrender.com` (requires an API key; see [Model API](#model-api))

---

## Architecture

```
┌───────────┐      ┌──────────────────────────────┐       HTTPS + X-API-Key       ┌──────────────────────────┐
│  Browser  │ ───► │  Flask UI  (docker compose)  │ ────────────────────────────► │  Model API  (Render)     │
└───────────┘      │  http://localhost:5000       │ ◄──────────────────────────── │  api_app.py + champion   │
                   └──────────────┬───────────────┘        JSON verdict           │  Keras model             │
                                  │ SQLAlchemy                                    └──────────────────────────┘
                                  ▼
                   ┌──────────────────────────────┐
                   │  PostgreSQL (db service)     │   named volume, no published port
                   └──────────────────────────────┘
```

The Flask UI and the database run locally through Docker Compose. Predictions come from the model API hosted in the cloud, and every result is written to the local database so it appears in the History page. If `INFERENCE_API_URL` is not set, the UI falls back to running the model in-process.

## The model

| Item | Value |
|---|---|
| Backbone | EfficientNetB0 (ImageNet weights), fine-tuned with 20 unfrozen layers |
| Training run | `cmp_efficientnet` (lr 1e-4, augmentation on) |
| Input | 224 × 224 RGB, `tf.keras.applications.efficientnet.preprocess_input` |
| Output | Single sigmoid unit, probability of the **real** class |
| Labels | `AiArtData = 0`, `RealArt = 1` (alphabetical class order) |
| Decision threshold | 0.5378, chosen with Youden's J on the validation set |
| Validation AUC | 0.8755 as logged in training; 0.8722 when re-evaluated through the serving preprocessing |
| Registry | MLflow Model Registry on DagsHub: `visionguard-model`, alias `champion` |

### Dataset

Trained on [AI Generated Images vs Real Images](https://www.kaggle.com/datasets/cashbowman/ai-generated-images-vs-real-images) from Kaggle, which has two classes: `AiArtData` (AI-generated) and `RealArt` (real). The raw images are versioned with DVC, and the train, validation, and test splits are stored as CSV files in `data/`.

### Metrics

Final-epoch metrics logged to MLflow for the champion run (`cmp_efficientnet`):

| Metric | Training | Validation |
|---|---|---|
| Accuracy | 0.9485 | 0.7877 |
| AUC | 0.9909 | 0.8755 |
| Loss | 0.2003 | 0.4441 |
| Precision | 0.9529 | 0.7576 |
| Recall | 0.9309 | not logged |

The gap between training and validation scores shows the model overfits the training data. Expect real-world accuracy closer to the validation figures than the training figures.

The model is an estimate, not proof. Confidence is the model's score for the predicted class and is not a calibrated probability.

## Repository layout

```
.
├── app/                    Flask web application
│   ├── __init__.py         app factory, routes, validation, error pages
│   ├── config.py           all settings from environment variables
│   ├── db.py               SQLAlchemy model for the prediction history
│   ├── inference.py        model loading (once at startup), preprocessing, prediction
│   ├── remote.py           client for the hosted model API
│   ├── templates/          HTML templates (upload, result, history, error)
│   └── static/style.css
├── api_app.py              standalone model API (JSON) deployed to the cloud
├── run.py                  entry point for the Flask UI
├── scripts/
│   ├── fetch_model.py      download the champion model from the MLflow registry
│   └── select_threshold.py choose the decision threshold on the validation set
├── training/               data pipeline, model definitions, run comparison,
│                           evaluation, error analysis, registry scripts (Tasks 1-3)
├── models/champion/        model.keras + meta.json (version, labels, threshold)
├── tests/                  pytest suite
├── Dockerfile              Flask UI image
├── Dockerfile.api          model API image
├── docker-compose.yml      web + db stack
├── requirements.txt        full development environment (training, MLflow, DVC)
├── requirements_serve.txt  minimal serving dependencies (used by the Docker images)
├── requirements-dev.txt    ruff and pytest
├── data/                   CSV splits and audit (raw images are tracked with DVC)
└── .github/workflows/ci.yml
```

## Quick start

### Prerequisites

- Docker with Docker Compose
- Python 3.13 (only needed for the helper scripts and running without Docker)

### 1. Configure the environment

Create an untracked `.env` file in the project root. `.env.example` lists every key.

```dotenv
# Database (used by docker compose)
POSTGRES_USER=visionguard
POSTGRES_PASSWORD=change_me
POSTGRES_DB=visionguard

# Hosted model API (leave empty to run the model inside the web container)
INFERENCE_API_URL=https://<your-api-host>
INFERENCE_API_KEY=<the API key>

# Only needed to fetch the model from the MLflow registry
MLFLOW_TRACKING_URI=https://dagshub.com/avocado258/23L-0731_A2_Ops.mlflow
MLFLOW_TRACKING_USERNAME=<dagshub username>
MLFLOW_TRACKING_PASSWORD=<dagshub token>
```

Use letters and digits only in `POSTGRES_PASSWORD`, because it is placed inside a connection URL.

### 2. Get the model (only for local inference or building the images)

```bash
python scripts/fetch_model.py      # downloads models/champion/model.keras from the registry
python scripts/select_threshold.py # writes the validation threshold into meta.json
```

`models/champion/meta.json` is committed, so you only need the first script unless the champion changes. If it does, rerun both.

### 3. Start the stack

```bash
docker compose up -d --build
docker compose ps          # both services should become healthy
```

Open http://localhost:5000, upload an image, and view results under **History**.

### Running without Docker

```bash
pip install -r requirements_serve.txt
python run.py              # uses SQLite (instance/history.db) when DATABASE_URL is not set
```

## Configuration

All settings come from environment variables.

| Variable | Used by | Default | Purpose |
|---|---|---|---|
| `MODEL_DIR` | UI, API | `models/champion` | Folder with `model.keras` and `meta.json` |
| `THRESHOLD` | UI, API | value in `meta.json` | Overrides the decision threshold |
| `DATABASE_URL` | UI | `sqlite:///history.db` | Database connection (Compose sets the Postgres URL) |
| `MAX_UPLOAD_BYTES` | UI, API | 5 MB | Maximum upload size |
| `INFERENCE_API_URL` | UI | empty | Base URL of the hosted model API; empty means local model |
| `INFERENCE_API_KEY` | UI | empty | Key sent in the `X-API-Key` header |
| `INFERENCE_API_TIMEOUT` | UI | 120 s | Request timeout, long enough for a cold start |
| `API_KEY` | API | none | Key the model API requires on `/predict` |
| `PORT` | API | 10000 | Port the model API listens on |

## Web application

| Route | Method | Description |
|---|---|---|
| `/` | GET | Upload page |
| `/predict` | POST | Validates the image, gets a verdict, stores the result, shows it |
| `/history` | GET | Table of past predictions |
| `/health` | GET | Returns 200 when the model and database are reachable, 503 otherwise |

Validation and safety:

- Wrong file types (only png, jpg, jpeg, webp), oversized files, and files that cannot be decoded are rejected with friendly error pages (415, 413, 422).
- File names are sanitised with `secure_filename`.
- Uploaded images are processed in memory and are not saved on the server.
- The app runs with `debug=False`.

## Database

Every prediction is logged to a `predictions` table:

| Column | Meaning |
|---|---|
| `id` | Primary key |
| `created_at` | UTC timestamp |
| `filename` | Sanitised original name (the image itself is never stored) |
| `label` | AI-generated or Real |
| `confidence` | Probability of the predicted class |
| `p_real` | Raw probability of the real class |
| `threshold` | Decision threshold used |
| `model_name` | Model name and version that produced the result |

In Compose, PostgreSQL runs in its own `db` container on an internal network, stores data in the named volume `pgdata`, publishes no port to the host, and gates the web service start through a health check.

## Model API

`api_app.py` is a small JSON service built on the same inference module and deployed from `Dockerfile.api`.

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | `{"status": "ok", "model": "..."}` |
| `/predict` | POST | multipart form-data with an `image` file and an `X-API-Key` header |

```bash
curl -i -H "X-API-Key: <key>" -F "image=@photo.jpg" https://<your-api-host>/predict
```

Successful response:

```json
{
  "label": "Real",
  "confidence": 0.91,
  "p_real": 0.91,
  "threshold": 0.5378,
  "model": "visionguard-model:v1"
}
```

| Status | Meaning |
|---|---|
| 200 | Prediction returned |
| 400 | No file in the `image` form field |
| 401 | Missing or wrong API key |
| 413 | File over the size limit |
| 415 | Unsupported file type |
| 422 | File could not be decoded as an image |

## Deployment (Render)

The model API is deployed as a Docker web service on [Render](https://render.com), built straight from this GitHub repository.

| Setting | Value |
|---|---|
| Runtime | Docker, Dockerfile path `./Dockerfile.api` |
| Environment variable | `API_KEY` (the secret the API checks on `/predict`) |
| Health check path | `/health` |
| Listening port | Render's `PORT` variable (the app reads it) |

Notes:

- **Model in the image:** Render builds from git, and the DagsHub token must not enter a Docker build, so `models/champion/model.keras` (about 28 MB) is committed to the repository with `git add -f`, since `*.keras` is otherwise git-ignored. It is the registry's `champion` model, version 1.
- **Free instance type:** Render's free web services have 512 MB of RAM and 0.1 CPU, and they spin down after 15 minutes without traffic. The first request after an idle period takes about a minute while the service restarts and TensorFlow loads the model. Paid instance types do not spin down.
- **Memory:** the container limits TensorFlow to one thread (`TF_NUM_INTRAOP_THREADS`, `TF_NUM_INTEROP_THREADS`) to keep memory use low. If the service is killed for exceeding memory, use a larger instance type.
- **Updating the key:** change `API_KEY` in the service's Environment tab (Render redeploys), then update `INFERENCE_API_KEY` in your local `.env` and run `docker compose up -d`.
- **Flask UI:** it is not deployed; it runs locally and calls this API.

## Docker

- **Slim base** (`python:3.13-slim`) and **CPU-only TensorFlow** (`tensorflow-cpu`), since no GPU is needed for single-image inference.
- Dependencies are installed before the source code is copied, so code edits reuse the cached dependency layer.
- The app runs as a **non-root user**, has a **health check**, and `.dockerignore` keeps `.env`, data, and training artifacts out of the image.
- No secrets are baked into any image. Passwords and keys arrive as environment variables at runtime.
- Size analysis: the slim image is 2.32 GB against 3.76 GB for a build on the full `python:3.13` base (471 MB against 842 MB compressed). TensorFlow dominates the remaining size.

Useful commands:

```bash
docker compose up -d --build     # start
docker compose logs -f web       # follow logs
docker compose exec web sh       # shell in the web container
docker compose down              # stop (keeps the database volume)
docker compose down -v           # stop and delete the database volume
```

## Continuous integration

`.github/workflows/ci.yml` runs on every push and pull request:

1. Install dependencies
2. Lint with ruff
3. Run the pytest suite
4. Fetch the champion model from the MLflow registry
5. Build the Docker image

Add these as repository secrets (Settings, Secrets and variables, Actions): `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`. They are never written in the workflow file.

## Development

```bash
pip install -r requirements_serve.txt -r requirements-dev.txt
ruff check app tests run.py api_app.py
pytest -v
```

The tests use a stub predictor, so they run without the model file or network access, but they still exercise the real image decoding and preprocessing code and every route and error path.

## Experiment tracking and data versioning

- Experiments, metrics, artifacts, and the model registry live on DagsHub through MLflow.
- Raw images are versioned with DVC (`data/raw.dvc`); the CSV splits (`train.csv`, `val.csv`, `test.csv`, `audit.csv`) are tracked in git.
- To fetch the data: `dvc pull` (needs DagsHub credentials in `.env`).

## Security notes

- `.env` is git-ignored. Never commit credentials.
- The model API requires a secret key on `/predict`, compared in constant time.
- Uploaded images are never written to disk or stored in the database.

## Limitations

VisionGuard is a coursework project. Validation accuracy is about 79% and validation AUC about 0.88, and the large gap to the training scores shows overfitting, so it will misclassify many images. Its accuracy on images from generators, styles, or compression levels unlike the training data is unknown. Treat verdicts as one signal, not evidence of origin.