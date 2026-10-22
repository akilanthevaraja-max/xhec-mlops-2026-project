<div align="center">

# MLOps Project: Abalone Age Prediction

[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)]()
[![Linting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-informational?logo=pre-commit&logoColor=white)](https://github.com/artefactory/xhec-mlops-project-student/blob/main/.pre-commit-config.yaml)
</div>

## 🎯 Project Overview

Welcome to your MLOps project! In this hands-on project, you'll build a complete machine learning system to predict the age of abalone (a type of sea snail) using physical measurements instead of the traditional time-consuming method of counting shell rings under a microscope.

**Your Mission**: Transform a simple ML model into a production-ready system with automated training, deployment, and prediction capabilities.

## 📊 About the Dataset

Traditionally, determining an abalone's age requires:
1. Cutting the shell through the cone
2. Staining it
3. Counting rings under a microscope (very time-consuming!)

**Your Goal**: Use easier-to-obtain physical measurements (shell weight, diameter, etc.) to predict the age automatically.

📥 **Download**: Get the dataset from the [Kaggle page](https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset)


## 🚀 Quick Start

### Prerequisites
- GitHub account
- [Kaggle account](https://www.kaggle.com/account/login?phase=startRegisterTab&returnUrl=%2F) (for dataset download)
- Python 3.11

### Setup Steps

1. **Fork this repository**
   - ⚠️ **Important**: Uncheck "Copy the `main` branch only" to get all project branches

2. **Add your team members** as admins to your forked repository

3. **Set up your development environment**:
   ```bash
   # Create and activate a virtual environment
   uv sync
   source .venv/bin/activate # on Windows: .venv\Scripts\activate

   # Install pre-commit hooks for code quality
   uv run pre-commit install
   ```

## 📋 What You'll Build

By the end of this project, you'll have created:

### 🤖 **Automated ML Pipeline**
- Training workflows using Prefect
- Automatic model retraining on schedule
- Reproducible model and data processing

### 🌐 **Prediction API**
- REST API for real-time predictions
- Input validation with Pydantic
- Docker containerization

### 📊 **Production-Ready Code**
- Clean, well-documented code
- Automated testing and formatting
- Proper error handling

## 📝 How to Work on This Project

### The Branch-by-Branch Approach

This project is organized into numbered branches, each representing a step in building your MLOps system. Think of it like a guided tutorial where each branch teaches you something new!

**Here's how it works**:

1. **Each branch = One pull request** with specific tasks
2. **Follow the numbers** (branch_0, branch_1, etc.) in order
3. **Read the PR instructions** (PR_0.md, PR_1.md, etc.) before starting
4. **Complete all TODOs** in that branch's code
5. **Create a pull request** when done
6. **Merge and move to the next branch**

### Step-by-Step Workflow

For each numbered branch:

```bash
# Switch to the branch
git checkout branch_number_i

# Get latest changes (except for branch_0)
git pull origin main
# Note: A VIM window might open - just type ":wq" to close it

# Push your branch
git push
```

Then:
1. 📖 Read the PR_i.md file carefully
2. 💻 Complete all the TODOs in the code
3. 🔧 Test your changes
4. 📤 Open **ONE** pull request to your main branch
5. ✅ Merge the pull request
6. 🔄 Move to the next branch

> **💡 Pro Tip**: Always integrate your previous work when starting a new branch (except branch_0)!

### 🔍 Understanding Pull Requests

Pull Requests (PRs) are how you propose and review changes before merging them into your main codebase. They're essential for team collaboration!

**Important**: When creating a PR, make sure you're merging into YOUR forked repository, not the original:

❌ **Wrong** (merging to original repo):
![PR Wrong](assets/PR_wrong.png)

✅ **Correct** (merging to your fork):
![PR Right](assets/PR_right.png)

## 💡 Development Tips

### Managing Dependencies

Use uv to manage dependencies. Install or update packages with:

```bash
uv add <package>==<version>
```

Then sync the environment and regenerate the dependency files:

```bash
uv sync
```

### Code Quality
- The pre-commit hooks will automatically format your code
- Remove all TODOs and unused code before final submission
- Use clear variable names and add docstrings

## 📊 Evaluation Criteria

Your project will be evaluated on:

### 🔍 **Code Quality**
- Clean, readable code structure
- Proper naming conventions
- Good use of docstrings and type hints

### 🎨 **Code Formatting**
- Consistent style (automated with pre-commit)
- Professional presentation

### ⚙️ **Functionality**
- Code runs without errors
- All requirements implemented correctly

### 📖 **Documentation & Reproducibility**
- Clear README with setup instructions
- Team member names and GitHub usernames
- Step-by-step instructions to run everything

### 🤝 **Collaboration**
- Effective use of Pull Requests
- Good teamwork and communication

---

## 🎯 Final Deliverables Checklist

When you're done, your repository should contain:

✅ **Automated Training Pipeline**
- [x] Prefect workflows for model training
- [x] Separate modules for training and inference
- [x] Reproducible model and encoder generation

✅ **Automated Deployment**
- [x] Prefect deployment for regular retraining

✅ **Production API**
- [x] Working REST API for predictions
- [x] Pydantic input validation
- [x] Docker containerization

✅ **Professional Documentation**
- [x] Updated README with team info
- [x] Clear setup and run instructions
- [x] All TODOs removed from code

---

**Ready to start? Head to branch_0 and read PR_0.md for your first task! 🚀**


## Team

| Name | GitHub |
| --- | --- |
| Akilan Thevaraja | akilanthevaraja-max |
| Hae In KEUM | conni2 |
| Nicolas Trancart | nicotranfr |
| Alexandre Le Bacon | Alexandre-Le-Bacon |

## Reproduce stage 1: EDA and modelling

From the repository root, use Python 3.11 and the committed uv lockfile:

```bash
uv sync --locked
uv run pre-commit install
uv run jupyter lab
```

Open and run `notebooks/eda.ipynb`, then `notebooks/modelling.ipynb` from top to bottom. Both notebooks resolve the dataset from the repository root or the notebooks directory. The current team branch includes `data/abalone.csv`; the notebooks verify its schema. The modelling run also records its SHA256 hash.

## Reproduce stage 2: modular training and inference

The reusable implementation lives in `src/modelling`. It validates the input schema,
fits categorical encoding and linear regression in one sklearn pipeline, evaluates the
recipe on a fixed holdout, refits on all observations, and serializes the complete
pipeline for later API use.

The pickle contains `pipeline` and `metadata`. `metadata.production_training_rows`
counts the observations used to fit the saved pipeline. The separate
`metadata.evaluation` block records the holdout model's training/test row counts,
metrics, and model scope. Those scores describe the evaluation model before the
full-data refit, not an independent test of the saved full-data pipeline. The
fixed split reproduces the stage-1 evaluation rather than introducing a new test set.

Train from the repository root:

```bash
uv sync --locked
uv run python -m modelling.main data/abalone.csv
```

The default artifact is written to
`src/web_service/local_objects/abalone_pipeline.pkl`. Pickle artifacts are ignored by
Git because they must be regenerated from the versioned code and data. Only load a
pickle file produced by a trusted source.

Use the saved artifact from Python:

```python
from modelling.predicting import predict_from_artifact

observation = {
    "Sex": "M",
    "Length": 0.455,
    "Diameter": 0.365,
    "Height": 0.095,
    "Whole weight": 0.514,
    "Shucked weight": 0.2245,
    "Viscera weight": 0.101,
    "Shell weight": 0.150,
}

prediction = predict_from_artifact(
    "src/web_service/local_objects/abalone_pipeline.pkl",
    observation,
)
print(prediction[0])
```

Run the automated checks:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run pre-commit run --all-files --show-diff-on-failure
```

The modelling notebook predicts `Rings` from the eight original predictors. It compares a mean baseline, linear regression, and Ridge (`alpha=1`) using a fixed train/validation/test split. Model selection uses validation RMSE; only the selected model is refitted and evaluated on the final test split. It logs parameters, metrics, and fitted sklearn pipelines to local `mlruns/`, then verifies model reload predictions. `Age` is excluded from all model inputs.

To inspect experiments, run in a separate terminal from the repository root:

```bash
uv run mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5002
```

Open http://127.0.0.1:5002 and select `abalone-rings-notebook`. Three validation runs and one final-test run are created per execution. `mlruns/` and `work/` are ignored; do not commit experiment artifacts or fitted models. Notebook outputs are stripped by pre-commit, so rerun locally to view figures and results. Local execution results are also written to `work/modelling_results.json`.

For a headless notebook run from the repository root:

```bash
mkdir -p work
uv run jupyter nbconvert --to notebook --execute notebooks/eda.ipynb --output eda.executed.ipynb --output-dir work
uv run jupyter nbconvert --to notebook --execute notebooks/modelling.ipynb --output modelling.executed.ipynb --output-dir work
uv run pre-commit run --all-files
```

Stage 1 does not implement the later stages' Python-module training CLI, pickle export, Prefect deployment, or FastAPI/Docker service.

## Stage 3: Prefect training, inference, and scheduled retraining

The implementation follows the course's `@task`, `@flow`, `to_deployment`, and
`serve` pattern. It reuses the stage-2 linear pipeline, validation, and pickle
format without changing the modelling recipe. Training and inference are separate
flow modules:

- `modelling.workflows`: load/validate, evaluate/refit, and save tasks in the training flow.
- `modelling.inference`: a separate inference flow using the saved pipeline.
- `modelling.deployment`: weekly training deployment, Sunday 00:00 in `Europe/Paris`.

Prefect 2.13.7 is retained from the course environment. `cachetools` is explicitly
included because this Prefect server version imports it. SQLAlchemy is pinned to
the course-tested 2.0.43 because 2.1 fails while generating scheduled runs for
this Prefect version. Dependencies remain
managed by the team's `pyproject.toml` and `uv.lock`, not separate requirements files.

### Start the local server

Run all commands from the repository root. Use the following environment variables
in **each terminal** used below. Port 4202 avoids colliding with other course servers;
keep the API URL and server port consistent if you choose another port.

```bash
uv sync --locked
mkdir -p work/prefect-stage3
export PREFECT_HOME="$PWD/work/prefect-stage3"
export PREFECT_API_URL="http://127.0.0.1:4202/api"
uv run prefect server start --host 127.0.0.1 --port 4202
```

Keep this terminal running. Open http://127.0.0.1:4202 for the UI. Its local SQLite
metadata and logs live under ignored `work/prefect-stage3/`.

### Run training and inference directly

In another terminal, export the same two variables, then run:

```bash
uv run python -m modelling.workflows data/abalone.csv
```

The training flow saves `src/web_service/local_objects/abalone_pipeline.pkl`.
Its `production_training_rows` describes the saved full-data model; `evaluation`
records the separate fixed-holdout model's scores. The split is reused from stage 1,
so these are reproducibility checks rather than a new independent test.

For an inference example, create a JSON list of observations with exactly the eight
predictors (no `Rings` or derived `Age`):

```bash
mkdir -p work
uv run python -c 'import pandas as pd; from pathlib import Path; df = pd.read_csv("data/abalone.csv"); Path("work/observations.json").write_text(df.drop(columns="Rings").head(3).to_json(orient="records"))'
uv run python -m modelling.inference src/web_service/local_objects/abalone_pipeline.pkl work/observations.json
```

The example selects existing rows only to verify the prediction interface; it is
not an unseen-data evaluation. Real inference inputs must use the same schema.
View both flow runs, task states, and logs under **Flow Runs** in the Prefect UI.

### Serve and test the scheduled deployment

In a terminal with the same variables, run:

```bash
uv run python -m modelling.deployment
```

This registers `Abalone model training/weekly-retraining` and keeps a local runner
listening for work. In the UI, confirm the weekly schedule and `Europe/Paris`
timezone. To test immediately without waiting until Sunday, use a third terminal
with the same variables:

```bash
uv run prefect deployment run 'Abalone model training/weekly-retraining' \
  --param output_path="$PWD/src/web_service/local_objects/abalone_pipeline.pkl"
```

The `--param` is a workaround: with Prefect 2.13.7 and the installed `typer`, the CLI
crashes with `TypeError: 'NoneType' object is not iterable` when no `--param` is given.
You can also trigger the run from the UI (**Deployments** > **Run**).

The CLI creates a run; confirm it reaches **Completed**, its tasks finish, and the
artifact is updated. This tests the registered deployment and runner, not just a
function call. A successful manual run does not prove that a weekly scheduled run
has already happened. Check the active schedule and upcoming scheduled runs
separately in the UI.

For an optional short scheduling exercise, the same runner accepts another cron:

```bash
uv run python -m modelling.deployment --cron '*/2 * * * *'
```

Run only one serving process for this deployment at a time. Stop the short-interval
runner and restart the default command to restore weekly retraining. The default
remains the course's weekly schedule. The server and runner both need to remain
running for local automation; closing the runner normally pauses its deployment.
This is a local learning setup, not a hosted always-on service.

### Checks

```bash
uv run pytest
uv run pre-commit run --all-files
```

Automated tests validate the weekly schedule configuration without starting a
server. Actual flow execution and deployment triggering require the live server
and runner described above. Generated models, database files, and runtime logs
are ignored by Git. FastAPI and Docker service delivery remain stage 4.

## Stage 4: FastAPI prediction service and Docker

The service lives in `src/web_service`:

- `lib/models.py`: Pydantic input (`AbaloneInput`) and output (`AbaloneOutput`) models. The input uses the
  dataset's column names (`Sex`, `Length`, `Whole weight`, ...) as JSON keys; `Sex` must be `F`, `I` or `M`
  and the seven measurements must be `>= 0`.
- `lib/inference.py`: runs `modelling.predicting.predict` on one validated observation.
- `utils.py`: `load_object`, which loads the trusted training pickle.
- `main.py`: `GET /` (health check) and `POST /predict`. The pipeline is loaded once at startup from
  `src/web_service/local_objects/abalone_pipeline.pkl` (override with the `MODEL_PATH` environment variable).
  If the file is missing, `/predict` answers `503`; invalid input answers `422`.

### Run the API locally

From the repository root, train the model first (it writes the ignored pickle), then start the API:

```bash
uv sync --locked
uv run python -m modelling.main data/abalone.csv
uv run uvicorn src.web_service.main:app --port 8001
```

Open http://127.0.0.1:8001/docs for the interactive documentation, or send a request:

```bash
curl -X POST http://127.0.0.1:8001/predict -H 'Content-Type: application/json' -d '{
  "Sex": "M", "Length": 0.455, "Diameter": 0.365, "Height": 0.095,
  "Whole weight": 0.514, "Shucked weight": 0.2245, "Viscera weight": 0.101, "Shell weight": 0.15
}'
# {"prediction": 9.17...}
```

`prediction` is the predicted number of rings (the age in years is `Rings + 1.5`).

### Run the API in Docker

`Dockerfile.app` installs the locked runtime dependencies with `uv`, trains the model while building the image
(the pickle is not versioned), and starts `bin/run_services.sh`, which runs the Prefect server (port 4201 in the
container) and the API (port 8001 in the container).

```bash
docker build -f Dockerfile.app -t abalone-app .
docker run --rm -p 0.0.0.0:8000:8001 -p 0.0.0.0:4200:4201 abalone-app
```

- API: http://localhost:8000 (docs at http://localhost:8000/docs)
- Prefect UI: http://localhost:4200

```bash
curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' -d '{
  "Sex": "M", "Length": 0.455, "Diameter": 0.365, "Height": 0.095,
  "Whole weight": 0.514, "Shucked weight": 0.2245, "Viscera weight": 0.101, "Shell weight": 0.15
}'
```

The Prefect server logs a harmless `database is locked` telemetry error while starting. Dependencies are managed by
`pyproject.toml` and `uv.lock` (no separate `requirements.in`/`requirements.txt`).
