# MLOps Project: Abalone Rings Prediction

Predict the number of abalone shell rings from physical measurements, then develop a reproducible training workflow and a containerized prediction API. Rings are the supervised target used by this assignment as an indicator of age.

## Team

| Name | GitHub username |
| --- | --- |
| Akilan Thevaraja | [akilanthevaraja-max](https://github.com/akilanthevaraja-max) |
| Hae In KEUM | [conni2](https://github.com/conni2) |
| Nicolas Trancart | [nicotranfr](https://github.com/nicotranfr) |
| Alexandre Le Bacon | [Alexandre-Le-Bacon](https://github.com/Alexandre-Le-Bacon) |

Team repository: [xhec-mlops-2026-project](https://github.com/akilanthevaraja-max/xhec-mlops-2026-project).

## Project stages and pull requests

Each numbered branch corresponds to one pull request into **this team repository's `main` branch**, not the original course template repository.

| Stage | Branch | Deliverable |
| --- | --- | --- |
| 0 | `0/environment_setup` | Python environment, dependencies, quality hooks, and CI |
| 1 | `1/eda_and_modelling_on_notebooks` | Dataset exploration and baseline modelling |
| 2 | `2/from_notebooks_to_modules` | Reusable preprocessing, training, and inference modules |
| 3 | `3/use_prefect` | Prefect tasks, flows, and scheduled retraining |
| 4 | `4/deploy_model_api` | Pydantic validation, FastAPI inference, and Docker |

The current implementation covers environment setup only. Training notebooks, modelling modules, scheduled workflows, and the prediction API belong to subsequent stages.

Before starting a later stage, read its `PR_i.md` instructions and integrate the previous work merged into `main`. The branch names above are the actual remote branch names. References to `master` in older task templates should be read as `main` for this repository. Do not merge the previous-stage branch directly before its PR has been reviewed and merged into `main`.

For example, when starting stage 1 after PR 0 is merged:

```bash
git fetch origin
git switch --track origin/1/eda_and_modelling_on_notebooks
git merge origin/main
```

Review the diff, run the checks, commit, push the branch, and open one PR to the team's `main`. Use English for code comments, documentation, commit messages, and pull requests.

## Course conventions

This project follows the coding best-practices activity in [the course repository](https://github.com/artefactory/xhec-mlops-2025):

- Python 3.11 and a project-local `.venv` managed with uv.
- `pyproject.toml` and a committed `uv.lock` for the reproducible environment.
- Ruff for linting, formatting, and import sorting, using the course rule families, Google-style docstrings, and a 130-character line length.
- Pre-commit hooks and GitHub Actions for repeatable checks.
- The course's Prefect 2.13.7, Pydantic 1, FastAPI 0.103.2, and scikit-learn 1.5.2 interfaces.

Ruff provides the formatting and import-ordering roles represented by Black and isort in the older stage-0 template. A second formatter or duplicate linter is intentionally not configured. NumPy is pinned to the compatible 1.26 line, `griffe` stays below 1, and HTTPX stays below 0.28 for this course stack. A small shared constraints file also retains the course-validated cryptography, SciPy, and IPython versions; this avoids a newer cryptography source build requiring additional system OpenSSL tooling on Intel macOS.

## Environment setup

Prerequisites:

- Git and access to this team repository.
- Python 3.11, or permission for uv to install the selected interpreter.
- [uv](https://docs.astral.sh/uv/getting-started/installation/). CI uses uv 0.12.21; use the same version when regenerating the lockfile.
- A Kaggle account if required to download the assignment dataset.

From a fresh clone:

```bash
git clone https://github.com/akilanthevaraja-max/xhec-mlops-2026-project.git
cd xhec-mlops-2026-project
git switch 0/environment_setup
uv sync --locked
uv run --locked pre-commit install
```

Activation is optional when using `uv run`. To activate manually:

```bash
# macOS / Linux
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Select `.venv/bin/python` as the VS Code interpreter and the notebook kernel on macOS/Linux. On Windows select `.venv\Scripts\python.exe`.

The unused conda template has been removed because uv is the selected environment manager. Stage 0 does not yet define an application package; the empty setuptools package list avoids referencing a `src` directory that has not been introduced yet.

## Dependency files

| File | Purpose |
| --- | --- |
| `pyproject.toml` | Direct application dependencies, development group, Python version, and tool configuration |
| `uv.lock` | Resolved environment used by `uv sync --locked` and CI |
| `requirements.in` | Direct application requirements required by the assignment |
| `requirements-dev.in` | Development requirements plus the application input file |
| `constraints-course.txt` | Course-validated transitive dependency constraints shared by uv and pip-compile |
| `requirements.txt` | Compiled, hashed application requirements |
| `requirements-dev.txt` | Compiled, hashed application and development requirements |

The direct requirements in the `.in` files mirror `pyproject.toml`. `scripts/check_dependencies.py` rejects drift between these inputs. Keep both declarations in sync when adding a dependency.

To change an application dependency:

```bash
uv add 'package==version'
# Add the same direct requirement to requirements.in.
uv run --locked python scripts/check_dependencies.py
uv pip compile --universal --python-version 3.11 --generate-hashes --output-file requirements.txt requirements.in
uv pip compile --universal --python-version 3.11 --generate-hashes --output-file requirements-dev.txt requirements-dev.in
uv sync --locked
```

For a development dependency use `uv add --dev 'package==version'` and update `requirements-dev.in`. The course pip-tools compiler remains available. The checked-in `.txt` files use the equivalent `uv pip compile --universal` command so platform markers are retained instead of freezing only the current Mac environment. These files are a requirements-based installation alternative. The universal uv lockfile is the primary cross-platform environment definition used by CI. Do not copy `.venv` between teammates or mount a macOS `.venv` into a Linux container; create a new environment on each platform from the same lockfile.

## Local data and generated artifacts

Download the exact [assignment Kaggle dataset](https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset) and extract its CSV into `data/`. Keep the original column names, including `Rings`. The expected layout is:

```text
data/
├── .gitkeep
└── abalone.csv
```

Raw data, trained models, local work files, credentials, and virtual environments are not committed. Only the directory markers are tracked under `data/` and `models/`. The README records how teammates obtain the same input; later stages will add reproducible processing and model generation.

Check before staging any local data:

```bash
git check-ignore data/abalone.csv
git status --short
```

Store any Kaggle credentials outside the repository. `kaggle.json` and local environment files are ignored as an additional safeguard. A dataset is not needed to run the stage-0 environment checks. The Kaggle download contains 4,177 rows and nine columns. The local CSV SHA-256 is `6fda3c4b0c0cff9bd5f7a46434509179069ed620c8fc7090d872a08d2c1a729a`; compare this hash when reproducing the same data snapshot.

## Code quality and environment checks

```bash
uv pip check
uv run --locked python scripts/check_dependencies.py
uv run --locked python scripts/check_environment.py
uv run --locked ruff check .
uv run --locked ruff check --select I .
uv run --locked ruff format --check .
uv run --locked pre-commit run --all-files --show-diff-on-failure
```

The environment check uses tiny synthetic inputs to verify sklearn transformation/training and FastAPI/Pydantic/HTTPX compatibility. It imports Prefect but does not start a server, run the Abalone assignment, or implement later-stage deliverables.

Pre-commit may fix formatting or strip notebook outputs. Review and stage those changes, then rerun it. Notebook output stripping follows the earlier course activity.

GitHub Actions installs the locked environment on Python 3.11 and runs dependency, environment, and pre-commit checks on pushes and pull requests. A local pass does not claim that GitHub Actions has already run remotely; inspect the PR's checks after pushing.

## Stage 0 checklist

- Direct runtime and development dependencies are declared.
- Course requirements inputs and compiled files are included.
- Python metadata, team authors, and repository URL are filled in.
- Ruff and pre-commit are configured; hooks are installed locally.
- GitHub Actions runs environment and quality checks.
- Local data and generated artifacts are ignored.
- The unused conda file and assignment-only `PR_0.md` are removed before the PR.

Later-stage setup and execution instructions will be added with the corresponding implementations.
