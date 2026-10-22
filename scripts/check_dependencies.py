"""Check that course requirements inputs agree with the uv project metadata."""

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_requirements(filename: str) -> set[str]:
    """Read direct requirements while excluding comments and included files."""
    return {
        line.strip()
        for line in (ROOT / filename).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith(("#", "-r ", "-c "))
    }


def main() -> None:
    """Reject dependency drift between project metadata and course input files."""
    with (ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)
    groups = [
        ("requirements.in", set(project["project"]["dependencies"])),
        ("requirements-dev.in", set(project["dependency-groups"]["dev"])),
    ]
    for filename, expected in groups:
        actual = read_requirements(filename)
        if actual != expected:
            raise ValueError(f"{filename}: missing={expected - actual}, extra={actual - expected}")
    constraints = read_requirements("constraints-course.txt")
    if constraints != set(project["tool"]["uv"]["constraint-dependencies"]):
        raise ValueError("Course constraints differ between constraints-course.txt and pyproject.toml")
    print("Dependency inputs and course constraints match pyproject.toml.")


if __name__ == "__main__":
    main()
