"""Check that the Domino tree is ready to publish as a source repository."""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parent
VERSION = "1.0.0"
REQUIRED = {
    ".github/workflows/tests.yml",
    ".gitignore",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "GITHUB_UPLOAD.md",
    "LICENSE",
    "README.md",
    "environment.yml",
    "pyproject.toml",
    "domino/__init__.py",
    "domino/cli.py",
    "examples/quickstart.py",
    "tests/test_core.py",
    "tests/test_optimization.py",
}
FORBIDDEN_PARTS = {
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "build",
    "dist",
    "example_output",
}
FORBIDDEN_SUFFIXES = {
    ".bed",
    ".bim",
    ".fam",
    ".npy",
    ".npz",
    ".parquet",
    ".pyc",
}
MAX_FILE_BYTES = 50 * 1024 * 1024


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def repository_files() -> list[Path]:
    return [path for path in ROOT.rglob("*") if path.is_file() and ".git" not in path.parts]


def check_required() -> None:
    missing = sorted(path for path in REQUIRED if not (ROOT / path).is_file())
    if missing:
        fail(f"required files are missing: {missing}")


def check_metadata() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    package_init = (ROOT / "domino/__init__.py").read_text(encoding="utf-8")
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    expected = [
        'name = "Domino"',
        f'version = "{VERSION}"',
        'domino = "domino.cli:main"',
        "github.com/unculturedbacterium/Domino",
    ]
    for value in expected:
        if value not in pyproject:
            fail(f"pyproject.toml is missing {value!r}")
    if f'__version__ = "{VERSION}"' not in package_init:
        fail("Python package version does not match the release")
    if f"version: {VERSION}" not in citation:
        fail("CITATION.cff version does not match the release")


def check_artifacts(files: list[Path]) -> None:
    for path in files:
        relative = path.relative_to(ROOT)
        if any(part in FORBIDDEN_PARTS or part.endswith(".egg-info") for part in relative.parts):
            fail(f"generated directory is present: {relative}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            fail(f"private or generated data file is present: {relative}")
        if path.stat().st_size > MAX_FILE_BYTES:
            fail(f"file exceeds 50 MiB: {relative}")


def check_markdown_links(files: list[Path]) -> None:
    pattern = re.compile(r"(?<!!)\[[^]]*\]\(([^)]+)\)")
    for path in files:
        if path.suffix.lower() != ".md":
            continue
        text = path.read_text(encoding="utf-8")
        for target in pattern.findall(text):
            target = target.strip().split(maxsplit=1)[0].strip("<>")
            if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            local = unquote(target.split("#", 1)[0])
            if local and not (path.parent / local).exists():
                fail(f"broken Markdown link in {path.relative_to(ROOT)}: {target}")


def check_release_identity(files: list[Path]) -> None:
    retired_name = "dom" + "gwas"
    retired_version = "0." + "3.0"
    stale_patterns = (
        retired_name,
        "domino-" + "gwas",
        retired_version,
        "CHANGE" + "LOG.md",
    )
    for path in files:
        if path.suffix.lower() not in {".md", ".py", ".toml", ".yml", ".yaml", ".cff"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in stale_patterns:
            if pattern.lower() in text.lower():
                fail(f"stale release identity {pattern!r} in {path.relative_to(ROOT)}")


def main() -> None:
    check_required()
    files = repository_files()
    check_metadata()
    check_artifacts(files)
    check_markdown_links(files)
    check_release_identity(files)
    print(f"PASS: Domino; package metadata {VERSION}; {len(files)} repository files")


if __name__ == "__main__":
    main()
