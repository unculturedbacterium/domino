# Publish Domino on GitHub

This directory is the complete repository root. In the published repository,
`README.md`, `pyproject.toml`, `domino/`, `tests/`, `docs/`, `examples/`, and
`.github/` must appear at the top level. The repository name is **Domino**.

## Recommended command-line upload

Create an empty repository named `Domino` under the `unculturedbacterium`
account. Do not initialize it with another README, license, or `.gitignore`.
Run the following commands from this directory:

```bash
python release_check.py
python -m pip install -e ".[test]"
python -m pytest -q
python -m ruff check .

git init
git add .
git commit -m "Publish Domino"
git branch -M main
git remote add origin https://github.com/unculturedbacterium/Domino.git
git push -u origin main
```

Open the GitHub Actions tab after the push and confirm that the Windows,
macOS, and Linux test jobs pass.

## GitHub website upload

1. Create the empty `Domino` repository.
2. Select **Add file**, then **Upload files**.
3. Extract `Domino.zip` locally.
4. Upload everything inside the extracted `Domino` directory while preserving
   the directory structure.
5. Confirm that `.github/workflows/tests.yml` is present.
6. Commit to `main` and check the Actions tab.

GitHub does not unpack a ZIP into a repository through its ordinary file
upload page. Upload the extracted contents, not `Domino.zip` itself.

## Verify repository installation

After the repository is public, test installation in a fresh environment:

```bash
python -m venv domino-test

# Linux or macOS
source domino-test/bin/activate

# Windows PowerShell
domino-test\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install "git+https://github.com/unculturedbacterium/Domino.git"
domino --version
python -c "import domino; print(domino.__version__)"
```

For a persistent paper citation, archive the public repository commit with
Zenodo or another DOI service and add the assigned DOI to `CITATION.cff`.

Private or restricted genotype and phenotype data are intentionally absent.
Do not add study data unless data-owner permission, participant consent, and
the applicable repository policy explicitly allow public redistribution.
