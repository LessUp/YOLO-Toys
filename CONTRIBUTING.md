# Contributing

Thanks for your interest in contributing!

## Quick Start
1. Fork and clone the repo.
2. Create a new branch: `git checkout -b feat/your-change`.
3. Create a virtualenv and install deps:
   - `python3 -m venv .venv`
   - `.venv/bin/pip install -r requirements.txt`
   - `.venv/bin/pip install -r requirements-dev.txt`
   - `pre-commit install`
4. Run checks locally:
   - `pre-commit run --all-files`
   - `pytest -q`
5. Commit, push, and open a Pull Request.

## Guidelines
- Keep PRs small and focused. Include description, screenshots if UI changes.
- Include tests when possible. Avoid large binary files in git.
- Follow the style configs in `pyproject.toml` and pre-commit hooks.
- Prefer Conventional Commits style for commit messages (e.g. `feat: ...`, `fix: ...`).

## Issues
- Use GitHub Issues for bugs/features. Provide steps to reproduce and logs.
