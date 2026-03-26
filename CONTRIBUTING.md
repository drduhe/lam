# LAM Contributing Guidelines

Thank you for your interest in contributing to LAM. This document describes our **trunk-based** workflow (PRs merge to `main`) and what we expect in pull requests. Contributions are **fork-and-PR** only. Please read this before opening a PR.

## Table of Contents

- [Trunk-Based Development](#trunk-based-development)
- [Linting and tests](#linting-and-tests)
- [Code Style](#code-style)
- [Commit Messages](#commit-messages)
- [Issue Tracking](#issue-tracking)

## Trunk-Based Development

![Trunk-Based Development](https://github.com/aws-solutions-library-samples/osml-imagery-toolkit/assets/4109909/0b3c03ae-4518-471e-9331-da850f0d2e22)

We use **trunk-based development**: short-lived branches merge into `main` (the trunk). Please use **`feature/*`** branches for new features or substantial enhancements.

### How to contribute (fork and pull request)

#### Step 1: Fork the repository

On GitHub, use **Fork** to create a copy under your account.

#### Step 2: Clone your fork

```bash
git clone <url-of-your-fork>
cd lam
```

Optional: add the canonical repo as `upstream` so you can update `main` locally:

```bash
git remote add upstream <url-of-canonical-repository>
```

#### Step 3: Branch from up-to-date `main`

Sync your local `main` with the upstream project (GitHub **Sync fork**, or `git fetch upstream && git checkout main && git merge upstream/main`), then:

```bash
git checkout -b feature/<short-description>
```

To start from a **release tag** instead of `main` (requires a remote that has tags, usually `upstream`):

```bash
git fetch upstream --tags
git checkout -b my-branch tags/<tag-name>
```

#### Step 4: Develop, commit, push

```bash
git add -p   # or git add <paths>
git commit -m "type: short description"
git push -u origin feature/<short-description>
```

Pushing runs **GitHub Actions** (lint + tests) on the PR.

#### Step 5: Open a pull request

Open a PR with **base** = this repository’s `main` and **compare** = your branch. Open it early if you want feedback (**draft** / **WIP** is fine). Describe the change and link related issues.

#### Step 6: Review and keep the branch current

Address feedback with additional commits or by amending as maintainers prefer. Before merge, rebase on the latest `main` when asked:

```bash
git fetch upstream
git rebase upstream/main
git push --force-with-lease
```

If you did not add `upstream`, use `git fetch origin` and `git rebase origin/main` after merging the latest `main` into your fork’s default branch.

A maintainer merges after approval.

## Linting and tests

CI runs **Ruff**, **Black**, and **isort** on `lam/`, `app.py`, and `scripts/`, then **pytest** (see [`.github/workflows/ci.yml`](.github/workflows/ci.yml)).

Locally, install dev dependencies and enable hooks:

```bash
pip install -e ".[dev]"
pre-commit install
```

Hooks (see [`.pre-commit-config.yaml`](.pre-commit-config.yaml)) include trailing whitespace, YAML/TOML/JSON checks, **isort**, **Black**, and **Ruff** (with `--fix` where applicable).

Run hooks on everything once:

```bash
pre-commit run --all-files
```

Match CI lint commands:

```bash
ruff check lam app.py scripts
black --check lam app.py scripts
isort --check-only lam app.py scripts
```

Run tests:

```bash
pytest
```

## Code Style

- **Formatting**: Black, **120** character lines (`[tool.black]` in [`pyproject.toml`](pyproject.toml)).
- **Imports**: isort with **black** profile and `lam` as first-party.
- **Lint**: Ruff rules in `pyproject.toml` (subset of pycodestyle/pyflakes; some rules ignored for legacy/upstream-aligned code).
- **Docstrings**: **Google** style where modules are documented (summary line, then Args / Returns / Raises as appropriate).

Follow patterns in existing `lam/` code for public APIs, typing, and error handling.

## Commit Messages

We follow **[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/#summary)** so history stays readable and tooling-friendly.

```text
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

Examples: `feat: add CLI flag for tile overlap`, `fix(geospatial): handle missing CRS`, `docs: link pre-commit in README`.

## Issue Tracking

Use **GitHub Issues** for bugs and tracked feature work (search open and closed issues first; use the issue templates when they fit). Use **GitHub Discussions** for questions and open-ended topics when that feature is enabled on this repository. For maintenance expectations and how to get involved as a co-maintainer, see **[Community and support](README.md#community-and-support)** in the README.

New bug reports should include a clear description, expected vs actual behavior, and reproduction steps (commands, sample inputs, environment).

Thank you for helping improve LAM.
