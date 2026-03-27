# Changelog

<!-- markdownlint-disable-file MD024 -->

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.1] - 2026-03-27

### Added

- GeoTIFF CLI wall-clock benchmarking: `scripts/benchmark_geotiff_cli.py` runs `python -m lam` in a subprocess (aligned with typical SageMaker `%%time` usage), with `--json`, `--warmup`, `--repeat`, `--raster-info-only`, and passthrough of extra `lam` flags.
- `benchmarks/README.md` documenting methodology, script usage, and regression-tracking tips; `benchmarks/g6.4xlarge.json` as a sample machine-readable result.
- Root `README.md` performance-benchmarking subsection and repository-layout entries for `benchmarks/` and the benchmark script.

### Changed

- CLI loads weights from a **local file only** (`$CHECKPOINT_PATH` or `<repo>/assets/weights/sam3.pt`); it no longer downloads from Hugging Face. Removed `--no-hf`. Missing weights exit with a clear message before model load.
- `lam/__init__.py` version set to **0.2.1**.

## [0.2.0] - 2026-03-27

### Added

- AWS CDK app under `cdk/` to provision a VPC, SageMaker execution role, and a real-time SageMaker endpoint for the LAM container built from `docker/Dockerfile.lam-sagemaker`, with deployment settings loaded from `bin/deployment/deployment.json` (see `deployment.json.example`).
- CDK unit tests and ESLint; CI job to install CDK dependencies, lint, and run Jest.
- Pre-commit hooks to type-check and lint the CDK TypeScript project when files under `cdk/` change.

### Changed

- Expanded root `.dockerignore` and `.gitignore` so Docker build contexts stay small and safe (for example exclude CDK and test trees from the image context; ignore weight checkpoints via `**/assets/weights/*.pt`).
- Documented SageMaker/AWS deployment via `cdk/README.md` and short pointers in the root `README.md`.

## [0.1.0] - 2026-03-26

### Added

- Initial release.
