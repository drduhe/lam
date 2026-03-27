# Performance benchmarks

This folder documents how we measure **end-to-end wall time** for the GeoTIFF CLI (`python -m lam` / `lam`). Benchmarks are **environment-sensitive**: GPU model, driver, CUDA, PyTorch build, disk I/O, and whether weights are already cached locally all move the number.

## What we benchmark

- **Scope**: Full pipeline as users run it from a notebook or shell—**model load**, tiled raster read, inference, cross-tile NMS, GeoJSON write.
- **Method**: A small driver invokes the same subprocess you would use in SageMaker (`sys.executable -m lam ...`) and records `time.perf_counter()` around each run. That matches **`%%time`** on a `subprocess.run(..., check=True)` cell and avoids accidentally timing a long-lived Python kernel that already holds a loaded model.
- **Defaults**: Same as the CLI: **1008** px tiles, **128** px overlap, unless you override **`--tile-size`** / **`--overlap`** on the benchmark script.

## Script: `scripts/benchmark_geotiff_cli.py`

From the **repository root** (after `pip install -e ".[geospatial]"` and with weights at **`assets/weights/sam3.pt`** or **`CHECKPOINT_PATH`**):

```bash
# One timed run; writes to a temp GeoJSON and deletes it
python scripts/benchmark_geotiff_cli.py /path/to/large.tif --prompt aircraft

# Machine-readable report (paste into issues or regression notes)
python scripts/benchmark_geotiff_cli.py /path/to/large.tif --prompt aircraft --json

# Warmup runs (not timed) then 3 timed runs — useful to separate “first CUDA compile” from steady state
python scripts/benchmark_geotiff_cli.py /path/to/large.tif --warmup 1 --repeat 3 --json

# Raster size and approximate tile count only (no GPU work)
python scripts/benchmark_geotiff_cli.py /path/to/large.tif --raster-info-only
```

Forward extra flags straight through to `lam` (anything this script does not define):

```bash
python scripts/benchmark_geotiff_cli.py /path/to/large.tif --checkpoint /path/sam3.pt
```

A bare **`--`** still works if you want to separate benchmark args from `lam` args explicitly.

## Reference result (reported)

| Field | Value |
|-------|--------|
| **Host** | SageMaker notebook, **G6** instance, GPU enabled |
| **Input** | `large.tif` — **12 987 × 12 438** px, **EPSG:4326**, **3**×**uint8** bands, **225** tiles at default 1008 / 128 |
| **Command** | `python -m lam --prompt aircraft -o small.geojson …/large.tif` (equivalent to the benchmark script’s subprocess) |
| **Wall time** | **1 min 48 s** (`%%time` on `subprocess.run`, single run) |

Treat this row as a **single datapoint**, not a guarantee. Re-run `scripts/benchmark_geotiff_cli.py` on your hardware and commit or attach the **`--json`** output when comparing releases.

## Tips

- **First run** after boot often includes one-time costs (CUDA kernel compilation, cache cold start). Use **`--warmup 1`** when you care about steady-state throughput.
- **Regression tracking**: Save JSON from **`--json`** next to a git tag or release version; diff `wall_seconds_mean` and the pinned `torch` / `cuda_device` fields.
- **CI**: Full GPU benchmarks usually do not belong in default CI; keep them as **manual** or **scheduled** jobs on a known GPU runner if you automate later.
