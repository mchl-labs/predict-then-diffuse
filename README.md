# Predict-Then-Diffuse

This repository contains two notebook-based workflows:

- analytical simulation for the predict-then-diffuse setup as well as length predictor model performance analysis
- empirical profiling for generation cost (FLOPs, GPU time, VRAM)

## Repository Contents

- `ptd_analytical_simulation.ipynb`
	- analytical and simulation-side experiments
	- produces predicted length artifacts used by profiling
- `ptd_empirical_profiling_comparison.ipynb`
	- empirical profiling and comparison runs
	- includes baseline vs fallback/fixed length-profile experiments
- `data/predicted_lengths.csv`
	- baseline predicted lengths
- `data/predicted_lengths_with_fallback.csv`
	- fallback-expanded predicted lengths
- `data/predicted_lengths_fixed.csv`
	- fixed/cleaned fallback lengths
- `pyproject.toml`
	- project dependencies managed with `uv`

## Requirements

- Python 3.13+
- `uv` installed
- NVIDIA GPU (recommended for profiling cells)

Some profiling cells use optional packages not pinned in `pyproject.toml` (for example `deepspeed`).

## Setup

From the repository root:

```bash
uv sync
```

If you need optional profiling dependencies used in specific notebook cells:

```bash
uv add deepspeed
```

## How To Run

## 1) Analytical Simulation

Open `ptd_analytical_simulation.ipynb` and run cells top-to-bottom.

This notebook is the best place to:

- build/inspect the analytical setup
- generate or validate length prediction artifacts

## 2) Empirical Profiling

Open `ptd_empirical_profiling_comparison.ipynb` and run cells top-to-bottom.

The final profiling sections compare multiple input policies:

- Experiment A: `predicted_lengths.csv` (baseline)
- Experiment B: `predicted_lengths_fallback.csv` (legacy fallback variant, if present)
- Experiment C: `predicted_lengths_fixed.csv` (current fixed length variant)


## Notes

- Notebook output can depend on GPU memory and driver/toolkit versions.
- If a CSV file is reported missing, run the analytical notebook first or place the required file under `data/`.
- If the model download is slow/fails, rerun after network/auth checks for your Hugging Face environment.