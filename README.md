
# MEGPypes

User-friendly, automated electrophysiology preprocessing pipelines in Python.

MEGPypes is a configuration-driven preprocessing package for MEG/EEG-style workflows, built largely on the NiPype neuroimaging workflow framework and centered on MNE for electrophysiological signal processing.

The project is authored by Peli under the Embodied Computation Group, Aarhus University.

## Why MEGPypes

MEGPypes is designed to make preprocessing reproducible, transparent, and scalable:

- NiPype-native workflow orchestration for modular, cacheable pipelines.
- MNE-based signal processing primitives for robust electrophysiology handling.
- Single-YAML experiment setup for both processing parameters and step activation.
- Parallel execution support through NiPype plugins (`Linear`, `MultiProc`).
- BIDS-style output structuring utilities and QC asset export.
- Built-in Streamlit report app for visual quality control review.

## Pipeline Overview

The current preprocessing flow is assembled in a NiPype workflow and includes:

1. Initial preprocessing
- Event-based crop windowing
- Band-pass filtering
- Gradient compensation (e.g., CTF use cases)

2. Artifact rejection
- Optional Zapline denoising
- Optional ICA decomposition and diagnostic plots

3. Automatic ICA cleaning
- Optional ICLabel-based component exclusion
- Artifact-aware ICA application to raw data

4. Epoching
- Event-mapped epoch extraction
- Optional autoreject-based epoch cleaning

In practice, these stages are represented by configurable interfaces (`initial_preproc`, `artifact_rejection`, `auto_ica`, `epoching`) connected in the main workflow builder.

## Configuration-First Design

MEGPypes is driven by a single YAML configuration file (see `demos/config_effort.yaml`) with three top-level sections:

- `paths`: data locations, file templates, iterable dimensions (e.g., subject/session).
- `workflow`: execution backend and worker strategy.
- `pipeline_config`: per-stage `steps` and `args`.

The key idea is that optional parts of the pipeline can be toggled on/off directly in YAML, while keeping all parameters in one reproducible spec.

Example pattern:

```yaml
pipeline_config:
	artifact_rejection:
		steps:
			zapline: true
			compute_ica: true
		args:
			zapline:
				fline: 60.0
				n_chunks: 10
			compute_ica:
				ica_random_state: 821
```

This enables one config file to define both processing behavior and execution logic for full cohorts.

## Quickstart

### Install

From the project root:

```bash
pip install -e .
```

### Run from Python

```python
from pathlib import Path
from megpypes.runner import MegPypesRunner

config_path = Path("demos/config_effort.yaml")
runner = MegPypesRunner.from_yaml(config_path)

# Build workflow
wf = runner.create_workflow()

# Execute using config defaults
run_info = runner.run(write_graph=True)
print(run_info.plugin, run_info.n_workers)
```

### Run with overrides

```python
custom_wf = runner.create_workflow(
		paths_override={"iterable_values": {"subject": ["0001"], "session": ["01"]}}
)

runner.run(
		workflow=custom_wf,
		plugin="MultiProc",
		n_workers=4,
)
```

### Launch QC Report App

```python
proc = runner.launch_report_app(report_root="workdir/megpreproc/output/", port=8501)
print(runner.report_url(port=8501))

# stop when done
runner.stop_report_app()
```

## Citations

If you use MEGPypes, please cite the core workflow and signal-processing frameworks:

1. Gorgolewski K, Burns CD, Madison C, Clark D, Halchenko YO, Waskom ML, Ghosh SS. (2011). Nipype: a flexible, lightweight and extensible neuroimaging data processing framework in Python. Front. Neuroinform. 5:13.
2. Alexandre Gramfort, Martin Luessi, Eric Larson, Denis A. Engemann, Daniel Strohmeier, Christian Brodbeck, Roman Goj, Mainak Jas, Teon Brooks, Lauri Parkkonen, and Matti S. Hamalainen. MEG and EEG data analysis with MNE-Python. Frontiers in Neuroscience, 7(267):1-13, 2013. doi:10.3389/fnins.2013.00267.

## Project Inspiration

MEGPypes is inspired by ephypype:

- https://github.com/neuropycon/ephypype

## Repository Layout

- `src/megpypes/runner.py`: high-level user API (`MegPypesRunner`).
- `src/megpypes/pipelines/meg_preprocessing.py`: workflow construction and node wiring.
- `src/megpypes/interfaces/`: NiPype interfaces for each preprocessing stage.
- `src/megpypes/proc_funcs/`: processing helper functions.
- `src/megpypes/report/`: Streamlit-based QC viewer.
- `demos/config_effort.yaml`: example full workflow configuration.
- `demos/run_workflow_example.ipynb`: end-to-end usage tutorial.

 