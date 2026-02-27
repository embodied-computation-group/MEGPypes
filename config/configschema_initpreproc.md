# Configuration Schema

## Paths
- `basedir`: Base directory for raw data (string)
- `workdir`: Directory for intermediate files (string)
- `outputdir`: Directory for final outputs (string)
- `subjects`: List of subjects to process (list of strings)

## Workflow
- `plugin`: Plugin for parallel processing (string: "Linear" or "MultiProc")
- `n_workers`: Number of workers for parallel processing (int)
- `auto_workers`: Automatically set workers to max CPU cores - 2 (bool)

## Initial Preprocessing
**1. Crop**
- `crop.stim_channel`: Channel to use for stimulus-based cropping (string).
  - If Empty: Falls back to mne's auto definition of stimuli channels.
- `crop.min_buffer`: Minimum time buffer (seconds) before stimulus onset (float)
- `crop.max_buffer`: Maximum time buffer (seconds) after stimulus offset (float)
**2. Filter (Rough or Biased)**
- `filter.l_freq`: Low-frequency cutoff (Hz) (float)
- `filter.h_freq`: High-frequency cutoff (Hz) (float)
**3. Gradient Compensation** 
- `gradcomp.auto`: Automatically apply gradient compensation (bool)
- `gradcomp.order`: Order of gradient compensation (int)
**4. Compute ICA**
- `ica.compute`: Whether to compute ICA (bool)
- `ica.random_state`: Random seed for reproducibility (int)
  - Is mandatory for reproducability
- `ica.n_components`: Number of ICA components (int)
- `ica.l_freq`: Low-frequency cutoff for ICA (Hz) (float)
- `ica.h_freq`: High-frequency cutoff for ICA (Hz) (float)
- `ica.method`: ICA method (string: "fastica", "infomax", etc.)
