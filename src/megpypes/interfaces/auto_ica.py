"""
Automatic ICA-based artifact removal interface for MEG data using MNE-Python and ICLabel classification.

The interface is mainly designed to remove biological artifacts such as eye blinks, muscle activity, and heartbeats from MEG recordings. 
It is designed to be used after the initial preprocessing interface and power line artifact rejection steps like Zapline denoising.

This interface:
- computes ICA components from the raw filtered MEG
- applies ICLabel classification to identify artifact components based on user-defined class labels and probability thresholds
- applies ICA cleaning to remove identified artifact components from the raw data

The user can also provide a pre-computed ICA decomposition and/or manually specify which ICA components indices to exclude instead of relying on automatic ICLabel classification.

"""
import os
import mne
from mne_icalabel import label_components
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec, traits
import logging

from megpypes.proc_funcs.preprocessing import compute_ica


logger = logging.getLogger(__name__)

class AutoICAInputSpec(BaseInterfaceInputSpec):
    in_file = traits.File(exists=True, mandatory=True, desc="Path to the raw FIF file to process")

    # 1. Compute ICA components (if not already computed)
    ica_random_state = traits.Int(mandatory=True, desc="Random seed for ICA reproducibility")
    ica_n_components = traits.Int(20, usedefault=True, desc="Number of ICA components to compute")
    ica_l_freq = traits.Float(1.0, usedefault=True, desc="High-pass frequency for ICA fitting (Hz)")
    ica_h_freq = traits.Float(30.0, usedefault=True, desc="Low-pass frequency for ICA fitting (Hz)")
    ica_method = traits.Enum("fastica","picard","infomax",usedefault=True,desc="ICA algorithm to use")

    # 2. ICLabel classification and exclusion
    ica_exclude = traits.List(traits.Int(), usedefault=True, mandatory=False, desc="List of ICA component indices to exclude (e.g., [0, 1, 2])")
    ic_labels_exclude = traits.List(traits.Str(), desc="List of ICLabel class labels corresponding to the ICA components (e.g., ['muscle artifact', 'eye blink', 'heart beat'])")
    ic_label_threshold = traits.Float(0.7, usedefault=True, desc="Probability threshold for ICLabel-based exclusion (e.g., 0.7 means exclude components with >70% probability of being an artifact)")

    # Output
    out_file = traits.File(desc="Path to save the ICA-applied raw FIF file")
    ica_file = traits.Str("auto_ica-icasolution.fif", usedefault=True, desc="ICA output filename")

class AutoICAOutputSpec(TraitedSpec):
    out_file = traits.File(exists=True, desc="ICA-applied raw FIF file")
    ica_file = traits.File(exists=True, desc="ICA decomposition used for cleaning")

class AutoICA(BaseInterface):
    input_spec = AutoICAInputSpec
    output_spec = AutoICAOutputSpec

    def _run_interface(self, runtime):
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

        logger.info(f"NODE: AutoICA | Raw File: {self.inputs.in_file} | ICA File: {self.inputs.ica_file}")

        # assert if ic_labels are valid
        if self.inputs.ic_labels_exclude:
            valid_labels = ['muscle artifact', 'eye blink', 'heart beat', 'line noise', 'channel noise']
            for label in self.inputs.ic_labels_exclude:
                if label not in valid_labels:
                    raise ValueError(f"Invalid ICLabel class label: '{label}'. Valid options are: {valid_labels}")

        # load Raw
        raw = mne.io.read_raw_fif(self.inputs.in_file, preload=True)

        # apply a common average refercing to comply with ICLabel
        raw.set_eeg_reference('average')
        
        # 1. compute ICA
        if self.inputs.ica_file:
            logger.info(f"Loading existing ICA decomposition from: {self.inputs.ica_file}")
            ica = mne.preprocessing.read_ica(self.inputs.ica_file)
        else:
            logger.info("No ICA file provided. Computing ICA decomposition from raw data.")
            ica = compute_ica(
                raw=raw,
                random_state=self.inputs.ica_random_state,
                n_components=self.inputs.ica_n_components,
                filt_low=self.inputs.ica_l_freq,
                filt_high=self.inputs.ica_h_freq,
                method=self.inputs.ica_method
            )
            # save ica
            ica_path = os.path.abspath(self.inputs.ica_file)
            ica_comps.save(ica_path, overwrite=True)
            logger.info(f"Saved ICA: {ica_path}")

        # 2.Pick ICA exclusion indices Label ICA components ICA cleaning
        if self.inputs.ica_exclude:
            ica_exclude_idx = self.inputs.ica_exclude
        else:
            # ICLabel Automatic ICA Eexclusion
            logger.info("No hardcoded ICA components to exclude. Automatic ICLabel classification of exclusion.")
            ic_labels = label_components(raw, ica, method="iclabel") # extracts automatic ICA component labels with probabilities
            labels = ic_labels["labels"]
            probs = ic_labels["probs"]
            # Define a threshold for exclusion
            threshold = 0.7
    
            # Get the indices of ICA coomponents to exclude based on user defined class labels and probability threshold
            ica_exclude = [
                (idx, label) for idx, label in enumerate(zip(labels, probs))
                if label[0] in self.inputs.ic_labels_exclude and label[1] > self.inputs.ic_label_threshold
            ]
            ica_exclude_labels = [label for idx, label in ica_exclude]
            ica_exclude_idx = [idx for idx, label in ica_exclude]
            logger.info(f"Excluding ICA components with labels: {ica_exclude_labels}")
            logger.info(f"and with indices: {ica_exclude_idx}")
            # TODO: Log the rejected indices of component labelos that were below threshold for later QC
        
        # 3. Apply ICA cleaning
        raw_clean = ica.apply(raw, exclude=ica_exclude_idx)

        # Save ICA cleaned raw data
        out_path = os.path.abspath(self.inputs.out_file)
        raw_clean.save(out_path, overwrite=True)
        logger.info(f"Saved ICA-applied raw file: {out_path}")
        runtime.returncode = 0
        return runtime
    
    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["out_file"] = os.path.abspath(self.inputs.out_file)
        return outputs
    
