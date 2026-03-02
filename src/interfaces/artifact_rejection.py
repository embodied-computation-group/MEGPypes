from nipype.interfaces.base import (
    BaseInterface, BaseInterfaceInputSpec, TraitedSpec,
    File, traits, isdefined, OutputMultiPath
)
import mne
from mne import find_events
import logging
import os
from src.proc_funcs.artifacts import apply_zapline_denoising

logger = logging.getLogger(__name__)

class ArtifactRejectionInputSpec(BaseInterfaceInputSpec):
    
    in_file = File(exists=True, mandatory=True, desc="Input MEG file")

    # Enable steps on/off
    enable_zapline = traits.Bool(True, usedefault=True, desc="Flag to enable Zapline Denoising.")
    enable_ica = traits.Bool(True, usedefault=True, desc="Flag to enable compute ica components file.")

    # 1. Zapline denoising
    fline = traits.Float(50.0, usedefault=True, desc="Power line frequency (Hz)")
    n_chunks = traits.Int(10, usedefault=True, desc="Number of chunks to split data into")
    spot_sz = traits.Int(7, usedefault=True, desc="Spot size for DSS line removal")
    win_sz = traits.Int(12, usedefault=True, desc="Window size for DSS line removal")
    nfft = traits.Int(2048, usedefault=True, desc="Number of FFT points for DSS")
    n_iter_max = traits.Int(30, usedefault=True, desc="Maximum number of iterations for DSS")
    mag_only = traits.Bool(True, usedefault=True, desc="Process only magnetometer channels")
    # 2. 

    # 4. Compute ica components
    ica_random_state = traits.Int(mandatory=True, desc="Random seed for ICA reproducibility")
    ica_n_components = traits.Int(20, usedefault=True, desc="Number of ICA components to compute")
    ica_l_freq = traits.Float(1.0, usedefault=True, desc="High-pass frequency for ICA fitting (Hz)")
    ica_h_freq = traits.Float(30.0, usedefault=True, desc="Low-pass frequency for ICA fitting (Hz)")
    ica_method = traits.Enum("fastica","picard","infomax",usedefault=True,desc="ICA algorithm to use")

    # Output
    out_file = traits.Str("artifact_cleaned_raw.fif", usedefault=True, desc="Output filename")
    
    # TODO: ... Write all input traits here

class ArtifactRejectionOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="Artifact-cleaned MEG file")
    ica_file = File(exists=True, desc="ICA file as output")

class ArtifactRejection(BaseInterface):
    input_spec = ArtifactRejectionInputSpec
    output_spec = ArtifactRejectionOutputSpec

    def _run_interface(self, runtime):
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

        logger.info(f"WF: Artifact Rejection | In-File: {self.inputs.in_file}")

        # Load data
        raw = mne.io.read_raw_fif(self.inputs.in_file, preload=True)

        # 1. Zapline denoising (removes line noise)
        if self.inputs.enable_zapline:
            raw = apply_zapline_denoising(
                raw=raw,
                fline=self.inputs.fline,
                n_chunks=self.inputs.n_chunks,
                spot_sz=self.inputs.spot_sz,
                win_sz=self.inputs.win_sz,
                nfft=self.inputs.nfft,
                n_iter_max=self.inputs.n_iter_max,
                mag_only=self.inputs.mag_only
            )

        
        # Compute ICA components if enabled (ICA fitting is separate from artifact removal - happens in a later step)
        if self.inputs.enable_ica:
            logger.info(f"ICA FILE NAME: {self.inputs.ica_file}")
            # compute ica components
            ica_comps = compute_ica(
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

        # Save output file
        logger.info(f"OUT FILE PATH: {self.inputs.out_file}")
        out_path = os.path.abspath(self.inputs.out_file)
        raw.save(out_path, overwrite=True)
        logger.info(f"Saved: {out_path}")

        runtime.returncode = 0
        return runtime
    
    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["out_file"] = os.path.abspath(self.inputs.out_file)
        return outputs
    
