from nipype.interfaces.base import (
    BaseInterface, BaseInterfaceInputSpec, TraitedSpec,
    File, traits, isdefined, OutputMultiPath
)
import mne
from mne import find_events
import logging
import os
from megpypes.proc_funcs.artifacts import apply_zapline_denoising
from megpypes.proc_funcs.preprocessing import compute_ica

logger = logging.getLogger(__name__)

class ArtifactRejectionInputSpec(BaseInterfaceInputSpec):
    
    in_file = File(exists=True, mandatory=True, desc="Input MEG file")

    # Enable steps on/off
    enable_zapline = traits.Bool(True, usedefault=True, desc="Flag to enable Zapline Denoising.")
    enable_compute_ica = traits.Bool(True, usedefault=True, desc="Flag to enable ICA decomposition for artifact rejection.")

    # 1. Zapline denoising
    fline = traits.Float(50.0, usedefault=True, desc="Power line frequency (Hz)")
    n_chunks = traits.Int(10, usedefault=True, desc="Number of chunks to split data into")
    spot_sz = traits.Int(7, usedefault=True, desc="Spot size for DSS line removal")
    win_sz = traits.Int(12, usedefault=True, desc="Window size for DSS line removal")
    nfft = traits.Int(2048, usedefault=True, desc="Number of FFT points for DSS")
    n_iter_max = traits.Int(100, usedefault=True, desc="Maximum number of iterations for DSS")
    mag_only = traits.Bool(True, usedefault=True, desc="Process only magnetometer channels")
    detect_line_freq = traits.Bool(True, usedefault=True, desc="Automatically detect line frequency from data (overrides fline if True)")

    # 2. Compute ICA components (if not already computed)
    ica_random_state = traits.Int(mandatory=True, desc="Random seed for ICA reproducibility")
    ica_n_components = traits.Int(20, usedefault=True, desc="Number of ICA components to compute")
    ica_l_freq = traits.Float(1.0, usedefault=True, desc="High-pass frequency for ICA fitting (Hz)")
    ica_h_freq = traits.Float(30.0, usedefault=True, desc="Low-pass frequency for ICA fitting (Hz)")
    ica_method = traits.Enum("fastica","picard","infomax",usedefault=True,desc="ICA algorithm to use")

    # Output
    out_file = traits.Str("artifact_cleaned_raw.fif", usedefault=True, desc="Output filename")
    ica_file = traits.Str("ica-icasolution.fif", usedefault=True, desc="ICA output filename")
    ica_plot_path = traits.Str("ica_comps_source_plot.png", usedefault=True, desc="ICA source plot filename")
    
class ArtifactRejectionOutputSpec(TraitedSpec):
    out_file = traits.File(exists=True, desc="Artifact-cleaned MEG file")
    ica_file = traits.File(exists=True, desc="ICA decomposition used for cleaning")
    ica_plot = traits.File(exists=True, desc="Plot for inspection of ICA components")

class ArtifactRejection(BaseInterface):
    input_spec = ArtifactRejectionInputSpec
    output_spec = ArtifactRejectionOutputSpec

    def _run_interface(self, runtime):
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

        logger.info(f"NODE: Artifact Rejection | In-File: {self.inputs.in_file}")

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
                mag_only=self.inputs.mag_only,
                detect_line_freq=self.inputs.detect_line_freq
            )

        if self.inputs.enable_compute_ica:
            # 2. ICA 
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

            # Plot ICA sources and save png
            fig = ica_comps.plot_components(show=False)
            fig.savefig(self.inputs.ica_plot_path)

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
        outputs["ica_file"] = os.path.abspath(self.inputs.ica_file)
        outputs["ica_plot"] = os.path.abspath(self.inputs.ica_plot_path)
        return outputs
    
