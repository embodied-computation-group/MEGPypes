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
from megpypes.interfaces.utils import abspath_with_time

logger = logging.getLogger(__name__)

class ArtifactRejectionInputSpec(BaseInterfaceInputSpec):
    
    in_file = File(exists=True, mandatory=True, desc="Input MEG file")

    enable_zapline = traits.Bool(True, usedefault=True)
    enable_compute_ica = traits.Bool(True, usedefault=True)

    fline = traits.Float(50.0, usedefault=True)
    n_chunks = traits.Int(10, usedefault=True)
    spot_sz = traits.Int(7, usedefault=True)
    win_sz = traits.Int(12, usedefault=True)
    nfft = traits.Int(2048, usedefault=True)
    n_iter_max = traits.Int(100, usedefault=True)
    mag_only = traits.Bool(True, usedefault=True)
    detect_line_freq = traits.Bool(True, usedefault=True)

    ica_random_state = traits.Int(mandatory=True)
    ica_n_components = traits.Int(20, usedefault=True)
    ica_l_freq = traits.Float(1.0, usedefault=True)
    ica_h_freq = traits.Float(30.0, usedefault=True)
    ica_method = traits.Enum("fastica","picard","infomax",usedefault=True)

    out_file = traits.Str("artifact-cleaned_raw.fif", usedefault=True)
    ica_file = traits.Str("ica_icasolution.fif", usedefault=True)

    psd_before = traits.Str("psd_before_zapline.png", usedefault=True)
    psd_after = traits.Str("psd_after_zapline.png", usedefault=True)
    ica_plot_path = traits.Str("ica-components.png", usedefault=True)


class ArtifactRejectionOutputSpec(TraitedSpec):
    out_file = File(exists=True)
    ica_file = File(exists=True)
    psd_before = File(exists=True)
    psd_after = File()
    ica_plot = File(exists=True)


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

        raw = mne.io.read_raw_fif(self.inputs.in_file, preload=True)

        fig = raw.copy().compute_psd().plot(show=False)
        self.inputs.psd_before = abspath_with_time(self.inputs.psd_before)
        psd_before_path = self._save_plot(fig, self.inputs.psd_before)

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

            fig = raw.copy().compute_psd().plot(show=False)
            self.inputs.psd_after = abspath_with_time(self.inputs.psd_after)
            psd_after_path = self._save_plot(fig, self.inputs.psd_after)

        if self.inputs.enable_compute_ica:
            ica_comps = compute_ica(
                raw=raw,
                random_state=self.inputs.ica_random_state,
                n_components=self.inputs.ica_n_components,
                filt_low=self.inputs.ica_l_freq,
                filt_high=self.inputs.ica_h_freq,
                method=self.inputs.ica_method
            )

            self.inputs.ica_file = abspath_with_time(self.inputs.ica_file)
            ica_comps.save(self.inputs.ica_file, overwrite=True)

            fig = ica_comps.plot_components(show=False)
            self.inputs.ica_plot_path = abspath_with_time(self.inputs.ica_plot_path)
            ica_plot_path = self._save_plot(fig, self.inputs.ica_plot_path)

        self.inputs.out_file = abspath_with_time(self.inputs.out_file)
        raw.save(self.inputs.out_file, overwrite=True)

        runtime.returncode = 0
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()

        outputs["out_file"] = self.inputs.out_file
        outputs["ica_file"] = self.inputs.ica_file
        outputs["psd_before"] = self.inputs.psd_before
        outputs["psd_after"] = self.inputs.psd_after
        outputs["ica_plot"] = self.inputs.ica_plot_path

        return outputs
    
    def _save_plot(self, fig, filename):
        path = filename
        fig.savefig(path)
        return path