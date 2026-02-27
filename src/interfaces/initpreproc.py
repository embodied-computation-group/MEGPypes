"""
src/interfaces/preproc.py
Creates the NiPype interface for initial raw preprocessing

Heavily inspired by the work in the ephypype package that wraps MNE functionality in NiPype.
(ref: https://github.com/neuropycon/ephypype/blob/master/ephypype/preproc.py)


""" 

from nipype.interfaces.base import (
    BaseInterface, BaseInterfaceInputSpec, TraitedSpec,
    File, traits, isdefined, OutputMultiPath
)
import mne
from mne import find_events
import logging
import os
from src.proc_funcs.preprocessing import compute_ica

logger = logging.getLogger(__name__)

class InitialPreprocInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="Input MEG file")
    # 1. Crop
    stim_channel = traits.Either(
        traits.Str(),
        traits.List(traits.Str()),
        None,
        desc="Stimulus channel (string, list of strings, or None)"
    )
    min_buffer = traits.Float(0.1, usedefault=True, desc="Pre-event crop buffer (s)")
    max_buffer = traits.Float(0.1, usedefault=True, desc="Post-event crop buffer (s)")
    # 2. Filter
    l_freq = traits.Float(1.0, usedefault=True, desc="Low-pass filter cutoff (Hz)")
    h_freq = traits.Float(150.0, usedefault=True, desc="High-pass filter cutoff (Hz)")
    # 3. Gradient compensation
    gradcomp_auto = traits.Bool(True, usedefault=True, desc="Auto gradient compensation")
    gradcomp_order = traits.Int(3, usedefault=True, desc="Manual gradient compensation order")
    # 4. Compute ica
    compute_ica = traits.Bool(True, usedefault=True, desc="Flag to compute ica components file.")
    ## ICA configuration
    ica_random_state = traits.Int(mandatory=True, desc="Random seed for ICA reproducibility")
    ica_n_components = traits.Int(20, usedefault=True, desc="Number of ICA components to compute")
    ica_l_freq = traits.Float(1.0, usedefault=True, desc="High-pass frequency for ICA fitting (Hz)")
    ica_h_freq = traits.Float(30.0, usedefault=True, desc="Low-pass frequency for ICA fitting (Hz)")
    ica_method = traits.Enum("fastica","picard","infomax",usedefault=True,desc="ICA algorithm to use")
    # Output
    out_file = traits.Str("initial_preproc_raw.fif", usedefault=True, desc="Output filename")
    ica_file = traits.Str("initial_preproc_ica.fif", usedefault=True, desc="ICA output filename")

class InitialPreprocOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="Preprocessed MEG file")
    ica_file = File(exists=True, desc="ICA file as output")
    events_file = File(exists=False, desc="Events TSV (optional)")


class InitialPreproc(BaseInterface):
    """Combined cropping + filtering + gradient compensation."""
    input_spec = InitialPreprocInputSpec
    output_spec = InitialPreprocOutputSpec
    
    def _run_interface(self, runtime):
        # Configure logging for this subprocess
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        
        logger.info(f"Initial preproc: {self.inputs.in_file}")
        
        # Load data
        raw = mne.io.read_raw_fif(self.inputs.in_file, preload=True)
        
        # 1. Crop around events
        events = find_events(raw, stim_channel=self.inputs.stim_channel, shortest_event=1)
        tmin = max(0.0, raw.times[events[0][0]] + self.inputs.min_buffer)
        tmax = min(raw.times[-1], raw.times[events[-1][0]] + self.inputs.max_buffer)
        raw = raw.copy().crop(tmin=tmin, tmax=tmax)
        logger.debug(f"Cropped to [{tmin:.2f}, {tmax:.2f}] s")
        
        # 2. Filter
        raw.filter(self.inputs.l_freq, self.inputs.h_freq)
        logger.debug(f"Filtered {self.inputs.l_freq}-{self.inputs.h_freq} Hz")
        
        # 3. Gradient compensation
        if self.inputs.gradcomp_auto:
            comps = raw.info.get("comps", [])
            if comps:
                k = max(range(len(comps)))
                raw.apply_gradient_compensation(k)
                logger.debug(f"Auto gradcomp order: {k}")
            else:
                logger.warning("No gradcomp matrices available")
        else:
            raw.apply_gradient_compensation(self.inputs.gradcomp_order)
            logger.debug(f"Manual gradcomp order: {self.inputs.gradcomp_order}")
        
        # 4. Compute ICA
        if self.inputs.compute_ica:
            logger.info(f"ICA FILE NAME: {self.inputs.ica_file}")
            ica_comps = compute_ica(
                raw=raw,
                random_state=self.inputs.ica_random_state,
                n_components=self.inputs.ica_n_components,
                filt_low=self.inputs.ica_l_freq,
                filt_high=self.inputs.ica_h_freq,
                method=self.inputs.ica_method
            )

            ica_path = os.path.abspath(self.inputs.ica_file)
            ica_comps.save(ica_path, overwrite=True)
            logger.info(f"Saved ICA: {ica_path}")
            
        # Save
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
        return outputs