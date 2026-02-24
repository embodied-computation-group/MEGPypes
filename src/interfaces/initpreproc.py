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

logger = logging.getLogger(__name__)

class InitialPreprocInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="Input MEG file")
    stim_channel = traits.Either(
        traits.Str(default="STI 014", desc="Stimulus channel as a string"),
        traits.List(traits.Str(), desc="List of stimulus channels"),
        None,
        desc="Stimulus channel (string, list of strings, or None)"
    )
    min_buffer = traits.Float(default=0.1, desc="Pre-event crop buffer (s)")
    max_buffer = traits.Float(default=0.1, desc="Post-event crop buffer (s)")
    l_freq = traits.Float(default=1.0, desc="Low-pass filter cutoff (Hz)")
    h_freq = traits.Float(default=150.0, desc="High-pass filter cutoff (Hz)")
    gradcomp_auto = traits.Bool(default=True, desc="Auto gradient compensation")
    gradcomp_order = traits.Int(default=3, desc="Manual gradient compensation order")
    out_file = traits.Str(default="preproc_raw.fif", desc="Output filename")


class InitialPreprocOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="Preprocessed MEG file")
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
        
        # Save
        out_path = os.path.abspath(self.inputs.out_file)
        raw.save(out_path, overwrite=True)
        logger.info(f"Saved: {out_path}")
        
        runtime.returncode = 0
        return runtime
    
    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["out_file"] = os.path.abspath(self.inputs.out_file)
        return outputs