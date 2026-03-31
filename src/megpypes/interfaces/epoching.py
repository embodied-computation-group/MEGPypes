"""
Epoching Interface
Creates the NiPype interface for epoching MEG data based on events and optional autoreject-based epoch rejection.

Autoreject. https://autoreject.github.io/stable/index.html
"""
import os
import mne
from pathlib import Path
from collections import Counter
from autoreject import AutoReject
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, TraitedSpec, traits
from megpypes.proc_funcs.epoch import handle_find_events
from megpypes.interfaces.utils import abspath_with_time
import logging
logger = logging.getLogger(__name__)


class EpochingInputSpec(BaseInterfaceInputSpec):
    """Input Specification"""
    in_file = traits.File(exists=True, mandatory=True, desc="Input MEG BaseRaw file")
    
    # steps on/off
    epoch = traits.Bool(True, usedefault=True, desc="Flag to enable epoching of data")
    autoreject = traits.Bool(True, desc="Flag to enable autoreject-based epoch rejection")

    # 1. Epoching based on events
    stim_channels = traits.List(traits.Str())
    event_id = traits.Int(mandatory=True)
    event_label = traits.Str(mandatory=True)
    event_tmin = traits.Float(mandatory=True)
    event_tmax = traits.Float(mandatory=True)
    
    # output
    out_file = traits.Str(f"epoched_epo.fif", usedefault=True, desc="Output filename")
    plot_raw_epochs = traits.Str("raw_epochs.png", usedefault=True, desc="Filename for raw epochs plot")
    plot_ar_reject_log = traits.Str("ar_reject_log.png", usedefault=True, desc="Filename for autoreject reject log plot")
    plot_epochs_after_ar = traits.Str("ar_epochs.png", usedefault=True, desc="Filename for AR-cleaned epochs plot")

class EpochingOutputSpec(TraitedSpec):
    """Output Specification"""
    out_file = traits.File(exists=True, desc="Epoched MEG file")
    plot_raw_epochs = traits.File(exists=True, desc="Raw epochs plot")
    plot_ar_reject_log = traits.File(desc="Autoreject reject log plot")
    plot_epochs_after_ar = traits.File(desc="AR-cleaned epochs plot")

class Epoching(BaseInterface):
    """
    NiPype interface for epoching MEG data based on events and optional autoreject-based epoch rejection.

    Steps
    -----
        1. Epoching based on events
        2. (Optional) Autoreject-based epoch rejection

    Returns
    -------
    out_file : File
        Epoched MEG file.
    plot_raw_epochs : File
        Raw-epochs QC plot (before autoreject).
    plot_ar_reject_log : File, optional
        Autoreject reject-log QC plot.
    plot_epochs_after_ar : File, optional
        QC plot of epochs after autoreject.
    """
    input_spec = EpochingInputSpec
    output_spec = EpochingOutputSpec

    def _run_interface(self, runtime):
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

        logger.info(f"NODE: Epoching | In-File: {self.inputs.in_file}")
        logger.info(f"Epoching for event: {self.inputs.event_label}")
        logger.info(f"id: {self.inputs.event_id}, tmin: {self.inputs.event_tmin}, tmax: {self.inputs.event_tmax}")

        # Load data
        raw = mne.io.read_raw_fif(self.inputs.in_file, preload=True)

        # 1. Epoching based on events
        events = handle_find_events(raw, stim_channel=self.inputs.stim_channels)
        logger.debug(f"events: {events}")

        mask = events[:, 2] == self.inputs.event_id
        selected_events = events[mask]

        if len(selected_events) == 0:
            raise RuntimeError(
                f"No events found for trigger {self.inputs.event_code}"
            )
        
        epochs = mne.Epochs(
            raw,
            events=selected_events,
            tmin=self.inputs.event_tmin,
            tmax=self.inputs.event_tmax,
            baseline=None,
            preload=True
        )

        # Save Raw epochs for QC
        fig = epochs.copy().plot_image(picks="mag", combine="mean", show=False)
        self.inputs.plot_raw_epochs = abspath_with_time(self.inputs.plot_raw_epochs)
        plot_path = self._save_plot(fig, self.inputs.plot_raw_epochs)
        logger.debug(f"Saved raw epochs plot: {plot_path}")


        # 2. Autoreject-based epoch rejection
        if self.inputs.autoreject:
            logger.info("Running autoreject on epochs")
            # TODO: picks should be created based on channel types present in the data 
            # AutoReject can only handle on type at a time
            picks = mne.pick_types(
                epochs.info,
                meg="mag",
                eeg=False,
                eog=False,
                stim=False,
                ref_meg=False
            )

            picked_types = [mne.channel_type(epochs.info, i) for i in picks]
            logger.info(f"AutoReject picks: {Counter(picked_types)}")

            epochs_meg = epochs.copy().pick(picks) # Forces the picks abobe
            # TODO: Crosscheck whether these picks are sufficient or should be merged with grads e.g.
            # CTF system might operate differntly than elektromag

            logger.info(f"epochs montage: {epochs.get_montage()}")
            logger.info(len(epochs.info["dig"]) if epochs.info.get("dig") else "No dig points")

            ar = AutoReject(
                n_interpolate=[1, 4, 32], # TODO: What should these values be?
                consensus=[0.5, 1], # TODO: What should these values be?
                thresh_method="random_search",
                random_state=893
            )

            ar.fit(epochs_meg)
            ar_epochs = ar.transform(epochs_meg)

            # TODO: Save the ar object for later QC of rejected epochs and channels
            # we can use the included ar.get_reject_log() function 
            ar_reject_log = ar.get_reject_log(epochs_meg, show=False).plot()
            ar_reject_log_path = self._save_plot(ar_reject_log, self.inputs.plot_ar_reject_log)
            logger.debug(f"Saved AR reject log plot: {ar_reject_log_path}")

            # save AR-cleaned epochs
            fig = ar_epochs.copy().plot_image(picks="mag", combine="mean", show=False)
            ar_epochs_plot_path = self._save_plot(fig, self.inputs.plot_epochs_after_ar)
            logger.debug(f"Saved AR-cleaned epochs plot: {ar_epochs_plot_path}")

            final_epochs = ar_epochs
        else:
            final_epochs = epochs

        # Save epoched data
        new_out_file_str = f"{self.inputs.event_label}-{self.inputs.out_file}"
        # write this to input spec
        self.inputs.out_file = abspath_with_time(new_out_file_str)
        logger.info(f"OUT FILE PATH: {self.inputs.out_file}")
        final_epochs.save(self.inputs.out_file, overwrite=True)
        logger.info(f"Saved: {self.inputs.out_file}")

        runtime.returncode = 0
        return runtime
    
    def _list_outputs(self):
        """NiPype method to list outputs after interface execution"""
        outputs = self._outputs().get()
        outputs["out_file"] = self.inputs.out_file
        outputs["plot_raw_epochs"] = self.inputs.plot_raw_epochs
        outputs["plot_ar_reject_log"] = self.inputs.plot_ar_reject_log
        outputs["plot_epochs_after_ar"] = self.inputs.plot_epochs_after_ar
        return outputs
    
    def _save_plot(self, fig, filename):
        logger.debug(f"_save_plot the fig object: {fig}")
        path = os.path.abspath(filename)
        if fig is not None:
            if isinstance(fig, list):
                if len(fig) == 1:
                    fig = fig[0]  # Unpack single-item list
                elif len(fig) > 1:
                    logger.warning(f"Expected fig to be a single matplotlib figure, but got a list of length {len(fig)}. Attempting to save anyway.")
            else:
                # fig is already a single object, proceed to save
                pass
        else:
            logger.warning(f"Received None for fig, cannot save plot to {path}")
            return None

        fig.savefig(path)
        return path