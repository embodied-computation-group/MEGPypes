import mne
import logging
logger = logging.getLogger(__name__)

def handle_find_events(raw):
    """
    Finds events, but automatically handles the shortest_event duration that might be conflicted in mne.
    """
        
    try:
        logger.info("Finding events with default parameters")
        events = mne.find_events(raw)

    except ValueError as e:
        logger.warning("Default mne.find_events() failed. Attempting fallback.")

        # get stim channel
        stim_picks = mne.pick_types(raw.info, stim=True)

        if len(stim_picks) == 0:
            raise RuntimeError("No stim channel found for event detection")

        stim_data = raw.get_data(picks=stim_picks)[0]

        # find where trigger values change
        changes = (stim_data[1:] != stim_data[:-1]).nonzero()[0] + 1

        if len(changes) < 2:
            raise RuntimeError("Could not estimate event durations from stim channel")

        # estimate durations between changes
        durations = changes[1:] - changes[:-1]

        shortest = durations.min()

        logger.warning(
            f"Detected unusually short events. "
            f"Estimated shortest duration: {shortest} samples."
        )

        # fallback: allow events slightly shorter than shortest
        min_duration = (shortest - 1) / raw.info["sfreq"]

        logger.warning(
            f"Retrying find_events with min_duration={min_duration:.6f}"
        )

        events = mne.find_events(raw, min_duration=min_duration)

    return events