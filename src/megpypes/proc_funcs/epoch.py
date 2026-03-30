import mne
import logging
logger = logging.getLogger(__name__)

def _normalize_channel_list(stim_channel):
    """Normalize user input into a list of channel names."""
    if stim_channel is None:
        return []
    if isinstance(stim_channel, str):
        return [stim_channel]
    return [ch for ch in stim_channel if ch]


def _candidate_stim_channels(raw):
    """Return stim channel names ranked by amount of state changes."""
    stim_picks = mne.pick_types(raw.info, stim=True)
    if len(stim_picks) == 0:
        return []

    scored = []
    for pick in stim_picks:
        channel_name = raw.ch_names[pick]
        stim_data = raw.get_data(picks=[pick])[0]
        change_count = int((stim_data[1:] != stim_data[:-1]).sum())
        nonzero_count = int((stim_data != 0).sum())
        scored.append((channel_name, change_count, nonzero_count))

    scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
    ranked_channels = [name for name, _, _ in scored]
    logger.info(f"Stim channel ranking: {ranked_channels}")
    return ranked_channels


def _estimate_min_duration(raw, channel_name):
    """Estimate a permissive min_duration from trigger transitions on a channel."""
    pick = raw.ch_names.index(channel_name)
    stim_data = raw.get_data(picks=[pick])[0]
    changes = (stim_data[1:] != stim_data[:-1]).nonzero()[0] + 1

    if len(changes) < 2:
        return 0.0

    durations = changes[1:] - changes[:-1]
    shortest = int(durations.min())
    min_duration = max(0.0, (shortest - 1) / raw.info["sfreq"])
    return min_duration


def handle_find_events(raw, stim_channel=None, min_events=5):
    """
    Detect events robustly across different systems and trigger channel layouts.

    Strategy order per channel:
    1) Default mne.find_events
    2) initial_event=True for channels with non-zero initial value
    3) shortest_event=1 for one-sample pulses
    4) min_duration estimated from observed transitions
    """
    channels = _normalize_channel_list(stim_channel)
    if not channels:
        channels = _candidate_stim_channels(raw)

    if not channels:
        raise RuntimeError("No stim channel found for event detection")

    best_events = None
    best_channel = None
    failures = []

    for ch in channels:
        min_duration = _estimate_min_duration(raw, ch)
        attempts = [
            {"stim_channel": ch},
            {"stim_channel": ch, "initial_event": True},
            {"stim_channel": ch, "shortest_event": 1},
            {"stim_channel": ch, "shortest_event": 1, "initial_event": True},
            {"stim_channel": ch, "min_duration": min_duration},
            {"stim_channel": ch, "min_duration": min_duration, "initial_event": True},
        ]

        logger.info(f"Trying event detection on stim channel: {ch}")

        for kwargs in attempts:
            try:
                events = mne.find_events(raw, **kwargs)
                logger.info(
                    "find_events succeeded on %s with %s (%d events)",
                    ch,
                    kwargs,
                    len(events),
                )

                if best_events is None or len(events) > len(best_events):
                    best_events = events
                    best_channel = ch

                if len(events) >= min_events:
                    return events
            except ValueError as err:
                failures.append((ch, kwargs, str(err)))

    if best_events is not None and len(best_events) > 0:
        logger.warning(
            "Using best available events from channel %s (%d events), below min_events=%d",
            best_channel,
            len(best_events),
            min_events,
        )
        return best_events

    failure_summary = "; ".join(
        f"ch={ch}, kwargs={kwargs}, err={err}" for ch, kwargs, err in failures[-5:]
    )
    raise RuntimeError(
        f"Event detection failed for channels={channels}. Last failures: {failure_summary}"
    )