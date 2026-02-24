import numpy as np
import mne
from mne import find_events
from meegkit.dss import dss_line_iter
from mne.io import BaseRaw, RawArray
# logging
import logging
logger = logging.getLogger(__name__)

def crop_data(in_file, stim_channel, min_buffer, max_buffer):
    logger.info(f"Cropping file: {in_file}")
    logger.debug(f"Stim channel: {stim_channel}")

    raw = mne.io.read_raw_fif(in_file, preload=True)

    # crop logic here
    events = find_events(raw, stim_channel=stim_channel, shortest_event=1)
    tmin = raw.times[events[0][0]] + min_buffer
    tmin = max(0.0, tmin)  # Ensure tmin >= 0
    tmax = raw.times[events[-1][0]] + max_buffer
    tmax = min(tmax, raw.times[-1])  # Ensure tmax <= data length
    cropped = raw.copy().crop(tmin=tmin, tmax=tmax)

    out_file = "cropped_raw.fif"   # no path needed
    cropped.save(out_file, overwrite=True)

    logger.info(f"Saved cropped file to {out_file}")

    return out_file

def filter_data(in_file, l_freq, h_freq):
    import mne
    logger.info(f"Filtering file: {in_file}")
    logger.debug(f"Band-pass: {l_freq}-{h_freq} Hz")

    raw = mne.io.read_raw_fif(in_file, preload=True)

    filtered = raw.copy().filter(l_freq, h_freq)

    out_file = "filtered_raw.fif"
    filtered.save(out_file, overwrite=True)

    logger.info(f"Saved filtered file to {out_file}")

    return out_file

def gradient_compensation(in_file, auto=True, order=3):
    import mne
    logger.info(f"Applying gradient compensation to: {in_file}")

    raw = mne.io.read_raw_fif(in_file, preload=True)
    raw_copy = raw.copy()

    if auto:
        comps = raw_copy.info.get("comps", [])
        if comps:
            k = max(range(len(comps)))
            logger.debug(f"Auto gradient compensation order: {k}")
            raw_copy.apply_gradient_compensation(k)
        else:
            logger.warning("No gradient compensation matrices available.")
    else:
        logger.debug(f"Manual gradient compensation order: {order}")
        raw_copy.apply_gradient_compensation(order)

    out_file = "gradcomp_raw.fif"
    raw_copy.save(out_file, overwrite=True)

    logger.info(f"Saved gradient compensated file to {out_file}")

    return out_file
    
def set_channels(in_file, ch_dict: dict):
    logger.info(f"Setting channel types for: {in_file}")
    logger.debug(f"Channel dict: {ch_dict}")

    raw = mne.io.read_raw_fif(in_file, preload=True)
    raw_copy = raw.copy()

    invalid_channels = set(ch_dict.keys()) - set(raw_copy.ch_names)
    if invalid_channels:
        raise ValueError(f"Channels do not exist: {invalid_channels}")

    try:
        raw_copy.set_channel_types(ch_dict)
    except (TypeError, ValueError) as e:
        raise ValueError(
            "Invalid channel types or channels do not exist"
        ) from e

    out_file = "setchannels_raw.fif"
    raw_copy.save(out_file, overwrite=True)

    logger.info(f"Saved channel-modified file to {out_file}")

    return out_file

def apply_zapline_denoising(
    in_file,
    fline=50.0,
    n_chunks=10,
    spot_sz=7,
    win_sz=12,
    nfft=2048,
    n_iter_max=30,
    mag_only=True
):
    logger.info(f"Applying ZAPLINE denoising to: {in_file}")
    logger.debug(f"fline={fline}, n_chunks={n_chunks}")

    raw = mne.io.read_raw_fif(in_file, preload=True)
    raw_copy = raw.copy().load_data()

    raw_data = raw_copy.get_data()
    info = raw_copy.info

    # Select channels
    if mag_only:
        mag_ix = np.array([
            i for i, ch_type in enumerate(raw_copy.get_channel_types())
            if ch_type == "mag"
        ])
        if len(mag_ix) == 0:
            raise ValueError("No magnetometer channels found.")
        data_to_denoise = raw_data[mag_ix]
    else:
        mag_ix = np.arange(raw_data.shape[0])
        data_to_denoise = raw_data

    sfreq = info["sfreq"]

    chunks = np.array_split(data_to_denoise, n_chunks, axis=1)
    cleaned_chunks = []

    for i, chunk in enumerate(chunks):
        logger.debug(f"Processing chunk {i+1}/{len(chunks)}")

        chunk_T = np.moveaxis(chunk, 0, -1)

        cleaned_T, _ = dss_line_iter(
            chunk_T,
            fline=fline,
            sfreq=sfreq,
            spot_sz=spot_sz,
            win_sz=win_sz,
            nfft=nfft,
            n_iter_max=n_iter_max
        )

        cleaned = np.moveaxis(cleaned_T, -1, 0)
        cleaned_chunks.append(cleaned)

    cleaned_data = np.hstack(cleaned_chunks)

    new_data = raw_copy.get_data()
    new_data[mag_ix, :] = cleaned_data

    new_raw = RawArray(
        new_data,
        info,
        first_samp=raw_copy.first_samp
    )

    out_file = "zapline_raw.fif"
    new_raw.save(out_file, overwrite=True)

    logger.info(f"Saved ZAPLINE cleaned file to {out_file}")

    return out_file