import numpy as np
from meegkit.dss import dss_line_iter
import mne
from mne.io import BaseRaw, RawArray
from scipy.signal import welch
import logging
logger = logging.getLogger(__name__)

def apply_zapline_denoising(
    raw,
    fline=50.0,
    n_chunks=10,
    spot_sz=7,
    win_sz=12,
    nfft=2048,
    n_iter_max=30,
    mag_only=True,
    detect_line_freq=True
):
    logger.info(f"Running apply_zapline_denoising")
    logger.debug(f"fline={fline}, n_chunks={n_chunks}")

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

    # Detect line frequency if enabled, otherwise use provided fline
    if detect_line_freq:
        sfreq = raw.info["sfreq"]

        freqs, psd = welch(
            data_to_denoise,
            fs=sfreq,
            axis=1,
            nperseg=int(sfreq * 4),
            nfft=nfft
        )

        mean_power = psd.mean(axis=0)

        mask = (freqs >= 48) & (freqs <= 62)

        # Power Check
        peak_power = mean_power[mask].max()
        median_power = np.median(mean_power[mask])
        if peak_power < 3 * median_power: # Threshold 3 is not empirically derived
            logger.warning(f"No clear line frequency peak detected. Peak power: {peak:.2f}, Median power: {median_power:.2f}. Defaulting to fline={fline} Hz.")
            line_freq = None
        else:
            line_freq = freqs[mask][np.argmax(mean_power[mask])]
            line_freq = np.round(line_freq, 2) # round to fit DSS input
            logger.info(f"Detected line frequency peak: {line_freq} Hz")
    else:
        line_freq = fline

    if line_freq is None:
        logger.warning("No sufficient line frequency detected. Skipping Zapline denoising to avoid DSS non-convergence.")
        return raw_copy

    sfreq = info["sfreq"]

    chunks = np.array_split(data_to_denoise, n_chunks, axis=1)
    cleaned_chunks = []

    for i, chunk in enumerate(chunks):
        logger.debug(f"Processing chunk {i+1}/{len(chunks)}")

        chunk_T = np.moveaxis(chunk, 0, -1)

        cleaned_T, _ = dss_line_iter(
            chunk_T,
            fline=line_freq,
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

    logger.info(f"ZAPLINED raw file.")

    return new_raw