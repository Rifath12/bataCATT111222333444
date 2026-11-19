"""Fold one audio file to mono without cancelling a duplicated, polarity-inverted recording."""
import numpy as np


def fold_channels(audio):
    """Return mono PCM for a finite [samples, channels] recording."""
    if audio.ndim != 2 or not audio.shape[0] or not audio.shape[1] or not np.isfinite(audio).all():
        raise ValueError("Expected nonempty, finite audio samples and at least one channel.")
    if audio.shape[1] == 1:
        return audio[:, 0]
    # Some recorders write a silent second channel even when one microphone is connected.
    active = np.flatnonzero(np.any(audio != 0, axis=0))
    if len(active) == 0:
        return np.zeros(audio.shape[0], dtype=audio.dtype)
    if len(active) == 1:
        return audio[:, active[0]]
    selected = audio if len(active) == audio.shape[1] else audio[:, active]
    if selected.shape[1] == 2:
        # Confirm a near-perfect polarity reversal over the full recording. Independent
        # microphones and ordinary stereo share neither this sign nor this correlation.
        gram = np.zeros((2, 2), dtype=np.float64)
        for start in range(0, len(selected), 65536):
            block = selected[start:start + 65536].astype(np.float64)
            gram += block.T @ block
        denominator = np.sqrt(gram[0, 0] * gram[1, 1])
        if denominator > 0 and gram[0, 1] < -.995 * denominator:
            return (selected[:, 0] - selected[:, 1]) * .5
    return selected.mean(axis=1)
