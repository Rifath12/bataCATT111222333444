"""Single-channel dereverberation and speaker context for three-source file inference."""

import numpy as np
from scipy.signal import istft, stft


def dereverberate(mixture, delay=3, taps=8):
    """Single-channel delayed weighted linear prediction on the complex spectrum."""
    if len(mixture) < 640:
        return mixture.copy()
    _, _, spec = stft(mixture, nperseg=256, noverlap=192)
    lagged = np.stack(
        [
            np.pad(spec, ((0, 0), (i, 0)))[:, : spec.shape[1]]
            for i in range(delay, delay + taps)
        ],
        axis=1,
    )
    estimate = spec.copy()
    for _ in range(3):
        power = np.abs(estimate) ** 2
        variance = np.maximum(power, 1e-4 * power.mean(axis=1, keepdims=True) + 1e-12)
        design = lagged.conj() / variance[:, None, :]
        cov = design @ lagged.swapaxes(-1, -2)
        rhs = (design * spec[:, None, :]).sum(axis=-1)
        ridge = np.maximum(np.trace(cov, axis1=1, axis2=2).real / taps, 1e-8) * 0.015
        cov += np.eye(taps)[None] * ridge[:, None, None]
        weight = np.linalg.solve(cov, rhs[..., None]).squeeze(-1)
        estimate = spec - np.sum(lagged * weight[:, :, None], axis=1)
    repaired = istft(estimate, nperseg=256, noverlap=192)[1][: len(mixture)]
    return (0.85 * repaired + 0.15 * mixture).astype(np.float32)


def contextualize(mixture, cues, rate=8000):
    """Prepend a short solo sample for each voice; crop context before export."""
    cues_out = []
    for cue in cues:
        size = min(len(cue), int(0.68 * rate))
        if size == 0:
            raise ValueError("Empty speaker cue")
        envelope = np.r_[0.0, np.cumsum(cue**2, dtype=np.float64)]
        # Select one contiguous high-energy segment. Preserve its speaker identity.
        start = np.argmax(envelope[size:] - envelope[:-size])
        one = cue[start : start + size].copy()
        one *= min(0.06 / max(1e-8, np.sqrt(np.mean(one**2))), 8.0)
        edge = min(80, len(one) // 2)
        if edge:
            one[:edge] *= np.linspace(0.0, 1.0, edge)
            one[-edge:] *= np.linspace(1.0, 0.0, edge)
        cues_out.extend([one, np.zeros(int(0.08 * rate), np.float32)])
    prefix = np.concatenate(cues_out)
    return np.r_[prefix, mixture], len(prefix)
