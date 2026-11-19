"""Overlap-only multichannel covariance separation with fixed speaker enrollment."""

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.special import softmax


class AdaptiveArrayMask:
    """Estimate additive source images at the reference microphone.

    Learn shape matrices from predicted solo speech and background. For
    overlapping frames, estimate the nonnegative shape weights and reconstruct
    their channel-zero contributions. Solo frames retain the original beams.
    """

    iterations = 8
    # Each power update averages three frames after the likelihood averages
    # three frames once. Computing both halos makes chunk boundaries immaterial.
    context_frames = iterations + 1

    def __init__(self, covariance, activity, spatial_baseline):
        self.activity = activity
        self.likelihood_shapes = spatial_baseline.inverse
        self.log_determinants = spatial_baseline.log_determinants
        channels = covariance.shape[-1]
        self.identity = np.eye(channels, dtype=np.complex64)

        noise = covariance[3].astype(np.complex64)
        speech = (covariance[:3] - 0.65 * covariance[3]).astype(np.complex64)
        eigenvalues, eigenvectors = np.linalg.eigh(speech)
        speech_level = np.maximum(eigenvalues.sum(axis=-1) / channels, 1e-10)
        eigenvalues = np.maximum(eigenvalues, 0.02 * speech_level[..., None])
        speech = (eigenvectors * eigenvalues[..., None, :]) @ eigenvectors.conj().swapaxes(-1, -2)
        noise_level = np.maximum(noise.trace(axis1=-2, axis2=-1).real * 0.1 / channels, 1e-9)
        noise += noise_level[..., None, None] * self.identity
        if np.max(np.abs(covariance[3])) < 1e-10:
            # The control signals have digital silence in the background.
            noise[:] = self.identity
        self.shapes = np.concatenate((speech, noise[None]))
        traces = self.shapes.trace(axis1=-2, axis2=-1).real
        self.shapes /= np.maximum(traces[..., None, None], 1e-12)

    def spatial_prior(self, spectrum, activity):
        channels = spectrum.shape[-1]
        energy = np.maximum(np.sum(np.abs(spectrum) ** 2, axis=-1), 1e-14)
        logits = []
        for metric, logdet in zip(self.likelihood_shapes, self.log_determinants, strict=True):
            transformed = np.einsum("tfc,fcd->tfd", spectrum.conj(), metric, optimize=True)
            distance = np.einsum("tfd,tfd->tf", transformed, spectrum, optimize=True).real / energy
            logits.append(-channels * np.log(np.maximum(distance, 1e-5)) - logdet)
        logits = np.stack(logits, axis=1)
        logits = uniform_filter1d(logits, size=3, axis=0, mode="nearest")
        logits = uniform_filter1d(logits, size=3, axis=-1, mode="nearest")
        logits[:, :3] -= 20.0 * (~activity)[:, :, None]
        logits[:, 3] -= 3.5
        weights = 0.96 * softmax(logits / 1.8, axis=1) + 0.01
        weights[:, :3] *= activity[..., None]
        return weights / weights.sum(axis=1, keepdims=True)

    def reconstruct(self, spectrum, prior, active):
        supported = np.column_stack((active, np.ones(len(active), dtype=bool)))[..., None]
        energy = np.maximum(np.sum(np.abs(spectrum) ** 2, axis=-1, keepdims=True), 1e-12)
        normalized = spectrum / np.sqrt(energy)
        weights = prior.copy()
        ridge = self.identity * 0.0005
        for _ in range(self.iterations):
            covariance = np.einsum("tsf,sfij->tfij", weights, self.shapes, optimize=True)
            inverse = np.linalg.inv(covariance + ridge)
            fitted = (inverse @ normalized[..., None])[..., 0]
            quadratic = np.einsum("tfi,sfij,tfj->tsf", fitted.conj(), self.shapes, fitted).real
            trace = np.einsum("tfij,sfji->tsf", inverse, self.shapes).real
            update = np.sqrt(np.maximum(quadratic, 1e-6) / np.maximum(trace, 1e-6))
            weights = np.maximum(weights * np.clip(update, 0.25, 4.0), 1e-5) * supported
            weights /= weights.sum(axis=1, keepdims=True)
            weights = uniform_filter1d(weights, size=3, axis=0, mode="nearest") * supported
            weights /= weights.sum(axis=1, keepdims=True)
            weights = 0.975 * weights + 0.025 * prior
        covariance = np.einsum("tsf,sfij->tfij", weights, self.shapes, optimize=True)
        fitted = np.linalg.solve(covariance + ridge, spectrum[..., None])[..., 0]
        return np.einsum("tsf,sfj,tfj->tsf", weights[:, :3], self.shapes[:3, :, 0], fitted, optimize=True)

    def apply(self, spectrum, baseline, start, stop, *, spectrum_start=None):
        if not np.any(self.activity[:, start:stop].sum(axis=0) >= 2):
            return baseline
        offset = start if spectrum_start is None else spectrum_start
        active = self.activity[:, offset:offset + len(spectrum)].T
        overlap = active.sum(axis=1) >= 2
        prior = self.spatial_prior(spectrum, active)
        edges = np.flatnonzero(np.diff(np.r_[False, overlap, False]))
        for first, last in zip(edges[::2], edges[1::2], strict=True):
            left = max(first, start - offset)
            right = min(last, stop - offset)
            if left >= right:
                continue
            support_left = max(first, left - self.iterations)
            support_right = min(last, right + self.iterations)
            span = slice(support_left, support_right)
            estimate = self.reconstruct(spectrum[span], prior[span], active[span])
            baseline[left + offset - start:right + offset - start] = estimate[left - support_left:right - support_left]
        return baseline
