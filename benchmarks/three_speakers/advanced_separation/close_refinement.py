"""Optional low-strength waveform correction for synchronized close microphones."""

from pathlib import Path
import json

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter

RATE = 16000
FFT = 512
HOP = 128
BLEND = 0.25
MODEL = Path(__file__).with_name("close_refinement_weights.json")

def stft(audio):
    return signal.stft(audio, fs=RATE, nperseg=FFT, noverlap=FFT-HOP, axis=-1)[2]


def istft(spectrum, size):
    return signal.istft(spectrum, fs=RATE, nperseg=FFT, noverlap=FFT-HOP, time_axis=-1, freq_axis=-2)[1][..., :size]


def smooth(array, bandwidth, frames):
    return uniform_filter(array, size=(1, bandwidth, frames), mode="nearest")


def features(spectrum, mic):
    power = np.abs(spectrum) ** 2
    local = smooth(power, 3, 7)
    wide = smooth(power, 13, 19)
    voice = power.mean(axis=-2, keepdims=True)
    mic_power = local[mic]
    others = [index for index in range(3) if index != mic]
    norm = np.sqrt(np.maximum(mic_power, 2e-11))
    wanted = spectrum[mic]
    recipe = [np.log(np.maximum(mic_power, 1e-14))*0.2]
    # Linear subtraction basis remains tied to the dominant measured cross-phase.
    basis = [wanted/norm]
    for other in others:
        weight = local[other] + 1e-12
        cross = smooth((wanted*spectrum[other].conj())[None,:,:], 1, 13)[0]
        coherent = cross / np.sqrt(np.maximum(mic_power*weight,1e-21))
        projected = cross/(smooth(power[other:other+1], 1, 13)[0]+1e-11)*spectrum[other]/norm
        projected *= np.minimum(1., 2.5/np.maximum(np.abs(projected),1e-8))
        basis.extend((projected, 1j*projected))
        recipe.extend((
            np.log(weight / np.maximum(mic_power,1e-12))*0.3,
            np.log((wide[other]+1e-12)/(wide[mic]+1e-12))*0.3,
            np.broadcast_to(np.log((voice[other]+1e-12)/(voice[mic]+1e-12))*0.30, wanted.shape),
            coherent.real,
            coherent.imag,
            np.abs(coherent),
            np.abs(projected),
        ))
    frequency = np.linspace(0, 1, spectrum.shape[1])[:, None]
    recipe.extend((np.broadcast_to(frequency,wanted.shape), np.broadcast_to(np.log1p(frequency*80)/4,wanted.shape)))
    for idx in range(3):
        recipe.append(np.full(wanted.shape, float(mic==idx)))
    recipe.append(np.log((power[mic]+1e-12)/(mic_power+1e-12))*0.25)
    return (
        np.clip(np.stack(recipe,axis=-1).reshape(-1,len(recipe)), -6, 6).astype(np.float32),
        np.stack(basis,axis=-1).reshape(-1,5).astype(np.complex64),
        norm,
    )


def output_spectrum(raw, basis):
    logits=np.clip(raw[:,0],-12.,12.)
    sigmoid=1/(1+np.exp(-logits))
    gain=.07+.93*sigmoid
    tanh=np.tanh(raw[:,1:])
    weights=.50*tanh
    estimate=basis[:,0]*gain+np.sum(basis[:,1:]*weights,axis=1)
    return estimate, sigmoid, tanh


def forward(X, params):
    w1,b1,w2,b2,w3,b3=params
    H=X@w1+b1
    D=np.maximum(H,0.)
    G=D@w2+b2
    E=np.maximum(G,0.)
    return E@w3+b3,H,D,G,E


def predict(audio,params,protect=True):
    z=stft(audio)
    outputs=[]
    for mic in range(3):
        feature,basis,norm=features(z,mic)
        raw=forward(feature,params)[0]
        values=output_spectrum(raw,basis)[0].reshape(norm.shape)*norm
        if protect:
            # A strong physical own microphone is only lightly affected.
            own=np.mean(np.abs(z[mic])**2,axis=0)
            levels=np.mean(np.abs(z)**2,axis=1)
            strongest=np.max(np.delete(levels,mic,axis=0),axis=0)
            safe=np.clip(1.-0.12*own/np.maximum(strongest,1e-12),.08,1.)
            values=z[mic]+safe[None,:]*(values-z[mic])
        outputs.append(values)
    return istft(np.stack(outputs),audio.shape[-1])


def load_parameters(path=MODEL):
    """Read and validate the bundled numerical weights."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    arrays = tuple(np.asarray(value, dtype=np.float32) for value in payload["parameters"])
    expected = ((21, 40), (40,), (40, 32), (32,), (32, 5), (5,))
    if (
        payload["network_version"] != 1
        or tuple(value.shape for value in arrays) != expected
        or any(not np.isfinite(value).all() for value in arrays)
    ):
        raise ValueError("Invalid close-microphone refinement parameters")
    return arrays


def refine(audio, baseline, parameters):
    """Blend with the existing waveform, preserving length and channel identity."""
    if audio.ndim != 2 or audio.shape != baseline.shape or audio.shape[0] != 3:
        raise ValueError("Expected three aligned direct microphones and baseline tracks")
    if audio.shape[1] < FFT:
        return baseline
    learned = predict(audio, parameters)
    if learned.shape != baseline.shape or not np.isfinite(learned).all():
        raise ValueError("The close-microphone refinement produced invalid waveforms")
    result = baseline + BLEND * (learned - baseline)
    return result.astype(baseline.dtype, copy=False)
