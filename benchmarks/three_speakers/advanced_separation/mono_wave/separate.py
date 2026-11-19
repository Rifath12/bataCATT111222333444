"""Offline CPU adapter for pinned single-channel separators."""
from fractions import Fraction
import importlib.util
from pathlib import Path
import sys
from threading import Lock

import numpy as np
from scipy.signal import resample_poly
import torch


ROOT = Path(__file__).resolve().parent
MODEL_NAMES = {
    "recorrnet": "recorrnet-v2-1ch-aishell1mix-ver2-alimeeting",
    "sr_corrnet": "sr-corrnet-ss-1ch-aishell1mix-ver1",
    "sandglasset": "sandglasset-3spk-polyvox",
}
_MODEL_LOCK = Lock()


def _bound_batch(module, size):
    forward = module.forward

    def bounded(x):
        if x.shape[0] <= size:
            return forward(x)
        return torch.cat([forward(group) for group in x.split(size, dim=0)])

    module.forward = bounded


def _bound_corrnet(model, tf_type, speaker_type):
    for module in model.modules():
        if isinstance(module, tf_type):
            _bound_batch(module.freq_block.block, 24)
            _bound_batch(module.time_block.block, 24)
        elif isinstance(module, speaker_type):
            _bound_batch(module.block, 2048)

    split = model.spk_split

    def separate_active(x, n_spks=None, prob_thres=0.5):
        # Preserve the original global attractor computation for the entire
        # clip. Only fuse selected attractors, using independent time batches.
        # Other slots feed a training residual loss and are ignored in audio.
        assert x.shape[0] == 1
        _, time, freq, channels = x.shape
        if time > split.pe_tf.shape[0] or freq > split.pe_tf.shape[1]:
            split.pe_tf = split.get_2d_sinusoids(
                max(time, split.pe_tf.shape[0]), max(freq, split.pe_tf.shape[1]), channels
            )
        positions = split.pe_tf[:time, :freq].unsqueeze(0).to(x)
        keys = (split.layernorm(x) + positions).reshape(1, -1, channels)
        attractors = split.dec(split.spk_query, keys)
        attractors = attractors / (attractors.norm(dim=-1, keepdim=True) + 1e-8)
        logits = split.pres_linear(attractors).squeeze(-1)
        probabilities = logits.sigmoid()
        if n_spks is None:
            count = max(1, int((probabilities[0, 1 : 1 + split.max_n_spks] > prob_thres).sum()))
        else:
            count = int(n_spks.item())

        selected = attractors[:, 1 : 1 + count]
        separated = torch.cat([split.split(group, selected) for group in x.split(24, dim=1)], dim=2)
        return separated, {"logits": logits, "probs": probabilities, "split_res": None}, None

    split.forward = separate_active


def _corrnet(folder, source, dependencies, bounded, flash_attention):
    for directory in (dependencies, source):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    from sr_corrnet.models.SR_CorrNet_SS.engine_infer import EngineInfer
    from sr_corrnet.models.SR_CorrNet_SS.model import Model
    from sr_corrnet.models.SR_CorrNet_SS.modules.module import AttractorSplit, CrossSpkBlock, TF_Block
    from sr_corrnet.utils.util_engine import _fix_compiled_state_dict, _migrate_state_dict
    import yaml

    with (folder / "config.yaml").open(encoding="utf-8") as fp:
        conf = yaml.safe_load(fp)["config"]
    # The full preallocated grid is ~3.84 GB. Restrict the seed grid
    # during model initialization only. The normal forward path grows it
    # to the observed shape using the exact same analytical sinusoid.
    with _MODEL_LOCK:
        original = AttractorSplit.get_2d_sinusoids

        def initial_grid(self, time, frequency, channels, *extra):
            return original(self, min(time, 32), min(frequency, 32), channels, *extra)

        try:
            AttractorSplit.get_2d_sinusoids = initial_grid
            net = Model(**conf["model"])
        finally:
            AttractorSplit.get_2d_sinusoids = original

    raw = torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
    state = _migrate_state_dict(_fix_compiled_state_dict(raw.get("model_state_dict", raw)))
    net.load_state_dict(state, strict=True)
    if bounded:
        _bound_corrnet(net, TF_Block, CrossSpkBlock)
    if flash_attention:
        for block in net.modules():
            config = getattr(block, "flash_attention_config", None)
            if config is not None:
                # Recent CPU builds dispatch a memory-bounded attention kernel.
                # Keep the mathematical implementation available as a fallback.
                config["enable_flash"] = True
    return EngineInfer(conf, net, torch.device("cpu"))


def _sandglass(folder):
    spec = importlib.util.spec_from_file_location(
        "vnext_sandglasset", folder / "modeling_sandglasset.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    state = torch.load(folder / "best_model.pt", map_location="cpu", weights_only=True)
    model = module.SandglassetStudent(n_src=3)
    model.load_state_dict(state["state_dict"], strict=True)
    return model.eval()


def _convert_rate(wave, source, target):
    if source == target:
        return wave
    ratio = Fraction(target, source)
    return resample_poly(wave, ratio.numerator, ratio.denominator, axis=-1).astype(
        np.float32, copy=False
    )


class Separator:
    """Return unordered separator stems at the caller's input rate.

    Pass a known or estimated count to select that many CorrNet attractors.
    With `speakers=None`, preserve the independent count selected by the
    checkpoint's presence head. Sandglasset always returns exactly three.
    """

    def __init__(self, name, *, model_dir=None, source_dir=None, dependencies_dir=None,
                 bounded=True, flash_attention=False):
        if name not in MODEL_NAMES:
            raise ValueError(f"Unknown separator {name!r}")
        self.name = name
        folder = Path(model_dir) if model_dir else ROOT / "models" / MODEL_NAMES[name]
        if name == "sandglasset":
            self.network = _sandglass(folder)
            self.native_rate = 16000
        else:
            source = Path(source_dir) if source_dir else ROOT / "reference"
            dependencies = Path(dependencies_dir) if dependencies_dir else ROOT / "deps"
            self.network = _corrnet(folder, source, dependencies, bounded, flash_attention)
            self.native_rate = self.network.fs
            if self.native_rate != 8000:
                raise ValueError(f"Unexpected CorrNet checkpoint rate: {self.native_rate}")

    @torch.inference_mode()
    def __call__(self, wave, sample_rate=16000, speakers=3, *, return_first_pass=False):
        wave = np.asarray(wave, dtype=np.float32)
        if wave.ndim != 1 or len(wave) == 0 or not np.isfinite(wave).all():
            raise ValueError("Expected a finite non-empty mono waveform")
        if not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ValueError("Expected a positive integer sample rate")
        if speakers is not None and (not isinstance(speakers, int) or not 1 <= speakers <= 5):
            raise ValueError("Speaker count must be 1–5 or None")
        if self.name == "sandglasset" and speakers not in (None, 3):
            raise ValueError("This Sandglasset checkpoint is fixed at three tracks")

        native = _convert_rate(wave, sample_rate, self.native_rate)
        if not np.any(native):
            count = 3 if self.name == "sandglasset" else speakers or 1
            tracks = np.zeros((count, len(wave)), np.float32)
            return (tracks, None) if return_first_pass else tracks
        source = torch.from_numpy(np.pad(native, (0, max(0, 128 - len(native))))).unsqueeze(0)
        first_pass = None
        if self.name == "sandglasset":
            outputs = self.network(source)[0]
        else:
            std = source.std() + 1e-8
            mixture = self.network.stft(source / std, cplx=True)
            both = torch.cat((mixture.real, mixture.imag), dim=0)
            n = torch.tensor([speakers]) if speakers is not None else None
            frames, _, _ = self.network.model(both, n_spks=n)
            outputs = self._waveforms(frames) * std
            if return_first_pass and self.name == "recorrnet":
                initial = (self.network.model.last_recorr or {}).get("est_pass1")
                if initial is not None:
                    first_pass = self._adapt_output(self._waveforms(initial) * std, sample_rate, len(wave))
        tracks = self._adapt_output(outputs, sample_rate, len(wave))
        if speakers is not None and len(tracks) != speakers:
            raise RuntimeError(f"Expected {speakers} tracks, got {len(tracks)}")
        return (tracks, first_pass) if return_first_pass else tracks

    def _waveforms(self, frames):
        images = (
            torch.complex(f[:, self.network.ref_ch, ..., 0], f[:, self.network.ref_ch, ..., 1])
            for f in frames
        )
        return torch.stack([self.network.istft(x, cplx=True, squeeze=True) for x in images])

    def _adapt_output(self, outputs, rate, length):
        tracks = _convert_rate(outputs.float().cpu().numpy(), self.native_rate, rate)
        if tracks.shape[1] < length:
            tracks = np.pad(tracks, ((0, 0), (0, length - tracks.shape[1])))
        if not np.isfinite(tracks).all():
            raise RuntimeError("Separator output contains non-finite samples")
        return tracks[:, :length]
