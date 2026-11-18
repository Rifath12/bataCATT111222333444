"""Reference-guided CPU overlap extraction using the upstream WeSep weights."""

from pathlib import Path
import importlib
import sys
from types import ModuleType

import numpy as np
import torch
import yaml


class TargetExtractor:
    def __init__(self, model_dir, toolkit, wespeaker):
        # Load the upstream inference modules without the training and CLI
        # imports. The selected ECAPA implementation and weights are unchanged.
        for name, repository in (("wesep", toolkit), ("wespeaker", wespeaker)):
            namespace = ModuleType(name)
            namespace.__path__ = [str(Path(repository).resolve() / name)]
            sys.modules[name] = namespace
        only_ecapa = ModuleType("wespeaker.models.speaker_model")

        def speaker_factory(name):
            if name != "ECAPA_TDNN_GLOB_c512":
                raise KeyError(name)
            return importlib.import_module(
                "wespeaker.models.ecapa_tdnn"
            ).ECAPA_TDNN_GLOB_c512

        only_ecapa.get_speaker_model = speaker_factory
        sys.modules[only_ecapa.__name__] = only_ecapa
        from wesep.models import get_model

        model_dir = Path(model_dir)
        config = yaml.safe_load((model_dir / "config.yaml").read_text())
        config["model_args"]["tse_model"]["speaker"]["speaker_model"][
            "speaker_encoder"
        ]["pretrained"] = None
        model = get_model(config["model"]["tse_model"])(
            config["model_args"]["tse_model"]
        )
        weights = torch.load(
            model_dir / "avg_model.pt", weights_only=True, map_location="cpu"
        )
        weights = weights["models"][0]
        model.load_state_dict(weights, strict=True)
        self.model = model.eval()

    @torch.inference_mode()
    def anchors(self, wavs):
        return self.model.spk_ft.spkemb.compute(
            torch.from_numpy(np.stack(wavs)).float()
        )

    @torch.inference_mode()
    def extract(self, wave, cues):
        """Run the embedding-only WeSep path with precomputed speaker anchors."""
        model = self.model
        network = model.sep_model
        waves = torch.from_numpy(np.ascontiguousarray(wave, dtype=np.float32))[
            None
        ].expand(len(cues), -1)
        complex_mix = network.stft(waves)[-1]
        spec = torch.stack((complex_mix.real, complex_mix.imag), 1)
        feature = network.subband_norm(network.band_split(spec))
        feature = model.spk_ft.spkemb.post(feature, cues.unsqueeze(1).unsqueeze(3))
        out = network.band_masker(
            network.separator(feature), network.band_split(complex_mix)
        )
        complex_out = torch.complex(out[:, 0], out[:, 1])
        result = network.istft(complex_out)[:, 0].cpu().numpy()
        if result.shape[-1] < len(wave):
            result = np.pad(result, ((0, 0), (0, len(wave) - result.shape[-1])))
        return result[:, : len(wave)].copy()
