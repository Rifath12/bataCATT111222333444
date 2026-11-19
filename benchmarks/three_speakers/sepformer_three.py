"""Load SpeechBrain's pinned three-speaker SepFormer weights for 8 kHz audio."""

import numpy as np
import torch


class SepFormer:
    def __init__(self, weights):
        from speechbrain.lobes.models.dual_path import (
            Decoder,
            Dual_Path_Model,
            Encoder,
            SBTransformerBlock,
        )

        def attention():
            return SBTransformerBlock(
                num_layers=8,
                d_model=256,
                nhead=8,
                d_ffn=1024,
                dropout=0,
                use_positional_encoding=True,
                norm_before=True,
            )

        self.encoder = Encoder(kernel_size=16, out_channels=256).eval()
        self.decoder = Decoder(
            in_channels=256, out_channels=1, kernel_size=16, stride=8, bias=False
        ).eval()
        self.mask = Dual_Path_Model(
            num_spks=3,
            in_channels=256,
            out_channels=256,
            num_layers=2,
            K=250,
            intra_model=attention(),
            inter_model=attention(),
            norm="ln",
            linear_layer_after_inter_intra=False,
            skip_around_intra=True,
        ).eval()
        modules = (
            (self.encoder, "encoder"),
            (self.decoder, "decoder"),
            (self.mask, "masknet"),
        )
        for model, name in modules:
            model.load_state_dict(
                torch.load(
                    weights / f"{name}.ckpt", map_location="cpu", weights_only=True
                )
            )

    @torch.inference_mode()
    def __call__(self, wave):
        features = self.encoder(torch.from_numpy(wave).reshape(1, -1))
        tracks = self.mask(features) * features[None]
        audio = self.decoder(tracks.squeeze(1)).cpu().numpy()
        if audio.shape[-1] < len(wave):
            audio = np.pad(audio, ((0, 0), (0, len(wave) - audio.shape[-1])))
        return audio[:, : len(wave)]
