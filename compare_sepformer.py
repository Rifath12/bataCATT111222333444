"""Optional offline comparison for a clip known to contain four sources."""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from speechbrain.lobes.models.dual_path import Decoder, Dual_Path_Model, Encoder, SBTransformerBlock

from benchmark_examples import save_output, text_by_track
from process_meeting import Mandarin, RATE, read_audio, resample


def transformer():
    return SBTransformerBlock(num_layers=8, d_model=256, nhead=8, d_ffn=1024, dropout=0,
                              use_positional_encoding=True, norm_before=True)


def separate(y, weights):
    encoder = Encoder(kernel_size=16, out_channels=256).eval()
    decoder = Decoder(in_channels=256, out_channels=1, kernel_size=16, stride=8, bias=False).eval()
    masknet = Dual_Path_Model(num_spks=4, in_channels=256, out_channels=256, num_layers=2, K=250,
                             intra_model=transformer(), inter_model=transformer(), norm="ln",
                             linear_layer_after_inter_intra=False, skip_around_intra=True).eval()
    for module, name in ((encoder, "encoder"), (decoder, "decoder"), (masknet, "masknet")):
        module.load_state_dict(torch.load(weights / (name + ".ckpt"), weights_only=True, map_location="cpu"))
    x = torch.from_numpy(resample(y, RATE, 48000))[None]
    with torch.inference_mode():
        encoded = encoder(x)
        wave = decoder((masknet(encoded) * encoded).squeeze(1)).numpy()
    wave = resample(wave, 48000, RATE)
    wave = np.pad(wave, ((0, 0), (0, max(0, len(y) - wave.shape[-1]))))[:, :len(y)]
    gains = np.linalg.lstsq(wave.T, y, rcond=0.02)[0]
    return {f"spk_{chr(65+i)}": signal for i, signal in enumerate(wave * gains[:, None])}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("--weights", required=True, type=Path)
    p.add_argument("--models", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    torch.set_num_threads(4)
    y,_,_=read_audio(args.input)
    tracks = separate(y, args.weights)
    rows, _ = text_by_track(Mandarin(args.models, 4), tracks)
    args.out.mkdir(parents=True)
    save_output(args.out, tracks, y, [], rows)


if __name__ == "__main__":
    main()
