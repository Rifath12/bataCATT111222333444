"""Run official ESPnet TD-SpeakerBeam with a separate audio cue for each target."""
from pathlib import Path
import argparse
import hashlib
import numpy as np
import soundfile as sf
import torch
import yaml
from scipy.signal import resample_poly

from espnet2.enh.encoder.conv_encoder import ConvEncoder
from espnet2.enh.decoder.conv_decoder import ConvDecoder
from espnet2.enh.extractor.td_speakerbeam_extractor import TDSpeakerBeamExtractor

def enrollment_wave(wave):
    x = resample_poly(wave, 1, 2).astype(np.float32)
    blocks = x[:len(x)//256*256].reshape(-1,256)
    rms = np.sqrt((blocks**2).mean(1))
    x = blocks[rms>=max(10**(-45/20),.02*rms.max())].reshape(-1)[:48000]
    if len(x)<256: raise ValueError('Silent enrollment')
    return torch.from_numpy(x).unsqueeze(0)

class Model(torch.nn.Module):
    def __init__(self, config, path):
        super().__init__()
        self.encoder = ConvEncoder(**config['encoder_conf'])
        self.extractor = TDSpeakerBeamExtractor(self.encoder.output_dim, **config['extractor_conf'])
        self.decoder = ConvDecoder(**config['decoder_conf'])
        self.load_state_dict(torch.load(path,weights_only=True,map_location='cpu'),strict=True)
        self.eval()

    @torch.inference_mode()
    def extract_targets(self, waveform, embeddings):
        half = resample_poly(waveform,1,2).astype(np.float32)
        x = torch.from_numpy(half).unsqueeze(0)
        lens = torch.tensor([x.shape[-1]])
        features, flens = self.encoder(x,lens)
        tcn = self.extractor.tcn
        features_t = features.transpose(1,2)
        outputs=[]
        for emb in embeddings:
            # The extractor's public forward applies this same mask to mixture features.
            mask=tcn(features_t,emb).transpose(-1,-2)
            pred=self.decoder(features*mask,lens)[0][0].numpy()
            outputs.append(resample_poly(pred,2,1)[:len(waveform)])
        return np.stack(outputs)

    @torch.inference_mode()
    def prepare(self, enrollment):
        embeddings=[]
        for wave in enrollment:
            e=enrollment_wave(wave)
            ef,elen=self.encoder(e,torch.tensor([e.shape[-1]]))
            emb=self.extractor.auxiliary_net(ef.transpose(1,2)).squeeze(1)
            assert emb.shape[-1]==elen.item()
            embeddings.append(emb.mean(-1))
        return embeddings


WEIGHTS_SHA256 = "7e1d6831d0df187ae8b23ed4d6804d07d0ec314336266e99f7da41a2f8ad055f"
CONFIG_SHA256 = "8f73655ebb6c26fbe065d98ca3479efd27a8830eb2589bc0ecf8c6930809e7fb"

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_wav", type=Path)
    parser.add_argument("--enroll", type=Path, nargs=3, required=True, metavar=("A", "B", "C"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    def audio(path):
        wave, rate = sf.read(path, dtype="float32")
        if rate != 16000 or wave.ndim != 1 or not np.isfinite(wave).all():
            raise ValueError("Supply a finite mono WAV sampled at 16 kHz")
        return wave
    def verify(path, digest):
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                raise ValueError(f"Unexpected file digest for {path}")
    verify(args.checkpoint, WEIGHTS_SHA256)
    verify(args.config, CONFIG_SHA256)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model = Model(yaml.safe_load(args.config.read_text()), args.checkpoint)
    embeddings = model.prepare([audio(path) for path in args.enroll])
    mix = audio(args.input_wav)
    if len(mix) < 64:
        raise ValueError("Input must contain at least 64 samples")
    output = model.extract_targets(mix, embeddings)
    if output.shape != (3, len(mix)) or not np.isfinite(output).all():
        raise RuntimeError("Invalid output shape or samples")
    args.out.mkdir(parents=True, exist_ok=False)
    for name, wave in zip("ABC", output, strict=True):
        sf.write(args.out / f"Speaker_{name}.wav", wave, 16000, subtype="FLOAT")

if __name__ == "__main__":
    main()
