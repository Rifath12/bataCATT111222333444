"""Download and verify the public checkpoint files used in this demo."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import tarfile
import time
from tempfile import NamedTemporaryFile
import urllib.error
import urllib.request

BASE_URL="https://huggingface.co/JunzheJosephZhu/MultiDecoderDPRNN/resolve/cc9bb5b89e258919fa7893add25113e5552d849e/pytorch_model.bin"
TUNED_URL="https://huggingface.co/Namadgi/MultiDecoderDPRNN-finetuned/resolve/679e897ad3221b7b14598ce29fe874155cb6a5cc/weights/dprnn_rys_weights.pt"
RELEASE="https://github.com/k2-fsa/sherpa-onnx/releases/download"
VOICE_FILE="3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
VOICE_URL=(
    "https://huggingface.co/Maxkuo/eres2net-base-sv-zh-cn-onnx/resolve/"
    f"533e660fabc9553c0dc0cb5b395b24eac1ece0ad/{VOICE_FILE}"
)
ASR_FOLDER="sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03"
ASR_URL=f"{RELEASE}/asr-models/{ASR_FOLDER}.tar.bz2"
ASR_MIRROR=(
    f"https://huggingface.co/csukuangfj/{ASR_FOLDER}/resolve/"
    "824c678e9b80f301110e3ea8cee9819a7994c049"
)
ASR_ARCHIVE_SHA="f3ad1814fea34c407eab0cc3df6f6b625419ac9a60d8aebd8efe772a8e85ef67"
MODELS=(
    ("multi_decoder_dprnn.bin", BASE_URL, "ad367460c6641c6cd9441ada692c1e28fd9431525f52071f03851b928f659efb"),
    ("dprnn_rys_weights.pt", TUNED_URL, "6cee4cb59190c6fa67c38ec5f2d077bfce48a0cfca3ed93ab3cbdcc392df5827"),
    ("voiceprint.onnx", VOICE_URL, "1a331345f04805badbb495c775a6ddffcdd1a732567d5ec8b3d5749e3c7a5e4b"),
)
ASR_FILES={
    "model.int8.onnx": "e291b9c468b651e2697caa09bc684326c3addc6a019e78eb537cfd1a8248ca07",
    "tokens.txt": "6fed8c6c248516f38e7faa19404b57413e8ce259f1cbc1fa4aebc86eac32fdfd",
}


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def check(path, expected):
    return path.is_file() and digest(path)==expected


def fetch(path, url, expected):
    if check(path,expected):
        print("Verified",path.name,flush=True)
        return
    with NamedTemporaryFile(dir=path.parent, prefix=path.name+".", suffix=".part", delete=False) as pending:
        partial=Path(pending.name)
    try:
        for attempt in range(3):
            try:
                request=urllib.request.Request(url+"?download=true" if "huggingface.co" in url else url,
                                               headers={"User-Agent":"Mandarin overlap demo/1.0"})
                with urllib.request.urlopen(request,timeout=60) as remote, partial.open("wb") as target:
                    shutil.copyfileobj(remote,target,1024*1024)
                if not check(partial,expected):
                    raise ValueError(f"SHA-256 mismatch for {path.name}")
                partial.replace(path)
                print("Downloaded and verified",path.name,flush=True)
                break
            except (TimeoutError,OSError,ValueError,urllib.error.URLError):
                if attempt==2:
                    raise
                time.sleep(attempt+1)
    finally:
        partial.unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models",type=Path,default=Path(__file__).parent/"models")
    parser.add_argument("--audio-only",action="store_true")
    args=parser.parse_args()
    root=args.models
    root.mkdir(exist_ok=True,parents=True)
    for name,url,sha in MODELS:
        fetch(root/name,url,sha)
    if args.audio_only:
        return
    if all(check(root/ASR_FOLDER/name,sha) for name,sha in ASR_FILES.items()):
        print("Verified Mandarin ASR files.",flush=True)
        return
    archive=root/(ASR_FOLDER+".tar.bz2")
    (root/ASR_FOLDER).mkdir(exist_ok=True)
    if not check(archive, ASR_ARCHIVE_SHA):
        try:
            for name, sha in ASR_FILES.items():
                fetch(root/ASR_FOLDER/name, f"{ASR_MIRROR}/{name}", sha)
        except (OSError, ValueError) as error:
            print(f"Direct ASR download failed ({error}). Trying the verified release archive.", flush=True)
            fetch(archive, ASR_URL, ASR_ARCHIVE_SHA)
        else:
            print("Verified Mandarin ASR files.", flush=True)
            return
    with tarfile.open(archive,"r:bz2") as source:
        for member in source:
            name=Path(member.name).name
            if not member.isfile() or name not in ASR_FILES:
                continue
            output=root/ASR_FOLDER/name
            with NamedTemporaryFile(dir=output.parent, prefix=name+".", suffix=".part", delete=False) as target:
                partial=Path(target.name)
            try:
                with source.extractfile(member) as unpacked, partial.open("wb") as target:
                    shutil.copyfileobj(unpacked,target,1024*1024)
                if not check(partial,ASR_FILES[name]):
                    raise ValueError(f"SHA-256 mismatch for {name}")
                partial.replace(output)
            finally:
                partial.unlink(missing_ok=True)
    if not all(check(root/ASR_FOLDER/name,sha) for name,sha in ASR_FILES.items()):
        raise ValueError("The download does not contain all required ASR files.")
    print("Verified Mandarin ASR files.",flush=True)


if __name__=="__main__":
    main()
