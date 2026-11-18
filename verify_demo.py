"""Check the English speaker labels and timed Mandarin rows of the shipped demo."""
import argparse
import json
from pathlib import Path
import wave


def check(folder):
    manifest = json.loads((folder / "speakers.json").read_text(encoding="utf-8"))
    assert set(manifest["speakers"]) == {"spk_A","spk_B"}
    original = None
    for key in ("mixed.wav", "Speaker_A.wav", "Speaker_B.wav"):
        with wave.open(str(folder / key), "rb") as audio:
            properties = (audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getnframes())
            if original is None:
                original = properties
            assert properties == original
    rows = [json.loads(line) for line in (folder / "transcript.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {r["speaker_id"] for r in rows} == {"spk_A","spk_B"}
    for row in rows:
        assert row["timing"] == "estimated_ctc"
        assert any("\u4e00" <= c <= "\u9fff" for c in row["text"])
        chars = row["characters"]
        assert "".join(c["text"] for c in chars) == "".join(row["text"].split())
        assert all(row["start_ms"] <= c["start_ms"] <= c["end_ms"] <= row["end_ms"] for c in chars)
    assert (folder / "listen.html").is_file()
    print("Verified two synchronized PCM WAVs, speaker-labeled Mandarin text and character timestamps.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    check(parser.parse_args().folder)
