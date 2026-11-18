# Evaluation report

The first prototype was tested on three nominal counts using mixed Mandarin read speech and on public natural Mandarin conversations. It detected the correct count in three of three two-voice mixtures and two of three three-voice mixtures. All three controlled four-voice mixtures returned three tracks. One of two natural four-participant excerpts returned four tracks, with substantial crosstalk.

This branch adds an optional conservative-to-higher-count comparison, browser-playable PCM, an output report, and automated regression and demo checks. The default still uses the pretrained selector. Six local CPU regression checks and the bundled audio-to-text command passed, including two synchronized WAVs and valid Unicode Mandarin character timestamps. An expanded evaluation is running on independent speakers.
