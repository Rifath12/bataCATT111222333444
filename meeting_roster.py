"""Track an enrolled A/B/C roster through changes of decoder slot and active speaker."""
from itertools import permutations
from math import gcd

import numpy as np
from scipy.signal import resample_poly

from benchmarks.three_speakers.advanced_separation.mono_wave.enrolled_order import (
    SAMPLE_RATE, _envelope, fit_templates,
)

RATE = 8000
STEP = 2048
CONTEXT = 1024


class EnrolledRoster:
    def __init__(self, paths):
        import soundfile as sf
        from audio_input import fold_channels

        examples = []
        for path in paths:
            samples, rate = sf.read(path, dtype="float32", always_2d=True)
            wave = fold_channels(samples)
            if rate != SAMPLE_RATE:
                divisor = gcd(rate, SAMPLE_RATE)
                wave = resample_poly(wave, SAMPLE_RATE // divisor, rate // divisor)
            examples.append([wave])
        self.templates = fit_templates(examples, SAMPLE_RATE)
        if any(np.allclose(self.templates[i], self.templates[j]) for i in range(3) for j in range(i)):
            raise ValueError("Supply distinct isolated enrollment recordings for A, B and C")

    def route(self, tracks, begin, end):
        """Assign each decoded slot exactly once per tile on the current block's clock."""
        count = len(tracks)
        if count == 0:
            return np.zeros((3, end - begin), np.float32), []
        if not 1 <= count <= 3:
            raise ValueError("Enrolled A/B/C routing needs one to three decoded tracks")
        orders = np.asarray(list(permutations(range(3), count)))
        slots = np.arange(count)
        edges = list(range(begin, end, STEP)) + [end]
        distances = []
        activity = []
        for left, right in zip(edges, edges[1:]):
            start = max(0, left - CONTEXT)
            stop = min(tracks.shape[1], right + CONTEXT)
            tile = tracks[:, left:right]
            rms = np.sqrt(np.mean(tile.astype(np.float64)**2, axis=1))
            active = (rms > .00025) & (rms > .012 * rms.max())
            activity.append(active)
            cost = np.zeros((count, 3))
            for source in np.flatnonzero(active):
                snippet = tracks[source, start:stop]
                if snippet.size < 256:
                    continue
                try:
                    observed = _envelope(resample_poly(snippet.astype(np.float64), 2, 1))
                except ValueError:
                    continue
                delta = np.linalg.norm(self.templates - observed, axis=1)
                cost[source] = (delta - delta.min()) / max(float(delta.mean()), 1e-7)
            distances.append(cost[slots, orders].sum(axis=1))
        # Prefer consistent identity through adjacent audible tiles. A true pause
        # releases the previous mapping so a returning speaker can take any slot.
        cost = distances[0].copy()
        parents = []
        switches = orders[:, None, :] != orders[None, :, :]
        for n in range(1, len(distances)):
            continuing = activity[n - 1] & activity[n]
            transition = .15 * np.sum(switches * continuing, axis=2)
            previous = np.argmin(cost[:, None] + transition, axis=0)
            cost = cost[previous] + transition[previous, np.arange(len(orders))] + distances[n]
            parents.append(previous)
        states = [int(np.argmin(cost))]
        for previous in reversed(parents):
            states.append(int(previous[states[-1]]))
        states.reverse()
        result = np.zeros((3, end - begin), np.float32)
        records = []
        for n, (left, right) in enumerate(zip(edges, edges[1:])):
            order = orders[states[n]]
            active = activity[n]
            different = np.any(orders[:, active] != order[active], axis=1)
            margin = (float(distances[n][different].min() - distances[n][states[n]])
                      if different.any() else None)
            result[order, left - begin:right - begin] = tracks[:, left:right]
            records.append({
                "start_sample": left - begin, "end_sample": right - begin,
                "active_slots": sorted(order[active].tolist()),
                "assignment_margin": None if margin is None else round(margin, 5),
                "identity_needs_review": bool(margin is not None and margin < .05),
            })
            if n and states[n] != states[n - 1]:
                # Crossfade only the assignment at a switching boundary. All input
                # samples stay on the same clock and the sum of tracks is preserved.
                width = min(32, (right - left) // 2)
                if width:
                    old = np.zeros((3, width), np.float32)
                    old[orders[states[n-1]]] = tracks[:, left:left+width]
                    weight = np.linspace(0, 1, width, dtype=np.float32)
                    dest = result[:, left-begin:left-begin+width]
                    dest[:] = old * (1-weight) + dest * weight
        return result, records
