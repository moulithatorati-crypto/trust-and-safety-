"""
Behavioral drift scoring: a z-score-like float measuring how far a
session's behavior deviates from THAT account's own rolling history --
never a global population comparison. An account's first-ever session is
by definition not anomalous (no baseline yet to deviate from).
"""
from __future__ import annotations

import math
import threading
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class _Baseline:
    n: int = 0
    mean_duration: float = 0.0
    mean_cadence: float = 0.0
    m2_duration: float = 0.0  # Welford running variance accumulator
    m2_cadence: float = 0.0
    copy_paste_events: int = 0


class BehavioralDriftTracker:
    def __init__(self):
        self._lock = threading.RLock()
        self._baselines: dict[str, _Baseline] = defaultdict(_Baseline)

    def _update_welford(self, b: _Baseline, duration: float, cadence: float):
        b.n += 1
        d1 = duration - b.mean_duration
        b.mean_duration += d1 / b.n
        d2 = duration - b.mean_duration
        b.m2_duration += d1 * d2

        c1 = cadence - b.mean_cadence
        b.mean_cadence += c1 / b.n
        c2 = cadence - b.mean_cadence
        b.m2_cadence += c1 * c2

    def drift_score(self, account_id: str, session: dict) -> float:
        with self._lock:
            b = self._baselines[account_id]
            duration = float(session.get("duration_seconds", 0.0))
            cadence = float(session.get("typing_cadence_ms", 0.0))
            copy_paste = bool(session.get("copy_paste_detected", False))

            if b.n < 2:
                # Not enough history to have a meaningful baseline yet --
                # first session is never flagged as anomalous.
                score = 0.0
            else:
                var_duration = b.m2_duration / max(b.n - 1, 1)
                var_cadence = b.m2_cadence / max(b.n - 1, 1)
                std_duration = math.sqrt(var_duration) or 1.0
                std_cadence = math.sqrt(var_cadence) or 1.0

                z_duration = abs(duration - b.mean_duration) / std_duration
                z_cadence = abs(cadence - b.mean_cadence) / std_cadence

                combined_z = (z_duration + z_cadence) / 2.0
                # squash into 0..1 with a smooth cap
                score = 1.0 - math.exp(-combined_z / 3.0)

                if copy_paste and b.copy_paste_events == 0:
                    # first time this account has ever used copy/paste on
                    # checkout fields -- mild additional signal
                    score = min(1.0, score + 0.15)

            self._update_welford(b, duration, cadence)
            if copy_paste:
                b.copy_paste_events += 1

            return round(max(0.0, min(1.0, score)), 4)


_tracker = BehavioralDriftTracker()


def drift_score(account_id: str, session: dict) -> float:
    return _tracker.drift_score(account_id, session)
