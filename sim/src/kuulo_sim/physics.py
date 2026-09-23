"""Deliberately simple acoustic physics: spherical spreading + air absorption vs a noise floor."""

from __future__ import annotations

import random
from math import exp, log10

from kuulo_protocol.models import TimeQuality

SPEED_OF_SOUND_MPS = 343.0
ABSORPTION_DB_PER_M = 0.005
CLOCK_SIGMA_S = {TimeQuality.GPS: 1e-6, TimeQuality.NTP: 0.02, TimeQuality.MANUAL: 0.5}


def received_level_db(source_db: float, distance_m: float) -> float:
    r = max(distance_m, 1.0)
    return source_db - 20 * log10(r) - ABSORPTION_DB_PER_M * r


def detection_probability(snr_db: float, snr50: float = 6.0, width: float = 2.0) -> float:
    return 1.0 / (1.0 + exp(-(snr_db - snr50) / width))


def confidence_from_snr(snr_db: float, rng: random.Random) -> float:
    return min(1.0, max(0.0, 0.5 + 0.03 * snr_db + rng.gauss(0, 0.05)))


def propagation_delay_s(distance_m: float) -> float:
    return distance_m / SPEED_OF_SOUND_MPS


def clock_offset_s(time_quality: TimeQuality, rng: random.Random) -> float:
    return rng.gauss(0, CLOCK_SIGMA_S[time_quality])
