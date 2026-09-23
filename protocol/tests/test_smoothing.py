from uuid import UUID

from kuulo_protocol.models import Phase
from kuulo_protocol.smoothing import DetectionSmoother, SmootherConfig


def feed(smoother, scores, start=0.0, dt=1.0):
    return [smoother.push(start + i * dt, s) for i, s in enumerate(scores)]


def test_isolated_spikes_never_start():
    s = DetectionSmoother()
    assert feed(s, [0.9, 0.1, 0.1, 0.9, 0.1, 0.1, 0.9, 0.1]) == [None] * 8
    assert not s.active


def test_three_of_five_starts_then_updates_then_ends():
    ids = iter([UUID(int=1), UUID(int=2)])
    s = DetectionSmoother(SmootherConfig(), id_factory=lambda: next(ids))
    phases = feed(s, [0.9, 0.9, 0.9])
    assert phases == [None, None, Phase.START]
    assert s.detection_id == UUID(int=1)
    phases = feed(s, [0.9] * 5, start=3.0)          # t = 3..7; START was at t=2
    assert phases == [None, None, None, None, Phase.UPDATE]


def test_update_timing_and_end():
    s = DetectionSmoother()
    out = feed(s, [0.9] * 8)                         # t = 0..7, start at t=2
    assert out[2] is Phase.START and out[7] is Phase.UPDATE
    assert [p for p in out if p is Phase.UPDATE] == [Phase.UPDATE]
    tail = feed(s, [0.0] * 6, start=8.0)             # last above at t=7 → end at t=12
    assert tail == [None, None, None, None, Phase.END, None]
    assert not s.active
    assert s.detection_id is not None                # kept for the END message


def test_new_detection_gets_new_id():
    s = DetectionSmoother()
    feed(s, [0.9] * 3)
    first = s.detection_id
    feed(s, [0.0] * 6, start=3.0)
    feed(s, [0.9] * 3, start=9.0)
    assert s.active and s.detection_id != first
