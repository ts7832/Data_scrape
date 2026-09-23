from datetime import UTC, datetime, timedelta, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from kuulo_protocol.models import (
    DRONE_LABELS,
    Label,
    Observation,
    Phase,
    to_utc_ms,
)
from kuulo_protocol.testing import make_heartbeat, make_observation

T0 = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)


def test_observation_json_round_trip():
    obs = make_observation(observed_at=T0)
    again = Observation.model_validate_json(obs.model_dump_json())
    assert again == obs


def test_naive_timestamp_rejected():
    with pytest.raises(ValidationError, match="timezone"):
        make_observation(observed_at=datetime(2026, 9, 24, 12, 0))


def test_timestamp_normalised_to_utc_milliseconds():
    helsinki = timezone(timedelta(hours=3))
    ts = datetime(2026, 9, 24, 14, 0, 0, 123456, tzinfo=helsinki)
    assert to_utc_ms(ts) == datetime(2026, 9, 24, 11, 0, 0, 123000, tzinfo=UTC)
    assert make_observation(observed_at=ts).observed_at.microsecond == 123000


def test_extreme_snr_db_rejected():
    # Review finding: an unbounded snr_db lets one observation (huge but finite,
    # e.g. 1e4) cause an OverflowError deep inside fusion's 10 ** (snr / 20)
    # weighting, silently disabling fusion for every observation grouped with it.
    with pytest.raises(ValidationError):
        make_observation(observed_at=T0, snr_db=1e4)


def test_infinite_or_nan_snr_db_rejected():
    # Review finding: a raw `Infinity`/`NaN` JSON literal is accepted by FastAPI's
    # parser, stored, then fails to reload inside recent_observations — breaking
    # fusion network-wide, not just for nearby nodes.
    with pytest.raises(ValidationError):
        make_observation(observed_at=T0, snr_db=float("inf"))
    with pytest.raises(ValidationError):
        make_observation(observed_at=T0, snr_db=float("nan"))


def test_confidence_out_of_range_rejected():
    with pytest.raises(ValidationError):
        make_observation(observed_at=T0, confidence=1.5)


def test_unknown_field_rejected():
    data = make_observation(observed_at=T0).model_dump(mode="json")
    data["surprise"] = 1
    with pytest.raises(ValidationError):
        Observation.model_validate(data)


def test_factory_overrides():
    det = uuid4()
    obs = make_observation(
        "n7", observed_at=T0, label=Label.BIRD, phase=Phase.END, detection_id=det, snr_db=None
    )
    assert obs.source.id == "n7"
    assert obs.detection.label is Label.BIRD
    assert obs.event.phase is Phase.END
    assert obs.event.detection_id == det
    assert obs.acoustic is None


def test_drone_labels():
    assert Label.DRONE_MULTIROTOR in DRONE_LABELS
    assert Label.BIRD not in DRONE_LABELS


def test_heartbeat_factory():
    hb = make_heartbeat("n1", sent_at=T0)
    assert hb.node_id == "n1" and hb.mic_ok and hb.queue_depth == 0
