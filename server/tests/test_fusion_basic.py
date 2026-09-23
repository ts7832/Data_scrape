from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pytest

from kuulo_protocol.api import NodeStatus
from kuulo_protocol.geo import distance_m, from_local
from kuulo_protocol.models import GeoPoint, Label, Observation, Phase, TrackStatus
from kuulo_protocol.testing import make_observation
from kuulo_server.fusion.base import NodeInfo
from kuulo_server.fusion.basic import BasicFusion
from kuulo_server.testing import T0

ORIGIN = GeoPoint(lat=60.1699, lon=24.9384)


def at(x: float, y: float) -> GeoPoint:
    return from_local(ORIGIN, x, y)


@dataclass
class Ctx:
    node_infos: dict[str, NodeInfo] = field(default_factory=dict)
    observations: list[Observation] = field(default_factory=list)

    def add_node(self, node_id, point, status=NodeStatus.ONLINE):
        self.node_infos[node_id] = NodeInfo(node_id, point, status)

    def nodes(self):
        return list(self.node_infos.values())

    def recent_observations(self, since: datetime):
        return [o for o in self.observations if o.observed_at >= since]


def observe(ctx, fusion, node_id, t: datetime, confidence=0.9, snr=12.0,
            label=Label.DRONE_MULTIROTOR, phase=Phase.START):
    p = ctx.node_infos[node_id].location
    obs = make_observation(node_id, observed_at=t, lat=p.lat, lon=p.lon, confidence=confidence,
                           snr_db=snr, label=label, phase=phase)
    ctx.observations.append(obs)
    return fusion.on_observation(obs, ctx)


def test_single_node_is_tentative_at_node():
    ctx, fusion = Ctx(), BasicFusion()
    ctx.add_node("n1", at(0, 0))
    [update] = observe(ctx, fusion, "n1", T0)
    track = update.track
    assert track.status is TrackStatus.TENTATIVE
    assert distance_m(track.position, ORIGIN) < 1
    assert track.uncertainty_m == 300
    assert track.confidence == pytest.approx(0.9)


def test_two_nodes_confirm_and_centroid_leans_to_louder():
    ctx, fusion = Ctx(), BasicFusion()
    ctx.add_node("n1", at(0, 0))
    ctx.add_node("n2", at(600, 0))
    observe(ctx, fusion, "n1", T0, snr=20)
    [update] = observe(ctx, fusion, "n2", T0 + timedelta(seconds=2), snr=6)
    x_east = distance_m(ORIGIN, GeoPoint(lat=ORIGIN.lat, lon=update.track.position.lon))
    assert update.track.status is TrackStatus.CONFIRMED
    assert x_east < 300  # pulled toward the louder n1
    assert len(fusion.tracks) == 1


def test_silent_online_neighbour_downgrades_offline_does_not():
    ctx, fusion = Ctx(), BasicFusion()
    ctx.add_node("n1", at(0, 0))
    ctx.add_node("n2", at(250, 0))                          # online, within 500 m, silent
    ctx.add_node("n3", at(0, 250), NodeStatus.OFFLINE)      # offline: ignored
    [update] = observe(ctx, fusion, "n1", T0)
    assert update.track.status is TrackStatus.DOWNGRADED
    assert update.track.silent_neighbour_ids == ["n2"]
    assert update.track.confidence == pytest.approx(0.9 * 0.7)


def test_downgraded_is_promoted_when_second_node_hears():
    ctx, fusion = Ctx(), BasicFusion()
    ctx.add_node("n1", at(0, 0))
    ctx.add_node("n2", at(250, 0))
    observe(ctx, fusion, "n1", T0)
    [update] = observe(ctx, fusion, "n2", T0 + timedelta(seconds=3))
    assert update.track.status is TrackStatus.CONFIRMED
    assert update.track.silent_neighbour_ids == []


def test_non_drone_and_end_phase_ignored():
    ctx, fusion = Ctx(), BasicFusion()
    ctx.add_node("n1", at(0, 0))
    assert observe(ctx, fusion, "n1", T0, label=Label.BIRD) == []
    assert observe(ctx, fusion, "n1", T0, phase=Phase.END) == []


def test_same_drone_updates_same_track_with_velocity():
    ctx, fusion = Ctx(), BasicFusion()
    ctx.add_node("n1", at(0, 0))
    ctx.add_node("n2", at(400, 0))
    [first] = observe(ctx, fusion, "n1", T0)
    [second] = observe(ctx, fusion, "n2", T0 + timedelta(seconds=5))
    assert second.track.track_id == first.track.track_id
    assert second.track.velocity is not None and second.track.velocity.speed_mps > 0
    assert len(second.track.observation_ids) == 2


def test_far_apart_groups_make_separate_tracks():
    ctx, fusion = Ctx(), BasicFusion()
    ctx.add_node("a", at(0, 0))
    ctx.add_node("b", at(5000, 0))
    [ta] = observe(ctx, fusion, "a", T0)
    [tb] = observe(ctx, fusion, "b", T0)
    assert ta.track.track_id != tb.track.track_id


def test_tick_closes_quiet_tracks():
    ctx, fusion = Ctx(), BasicFusion()
    ctx.add_node("n1", at(0, 0))
    observe(ctx, fusion, "n1", T0)
    assert fusion.on_tick(T0 + timedelta(seconds=29), ctx) == []
    [closed] = fusion.on_tick(T0 + timedelta(seconds=31), ctx)
    assert closed.track.status is TrackStatus.CLOSED
    assert fusion.tracks == {}
