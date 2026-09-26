"""CAP 1.2 (OASIS Common Alerting Protocol) rendering of an impulse event, for alerting systems."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from kuulo_protocol.geo import ellipse_ring
from kuulo_protocol.impulses import ImpulseEvent, ImpulseKind, LocationQuality

NS = "urn:oasis:names:tc:emergency:cap:1.2"
STATUSES = ("Actual", "Exercise", "System", "Test", "Draft")


def _sent(dt) -> str:
    return dt.replace(microsecond=0).isoformat()  # "+00:00": CAP forbids "Z"


def to_cap_xml(event: ImpulseEvent, *, sender: str, status: str,
               track_callsign: str | None = None) -> bytes:
    if status not in STATUSES:
        raise ValueError(f"CAP status must be one of {STATUSES}, got {status!r}")
    ET.register_namespace("", NS)
    q = lambda tag: f"{{{NS}}}{tag}"  # noqa: E731
    alert = ET.Element(q("alert"))

    def add(parent, tag, value):
        el = ET.SubElement(parent, q(tag))
        el.text = value
        return el

    impact = event.kind is ImpulseKind.DRONE_IMPACT
    add(alert, "identifier", f"kuulo-{event.event_id}")
    add(alert, "sender", sender)
    add(alert, "sent", _sent(event.updated_at))
    add(alert, "status", status)
    add(alert, "msgType", "Alert")
    add(alert, "scope", "Restricted")
    add(alert, "restriction", "Emergency services and authorised counter-UAS operators")
    info = ET.SubElement(alert, q("info"))
    add(info, "category", "Security")
    add(info, "event", "Drone impact" if impact else "Unexplained acoustic impulse")
    add(info, "urgency", "Immediate")
    add(info, "severity", "Severe" if impact else "Moderate")
    located = event.quality is LocationQuality.MULTILATERATED
    add(info, "certainty", "Likely" if impact and located else "Possible")
    add(info, "senderName", "Kuulo acoustic sensor network")
    e = event.ellipse
    add(info, "headline", f"{'Drone impact' if impact else 'Acoustic impulse'} located "
                          f"to ±{e.semi_major_m:.0f} m by {len(event.node_ids)} sensors")
    track_ref = track_callsign or event.associated_track_id
    track = f" Associated drone track: {track_ref}." if impact else ""
    add(info, "description",
        f"Location quality: {event.quality.value}. {int(e.confidence * 100)} % region: ellipse "
        f"{e.semi_major_m:.0f} m x {e.semi_minor_m:.0f} m, major axis bearing "
        f"{e.bearing_deg:.0f} deg. Estimated time {event.occurred_at.isoformat()}.{track}")
    area = ET.SubElement(info, q("area"))
    add(area, "areaDesc", f"{int(e.confidence * 100)} % confidence region of the impulse")
    ring = ellipse_ring(event.position, e.semi_major_m, e.semi_minor_m, e.bearing_deg, steps=32)
    add(area, "polygon", " ".join(f"{p.lat:.6f},{p.lon:.6f}" for p in ring))
    return ET.tostring(alert, encoding="utf-8", xml_declaration=True)
