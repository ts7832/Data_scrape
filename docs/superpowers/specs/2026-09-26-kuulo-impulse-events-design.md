# Kuulo — Impulse Events: Design Spec (public part)

**Date:** 2026-09-26
**Status:** Approved in conversation, awaiting the user's review of the plan
**Parent spec:** `2026-09-24-kuulo-milestone1-design.md`
**Plan:** `../plans/2026-09-26-kuulo-impulse-events.md`

## 1. Purpose

A drone can detonate or crash hard enough to make a sharp acoustic event. Kuulo nodes will report
such events, and the server will group them, place them on the map and export them. That gives
emergency services and counter-UAS operators a location to act on, as a Common Alerting Protocol
(CAP 1.2) alert as well as through the API and the dashboard.

This document covers the **public** part: the wire format, the node-side detector, server ingest,
the pluggable locator interface with a coarse public baseline, CAP export, the simulator and the
dashboard. Precise localization and attribution to drone tracks are provided by the private
fusion engine through the locator plug-in (§5). That is the same open-core split as
`BasicFusion` (public) versus the private track-fusion engine in the parent spec §2.

Terminology:

- **Impulse**: any short, sharp, broadband sound a node hears (a blast, a crash, a firework, a
  slammed door). Nodes report impulses and never claim to know what made them.
- **Impulse event**: the server's fusion of one physical impulse heard by one or more nodes.
- **Drone impact**: an impulse event that a locator has attributed to a drone track. The
  dashboard labels it IMPACT (red); every other event is labelled IMPULSE (orange).

## 2. Node: impulse detection (no audio leaves the node)

A detector runs beside the drone classifier on the same 16 kHz audio:

1. **Trigger (STA/LTA, from seismology).** A 2 ms short-term energy average is compared with a
   ≈5 s long-term average (frozen during a detection), after a 100 Hz high-pass that removes wind
   rumble. A jump of 20 dB or more fires the trigger.
2. **Onset picking (AIC, from seismology).** The Akaike Information Criterion picker finds the
   sample where the statistics change from noise to event.
3. **Checks.** The detector requires a minimum peak level, a fast rise (≤ 50 ms), a decay of at
   least 10 dB within 1 s (sustained noise that starts abruptly is not an impulse), and energy
   spread across at least 6 of the 32 bands (a tone switching on is rejected). A 2 s dead time
   stops echoes of the same bang being reported again. After a rejection the long-term average
   is re-based, so a sustained sound cannot keep re-triggering.
4. **Timestamp.** The onset sample is converted to UTC from the audio driver's capture-time
   stamp of each block (PortAudio `inputBufferAdcTime`), never from processing time: queue
   delays and sound-card clock drift would otherwise add error.
5. **Report.** A signed `ImpulseReport` carries the onset time (microsecond precision), the
   node's own onset uncertainty, its `time_quality`, and summary features only: peak level, SNR,
   rise time, duration, a clipping flag, and the 32-band spectrum of the 20 ms around the peak.
   **No waveform or envelope is sent**, so the parent spec's privacy guarantee is unchanged.

Impulse reports go through the durable outbox. When it is full, they are dropped last, after
heartbeats and drone observations.

## 3. Server: ingest and grouping

- `POST /v1/impulses`: signature and node checks as for observations; idempotent on
  `report_id`; rejected if more than 30 s in the future. **Late reports are accepted and used**,
  because an attack is exactly when networks fail.
- **Grouping by physics.** Two reports can come from the same impulse only if their onset times
  differ by no more than the sound's travel time between the two nodes, plus timing slack. A new
  report joins the consistent reports near it in time, at most one per node. The same event is
  updated as reports arrive, however late.

## 4. Public baseline locator

`CoarseLocator` places the event at the centroid of the reporting nodes, with a circle of
1.5 km plus the nodes' spread (`quality = coarse`), and never attributes it to a track. It says
honestly "these sensors heard the same bang, somewhere around here". It keeps the public system
complete and testable without the private engine.

## 5. Locator plug-in

The locator is loaded from a `"module:Class"` setting, `impulse_locator`. It can be overridden with
the `KUULO_IMPULSE_LOCATOR` environment variable, and `KUULO_FUSION_ENGINE` overrides the fusion
engine the same way. A locator receives a `LocatorConfig` (air temperature, timing floor, radii,
association limits) and implements
`on_report(report, ctx) -> ImpulseEvent | None`. The context gives read access to reports
near a time, to existing event membership, and to recently seen tracks. The wire model
`ImpulseEvent` already carries what a precise locator produces (an uncertainty ellipse,
alternative solutions, per-node residuals, excluded nodes, the associated track), so the dashboard
and CAP export show it without changes. Locator exceptions are caught and logged; ingest continues.

## 6. Outputs

- `GET /v1/impulse-events`, `GET /v1/impulse-events/{id}` (the event plus its reports), a
  `impulse_event` WebSocket event.
- `GET /v1/impulse-events/{id}/cap`: CAP 1.2 XML with category Security and the region as a
  polygon. Status defaults to **Test** and is configurable, so demo output cannot be mistaken
  for a real alert.
- Dashboard: impulse markers with their region (an ellipse or circle), plus any alternative
  points; a list; a detail panel; event-log entries.

## 7. Also in this work

- **BasicFusion speed fix.** Track speed is currently computed between observations milliseconds
  apart from different nodes. The demo showed 1295 and 2647 m/s for a 20 m/s drone. Speed will
  be computed over ≥ 3 s.
- **Simulator.** Standalone blasts, drones that detonate at the end of their route, and
  per-node extra delay for blocked sound paths. Propagation delays must not be compressed by
  `--speed`. New scenarios `impact_strike` and `firework`.
- **Real-audio evaluation of the detector** on licensed ESC-50 impulsive clips mixed into real
  backgrounds, and false triggers on continuous sounds and drone recordings. Fireworks are a
  proxy for blasts, and that is stated wherever results appear.

## 8. Out of scope

Classifying what made an impulse; 3-D; military interfaces; authentication (localhost only, as in
the parent spec); real multi-node hardware trials.
