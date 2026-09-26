# Kuulo — Impact/Detonation Localization: Roadmap Design Note

**Date:** 2026-09-26
**Status:** Idea, not scheduled — captured so it isn't lost, not yet a milestone plan
**Origin:** user request, session 2026-09-26 ("how do we handle a scenario where the drone
makes an impact? How can we map where it hit?")
**Parent spec:** `2026-09-24-kuulo-milestone1-design.md`. This slots into the **private,
Layer B fusion engine** (§2's open-core table already reserves "TDOA, Kalman/MHT, trust
scores" for `kuulo-fusion`) — it is not a change to the public Milestone 1 scope.

## 1. The idea

A hostile or malfunctioning drone can detonate (a warhead) or crash hard enough to produce a
sharp acoustic event. Beyond tracking the drone in flight, Kuulo could estimate **where that
event happened**, with a confidence region, and hand that to:

- **Emergency services** — a location to send responders, independent of any visual sighting.
- **Anti-drone / counter-UAS operators** — a cue for where a defensive intercept or search
  should be directed, and a record for after-action review.

## 2. Why this is a different problem from tracking the drone in flight

`BasicFusion` locates a *continuous* drone with a weighted centroid of detecting nodes
(`server/src/kuulo_server/fusion/basic.py`), deliberately coarse (hundreds of metres) — see
the parent spec §7. That's the right trade-off for a steady rotor hum: cross-correlating a
periodic, self-similar signal between two microphones to find its arrival-time lag is
ambiguous (the signal looks like a shifted copy of itself at multiple offsets).

An **impact** is the opposite kind of signal: a single sharp, broadband, high-amplitude
transient with a near-instant rise time (a "bang", not a hum). That is exactly the case
**TDOA multilateration** (Time Difference of Arrival) is built for — the same principle GPS
uses, run backwards. Sound travels at a known, fixed ~343 m/s; if several microphones at known
positions each timestamp the same instant the bang's wavefront reaches them, the *differences*
in arrival time between sensor pairs each constrain the source to one hyperbola, and three or
more hyperbolas from non-collinear sensors intersect at a point. No drone-specific modelling is
needed — it's geometry and one well-defined signal-processing step (onset detection via cross-
correlation, reliable specifically because the signal is impulsive).

This is the reason the parent spec already put TDOA in the *private* fusion engine's column,
not the public `BasicFusion`: it needs tighter time synchronization and a materially different
algorithm than the coarse weighted centroid, and is a natural, concrete first use of that
reserved slot.

## 3. Sketch of the pieces

### 3.1 Node: an impulse detector, separate from the drone-hum classifier

The existing `AcousticMeter` (`node/src/kuulo_node/acoustic.py`) computes a rough SNR and peak
frequency per 0.975 s window — far too coarse in time to time an impact to TDOA-useful
precision. A new, lightweight detector would run alongside it:

- Watch for a sudden, broadband jump in energy across most frequency bands **simultaneously**,
  within a few milliseconds — unlike a rotor's narrowband, sustained harmonics.
- On trigger, find the precise onset sample (not just the window it fell in) by thresholding
  the raw sample stream directly, so timing precision approaches the sample rate (16 kHz ⇒
  62.5 µs per sample), not the 20 ms frame period the existing FeatureTrace format uses.
- Timestamp that onset using the node's existing clock discipline (`time_quality`: `gps`,
  `ntp`, or `manual` — already a field on every message). GPS-timed nodes are what make
  sub-metre TDOA solutions possible; NTP-timed nodes still contribute, just with a wider
  uncertainty band (see §3.3).
- Still respects the privacy design: this stays an energy-transient detector over the same
  kind of coarse spectral representation as FeatureTraces, never raw intelligible audio.

### 3.2 A new wire message: `ImpactEvent`

Alongside `Observation`, `Heartbeat`, `Track` and `FeatureTraceHeader` in
`protocol/src/kuulo_protocol/`:

- `node_id`, `onset_at` (sample-precise timestamp), `peak_amplitude_db`, `time_quality`,
  signed like every other message.
- The server correlates `ImpactEvent`s from multiple nodes within a short window (sound's
  travel time across the sensor network, a few seconds at most) into one candidate impact.

### 3.3 Server: TDOA solver and an uncertainty *ellipse*, not a circle

- Solve the multilateration system (e.g. a Gauss-Newton or Chan-Ho closed-form solve over the
  pairwise time differences) for the impact's (lat, lon) — this is the private-engine piece the
  parent spec already anticipated.
- The uncertainty region this produces is naturally an **ellipse**: its size and orientation
  depend on the sensor array's geometry relative to the source (GDOP — geometric dilution of
  precision, the same concept a GPS receiver reports) and on each contributing node's timing
  precision (from `time_quality`). A tight cluster of GPS-synced nodes around the blast gives a
  small, well-shaped ellipse; nodes bunched on one side give a long, smeared one. This is more
  honest than `BasicFusion`'s single-radius circle and should be modelled as one
  (semi-major axis, semi-minor axis, orientation) on any `ImpactSite` record, not reduced to a
  radius for storage.
- **Corroboration with the flight track:** extrapolate the last CONFIRMED track's position
  using its existing `velocity` field (speed + heading, already computed by `BasicFusion`) to
  the impact's estimated time. An impact near that extrapolated point is strong evidence it's
  the same drone, not an unrelated bang (fireworks, a gunshot, a car backfiring); this becomes
  part of the impact record's evidence, the same way `Track.silent_neighbour_ids` is evidence
  today.

### 3.4 Dashboard and downstream relay

- A new map layer: an `ImpactSite` marker with its uncertainty ellipse (MapLibre supports
  arbitrary polygons, so this is an extension of the existing `uncertaintyToGeoJSON` pattern in
  `dashboard/src/map/layers.ts`, generalized from a circle to an ellipse), plus a link back to
  the track it was correlated with.
- Downstream relay to emergency services or counter-UAS operators is an integration question
  (a webhook, an API poll, a dedicated read-only feed) rather than a dashboard concern, and
  should stay out of the public repo per the open-core boundary (parent spec §2: "the live
  coverage map is never public").

## 4. Open questions for whoever picks this up

- How many GPS-timed nodes, at what density, are actually needed for a useful (sub-50 m)
  ellipse? Worth prototyping in `sim/` with synthetic TDOA before any node-side work, the same
  way `BasicFusion` was validated against `helsinki_pass`'s measured location error.
- False-positive rejection: what else produces a sharp broadband transient near a sensor
  network (thunder, a car crash, construction)? Likely needs its own hard-negative evaluation,
  mirroring `ml/RESULTS.md`'s approach for the drone-hum classifier.
- Whether onset detection is reliable enough over a consumer laptop mic (this repo's demo
  hardware) or genuinely needs purpose-built nodes with a real ADC and GPS discipline — the
  parent spec's Milestone 1 non-goals already exclude Raspberry Pi/hardware deployment, and
  this feature likely deepens that dependency rather than removing it.
