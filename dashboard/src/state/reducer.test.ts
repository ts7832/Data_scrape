import { describe, expect, it } from "vitest";
import type { NodeView, Track } from "../api/types";
import { initialState, reducer, TRAIL_MAX, PULSE_MS } from "./reducer";

const node = (id: string, status: NodeView["status"] = "online"): NodeView => ({
  node_id: id,
  location: { lat: 60.17, lon: 24.94, accuracy_m: 10 },
  time_quality: "ntp",
  status,
});

const track = (id: string, lat = 60.17, status: Track["status"] = "tentative"): Track => ({
  track_id: id,
  status,
  label: "drone_multirotor",
  confidence: 0.8,
  position: { lat, lon: 24.94 },
  uncertainty_m: 300,
  first_seen: "2026-09-24T12:00:00Z",
  last_seen: "2026-09-24T12:00:05Z",
  observation_ids: [],
  silent_neighbour_ids: [],
});

describe("reducer", () => {
  it("snapshot replaces everything", () => {
    let s = reducer(initialState, { type: "live", event: { type: "track", data: track("old") }, receivedAt: 0 });
    s = reducer(s, { type: "snapshot", nodes: [node("n1")], tracks: [track("t1")] });
    expect(Object.keys(s.tracks)).toEqual(["t1"]);
    expect(Object.keys(s.nodes)).toEqual(["n1"]);
    expect(s.trails["old"]).toBeUndefined();
  });

  it("live track upserts and builds a capped trail", () => {
    let s = initialState;
    for (let i = 0; i < TRAIL_MAX + 10; i++) {
      s = reducer(s, { type: "live", event: { type: "track", data: track("t1", 60 + i / 1000) }, receivedAt: 0 });
    }
    expect(s.tracks["t1"].position.lat).toBeCloseTo(60 + (TRAIL_MAX + 9) / 1000);
    expect(s.trails["t1"].length).toBe(TRAIL_MAX);
  });

  it("closed track is removed and deselected", () => {
    let s = reducer(initialState, { type: "live", event: { type: "track", data: track("t1") }, receivedAt: 0 });
    s = reducer(s, { type: "select", trackId: "t1" });
    s = reducer(s, { type: "live", event: { type: "track", data: track("t1", 60.17, "closed") }, receivedAt: 0 });
    expect(s.tracks["t1"]).toBeUndefined();
    expect(s.selectedTrackId).toBeNull();
  });

  it("node_status upserts; observation adds a pulse that expires", () => {
    let s = reducer(initialState, { type: "live", event: { type: "node_status", data: node("n1", "stale") }, receivedAt: 0 });
    expect(s.nodes["n1"].status).toBe("stale");
    const obs = { source: { type: "simulated_node", id: "n1" } } as never;
    s = reducer(s, { type: "live", event: { type: "observation", data: obs }, receivedAt: 1000 });
    expect(s.pulses).toEqual([{ nodeId: "n1", at: 1000 }]);
    s = reducer(s, { type: "expirePulses", now: 1000 + PULSE_MS + 1 });
    expect(s.pulses).toEqual([]);
  });

  it("snapshot drops closed tracks (review finding: they stayed on the map)", () => {
    const s = reducer(initialState, {
      type: "snapshot", nodes: [], tracks: [track("t1", 60.17, "closed"), track("t2")],
    });
    expect(Object.keys(s.tracks)).toEqual(["t2"]);
    expect(s.trails["t1"]).toBeUndefined();
  });

  it("resync event does not change state", () => {
    const s = reducer(initialState, { type: "live", event: { type: "resync" }, receivedAt: 0 });
    expect(s).toBe(initialState);
  });
});
