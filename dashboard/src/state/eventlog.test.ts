import { describe, expect, it } from "vitest";
import type { NodeView, Observation, Track } from "../api/types";
import { EVENT_LOG_MAX, initialState, reducer, type State } from "./reducer";

const track = (id: string, status: Track["status"]): Track => ({
  track_id: id,
  status,
  label: "drone_multirotor",
  confidence: 0.8,
  position: { lat: 60.17, lon: 24.94 },
  uncertainty_m: 300,
  first_seen: "2026-09-24T12:00:00Z",
  last_seen: "2026-09-24T12:00:05Z",
  observation_ids: [],
  silent_neighbour_ids: [],
});

const node = (id: string, status: NodeView["status"]): NodeView => ({
  node_id: id,
  location: { lat: 60.17, lon: 24.94, accuracy_m: 10 },
  time_quality: "ntp",
  status,
});

const observation = (nodeId: string, phase: Observation["event"]["phase"] = "start") =>
  ({
    source: { type: "simulated_node", id: nodeId },
    detection: { label: "drone_multirotor", confidence: 0.91 },
    event: { detection_id: "d", phase },
  }) as unknown as Observation;

const live = (s: State, type: "track" | "observation" | "node_status", data: unknown, at = 0) =>
  reducer(s, { type: "live", event: { type, data } as never, receivedAt: at });

describe("event log", () => {
  it("logs detections newest first", () => {
    let s = live(initialState, "observation", observation("n03"), 1);
    s = live(s, "observation", observation("n08"), 2);
    expect(s.events.map((e) => e.source)).toEqual(["n08", "n03"]);
    expect(s.events[0].message).toContain("START");
    expect(s.events[0].at).toBe(2);
  });

  it("logs a track only when its status changes", () => {
    let s = live(initialState, "track", track("t1", "tentative"));
    s = live(s, "track", track("t1", "tentative"));
    s = live(s, "track", track("t1", "confirmed"));
    s = live(s, "track", track("t1", "closed"));
    expect(s.events.map((e) => e.message)).toEqual(["CLOSED", "CONFIRMED", "TENTATIVE"]);
    expect(s.events[1].tone).toBe("danger");
  });

  it("logs node status changes but not repeats", () => {
    let s = live(initialState, "node_status", node("n1", "online"));
    s = live(s, "node_status", node("n1", "online"));
    s = live(s, "node_status", node("n1", "offline"));
    expect(s.events.map((e) => e.message)).toEqual(["OFFLINE", "ONLINE"]);
  });

  it(`never holds more than ${EVENT_LOG_MAX} entries`, () => {
    let s = initialState;
    for (let i = 0; i < EVENT_LOG_MAX + 50; i++) s = live(s, "observation", observation("n1"), i);
    expect(s.events.length).toBe(EVENT_LOG_MAX);
    expect(s.events[0].at).toBe(EVENT_LOG_MAX + 49);
  });

  it("gives every entry a unique key", () => {
    let s = initialState;
    for (let i = 0; i < 5; i++) s = live(s, "observation", observation("n1"), 0);
    expect(new Set(s.events.map((e) => e.seq)).size).toBe(5);
  });
});
