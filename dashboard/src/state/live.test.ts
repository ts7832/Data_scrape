import { describe, expect, it } from "vitest";
import type { Action } from "./reducer";
import { backoffDelay, startLive, type SocketLike } from "./live";

class FakeSocket implements SocketLike {
  onopen: (() => void) | null = null;
  onmessage: ((e: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  closed = false;
  close() { this.closed = true; }
}

function harness() {
  const sockets: FakeSocket[] = [];
  const actions: Action[] = [];
  const timers: { fn: () => void; ms: number }[] = [];
  let resolveSnapshot: (v: { nodes: never[]; tracks: never[] }) => void = () => {};
  let snapshots = 0;
  const stop = startLive({
    url: "ws://x/v1/live",
    openSocket: () => { const s = new FakeSocket(); sockets.push(s); return s; },
    fetchSnapshot: () => { snapshots++; return new Promise((r) => { resolveSnapshot = r; }); },
    dispatch: (a) => actions.push(a),
    schedule: (fn, ms) => { timers.push({ fn, ms }); },
    now: () => 42,
  });
  return { sockets, actions, timers, stop, resolve: () => resolveSnapshot({ nodes: [], tracks: [] }), snapshots: () => snapshots };
}

const flush = () => new Promise((r) => setTimeout(r, 0));

describe("live connection", () => {
  it("backoff doubles and caps", () => {
    expect([0, 1, 2, 3, 4, 10].map(backoffDelay)).toEqual([1000, 2000, 4000, 8000, 15000, 15000]);
  });

  it("buffers events that arrive while the snapshot loads, then replays them", async () => {
    // Review Focus 5.
    const h = harness();
    h.sockets[0].onopen!();
    h.sockets[0].onmessage!({ data: JSON.stringify({ type: "node_status", data: { node_id: "n1" } }) });
    expect(h.actions.some((a) => a.type === "live")).toBe(false);
    h.resolve();
    await flush();
    const types = h.actions.map((a) => a.type);
    expect(types.indexOf("snapshot")).toBeLessThan(types.indexOf("live"));
    expect(h.actions.at(-1)).toEqual({ type: "connection", connection: "live" });
  });

  it("reconnects with backoff after close", async () => {
    const h = harness();
    h.sockets[0].onclose!();
    expect(h.actions.at(-1)).toEqual({ type: "connection", connection: "reconnecting" });
    expect(h.timers[0].ms).toBe(1000);
    h.timers[0].fn();
    expect(h.sockets.length).toBe(2);
    h.sockets[1].onclose!();
    expect(h.timers[1].ms).toBe(2000);
  });

  it("resync event reloads the snapshot", async () => {
    const h = harness();
    h.sockets[0].onopen!();
    h.resolve();
    await flush();
    h.sockets[0].onmessage!({ data: JSON.stringify({ type: "resync" }) });
    expect(h.snapshots()).toBe(2);
  });

  it("stop closes the socket and prevents reconnect", () => {
    const h = harness();
    h.stop();
    expect(h.sockets[0].closed).toBe(true);
    h.sockets[0].onclose?.();
    expect(h.timers.length).toBe(0);
  });
});
