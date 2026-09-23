import type { LiveEvent, NodeView, Track } from "../api/types";
import type { Action } from "./reducer";

export interface SocketLike {
  onopen: (() => void) | null;
  onmessage: ((e: { data: string }) => void) | null;
  onclose: (() => void) | null;
  close(): void;
}

export interface LiveDeps {
  url: string;
  openSocket(url: string): SocketLike;
  fetchSnapshot(): Promise<{ nodes: NodeView[]; tracks: Track[] }>;
  dispatch(action: Action): void;
  schedule(fn: () => void, ms: number): void;
  now(): number;
}

export function backoffDelay(attempt: number): number {
  return Math.min(1000 * 2 ** attempt, 15000);
}

export function defaultLiveUrl(): string {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${window.location.host}/v1/live`;
}

/** Connects, loads a snapshot, applies live events; reconnects with backoff. Returns stop(). */
export function startLive(deps: LiveDeps): () => void {
  let attempt = 0;
  let stopped = false;
  let socket: SocketLike | null = null;

  const connect = () => {
    if (stopped) return;
    let loading = true;
    let buffer: LiveEvent[] = [];
    const ws = deps.openSocket(deps.url);
    socket = ws;

    const loadSnapshot = async () => {
      loading = true;
      try {
        const snap = await deps.fetchSnapshot();
        if (stopped || socket !== ws) return;
        deps.dispatch({ type: "snapshot", nodes: snap.nodes, tracks: snap.tracks });
        for (const event of buffer) deps.dispatch({ type: "live", event, receivedAt: deps.now() });
        buffer = [];
        loading = false;
        deps.dispatch({ type: "connection", connection: "live" });
      } catch {
        ws.close();
      }
    };

    ws.onopen = () => {
      attempt = 0;
      void loadSnapshot();
    };
    ws.onmessage = (e) => {
      const event = JSON.parse(e.data) as LiveEvent;
      if (event.type === "resync") {
        buffer = [];
        void loadSnapshot();
      } else if (loading) {
        buffer.push(event);
      } else {
        deps.dispatch({ type: "live", event, receivedAt: deps.now() });
      }
    };
    ws.onclose = () => {
      if (stopped) return;
      deps.dispatch({ type: "connection", connection: "reconnecting" });
      deps.schedule(connect, backoffDelay(attempt++));
    };
  };

  connect();
  return () => {
    stopped = true;
    socket?.close();
  };
}
