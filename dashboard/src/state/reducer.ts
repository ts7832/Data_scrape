import type { LiveEvent, NodeView, Observation, Track } from "../api/types";
import { percent, shortTrackId } from "../format";

export type Connection = "connecting" | "live" | "reconnecting";
export interface Pulse { nodeId: string; at: number }

export type Tone = "none" | "primary" | "success" | "warning" | "danger";
export interface LogEntry {
  seq: number;     // unique, increasing; used as the React key
  at: number;      // ms timestamp when the dashboard received it
  source: string;  // node id or track callsign
  message: string;
  tone: Tone;
}

export interface State {
  nodes: Record<string, NodeView>;
  tracks: Record<string, Track>;
  trails: Record<string, [number, number][]>; // [lon, lat]
  pulses: Pulse[];
  connection: Connection;
  selectedTrackId: string | null;
  events: LogEntry[]; // newest first, never longer than EVENT_LOG_MAX
  eventSeq: number;
}

export type Action =
  | { type: "snapshot"; nodes: NodeView[]; tracks: Track[] }
  | { type: "live"; event: LiveEvent; receivedAt: number }
  | { type: "connection"; connection: Connection }
  | { type: "select"; trackId: string | null }
  | { type: "expirePulses"; now: number };

export const TRAIL_MAX = 60;
export const PULSE_MS = 3000;
// Hard cap so a dashboard left open for days holds a bounded amount of history.
export const EVENT_LOG_MAX = 200;

export const initialState: State = {
  nodes: {}, tracks: {}, trails: {}, pulses: [], connection: "connecting", selectedTrackId: null,
  events: [], eventSeq: 0,
};

const TRACK_TONE: Record<Track["status"], Tone> = {
  confirmed: "danger", tentative: "warning", downgraded: "none", closed: "none",
};
const NODE_TONE: Record<NodeView["status"], Tone> = {
  online: "success", stale: "warning", offline: "danger",
};

function log(state: State, at: number, source: string, message: string, tone: Tone): State {
  const entry: LogEntry = { seq: state.eventSeq + 1, at, source, message, tone };
  return {
    ...state,
    eventSeq: entry.seq,
    events: [entry, ...state.events].slice(0, EVENT_LOG_MAX),
  };
}

function upsertTrack(state: State, track: Track): State {
  const tracks = { ...state.tracks };
  const trails = { ...state.trails };
  if (track.status === "closed") {
    delete tracks[track.track_id];
    delete trails[track.track_id];
    const selectedTrackId = state.selectedTrackId === track.track_id ? null : state.selectedTrackId;
    return { ...state, tracks, trails, selectedTrackId };
  }
  tracks[track.track_id] = track;
  const point: [number, number] = [track.position.lon, track.position.lat];
  const trail = trails[track.track_id] ?? [];
  const last = trail[trail.length - 1];
  const next = last && last[0] === point[0] && last[1] === point[1] ? trail : [...trail, point];
  trails[track.track_id] = next.slice(-TRAIL_MAX);
  return { ...state, tracks, trails };
}

export function reducer(state: State, action: Action): State {
  switch (action.type) {
    case "snapshot": {
      // The server's GET /v1/tracks intentionally still lists a track for a short
      // while after it closes (evidence for other API consumers); the map must not.
      const liveTracks = action.tracks.filter((t) => t.status !== "closed");
      const nodes = Object.fromEntries(action.nodes.map((n) => [n.node_id, n]));
      const tracks = Object.fromEntries(liveTracks.map((t) => [t.track_id, t]));
      const trails = Object.fromEntries(
        Object.entries(state.trails).filter(([id]) => id in tracks),
      );
      for (const t of liveTracks) {
        if (!trails[t.track_id]) trails[t.track_id] = [[t.position.lon, t.position.lat]];
      }
      const selectedTrackId = state.selectedTrackId && state.selectedTrackId in tracks ? state.selectedTrackId : null;
      return { ...state, nodes, tracks, trails, selectedTrackId };
    }
    case "live": {
      const { event } = action;
      switch (event.type) {
        case "track": {
          const t = event.data as Track;
          const previous = state.tracks[t.track_id]?.status;
          const next = upsertTrack(state, t);
          if (previous === t.status) return next;
          return log(next, action.receivedAt, shortTrackId(t.track_id), t.status.toUpperCase(),
            TRACK_TONE[t.status]);
        }
        case "node_status": {
          const n = event.data as NodeView;
          const previous = state.nodes[n.node_id]?.status;
          const next = { ...state, nodes: { ...state.nodes, [n.node_id]: n } };
          if (previous === n.status) return next;
          return log(next, action.receivedAt, n.node_id, n.status.toUpperCase(), NODE_TONE[n.status]);
        }
        case "observation": {
          const o = event.data as Observation;
          const withPulse = {
            ...state, pulses: [...state.pulses, { nodeId: o.source.id, at: action.receivedAt }],
          };
          const phase = o.event.phase.toUpperCase();
          const message = `DETECT ${phase} ${percent(o.detection.confidence)}`;
          return log(withPulse, action.receivedAt, o.source.id, message,
            o.event.phase === "start" ? "primary" : "none");
        }
        default:
          return state;
      }
    }
    case "connection":
      return { ...state, connection: action.connection };
    case "select":
      return { ...state, selectedTrackId: action.trackId };
    case "expirePulses": {
      const pulses = state.pulses.filter((p) => action.now - p.at <= PULSE_MS);
      return pulses.length === state.pulses.length ? state : { ...state, pulses };
    }
  }
}
