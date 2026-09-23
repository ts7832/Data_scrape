import type { LiveEvent, NodeView, Track } from "../api/types";

export type Connection = "connecting" | "live" | "reconnecting";
export interface Pulse { nodeId: string; at: number }

export interface State {
  nodes: Record<string, NodeView>;
  tracks: Record<string, Track>;
  trails: Record<string, [number, number][]>; // [lon, lat]
  pulses: Pulse[];
  connection: Connection;
  selectedTrackId: string | null;
}

export type Action =
  | { type: "snapshot"; nodes: NodeView[]; tracks: Track[] }
  | { type: "live"; event: LiveEvent; receivedAt: number }
  | { type: "connection"; connection: Connection }
  | { type: "select"; trackId: string | null }
  | { type: "expirePulses"; now: number };

export const TRAIL_MAX = 60;
export const PULSE_MS = 3000;

export const initialState: State = {
  nodes: {}, tracks: {}, trails: {}, pulses: [], connection: "connecting", selectedTrackId: null,
};

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
        case "track":
          return upsertTrack(state, event.data as Track);
        case "node_status": {
          const n = event.data as NodeView;
          return { ...state, nodes: { ...state.nodes, [n.node_id]: n } };
        }
        case "observation": {
          const nodeId = (event.data as { source: { id: string } }).source.id;
          return { ...state, pulses: [...state.pulses, { nodeId, at: action.receivedAt }] };
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
