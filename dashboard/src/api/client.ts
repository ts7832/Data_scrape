import type { ImpulseEvent, ImpulseEventDetail, NodeView, Track, TrackDetail } from "./types";

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${path}: ${response.status}`);
  return (await response.json()) as T;
}

export async function fetchSnapshot(): Promise<{ nodes: NodeView[]; tracks: Track[]; impulses: ImpulseEvent[] }> {
  const [nodes, tracks, impulses] = await Promise.all([
    getJson<NodeView[]>("/v1/nodes"), getJson<Track[]>("/v1/tracks"),
    getJson<ImpulseEvent[]>("/v1/impulse-events"),
  ]);
  return { nodes, tracks, impulses };
}

export const fetchTrackDetail = (id: string) => getJson<TrackDetail>(`/v1/tracks/${id}`);

export const fetchImpulseDetail = (id: string) => getJson<ImpulseEventDetail>(`/v1/impulse-events/${id}`);
