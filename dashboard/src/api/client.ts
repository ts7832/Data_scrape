import type { NodeView, Track, TrackDetail } from "./types";

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${path}: ${response.status}`);
  return (await response.json()) as T;
}

export async function fetchSnapshot(): Promise<{ nodes: NodeView[]; tracks: Track[] }> {
  const [nodes, tracks] = await Promise.all([getJson<NodeView[]>("/v1/nodes"), getJson<Track[]>("/v1/tracks")]);
  return { nodes, tracks };
}

export const fetchTrackDetail = (id: string) => getJson<TrackDetail>(`/v1/tracks/${id}`);
