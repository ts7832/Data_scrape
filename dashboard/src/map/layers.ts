import type { NodeView, Track } from "../api/types";
import type { Pulse } from "../state/reducer";
import { circlePolygon } from "./geo";

type Props = Record<string, string | number>;
export interface PointFeature { type: "Feature"; geometry: { type: "Point"; coordinates: [number, number] }; properties: Props }
export interface Collection<F> { type: "FeatureCollection"; features: F[] }

const point = (lon: number, lat: number, properties: Props): PointFeature => ({
  type: "Feature", geometry: { type: "Point", coordinates: [lon, lat] }, properties,
});

export function nodesToGeoJSON(nodes: Record<string, NodeView>): Collection<PointFeature> {
  return {
    type: "FeatureCollection",
    features: Object.values(nodes).map((n) =>
      point(n.location.lon, n.location.lat, { id: n.node_id, status: n.status })),
  };
}

export function tracksToGeoJSON(tracks: Record<string, Track>): Collection<PointFeature> {
  return {
    type: "FeatureCollection",
    features: Object.values(tracks).map((t) =>
      point(t.position.lon, t.position.lat, { id: t.track_id, status: t.status, confidence: t.confidence })),
  };
}

export function uncertaintyToGeoJSON(tracks: Record<string, Track>) {
  return {
    type: "FeatureCollection" as const,
    features: Object.values(tracks).map((t) => ({
      type: "Feature" as const,
      geometry: { type: "Polygon" as const, coordinates: [circlePolygon(t.position.lat, t.position.lon, t.uncertainty_m)] },
      properties: { id: t.track_id, status: t.status },
    })),
  };
}

export function trailsToGeoJSON(trails: Record<string, [number, number][]>, tracks: Record<string, Track>) {
  return {
    type: "FeatureCollection" as const,
    features: Object.entries(trails)
      .filter(([id, coords]) => id in tracks && coords.length > 1)
      .map(([id, coords]) => ({
        type: "Feature" as const,
        geometry: { type: "LineString" as const, coordinates: coords },
        properties: { id, status: tracks[id].status },
      })),
  };
}

export function pulsesToGeoJSON(pulses: Pulse[], nodes: Record<string, NodeView>): Collection<PointFeature> {
  return {
    type: "FeatureCollection",
    features: pulses
      .filter((p) => p.nodeId in nodes)
      .map((p) => point(nodes[p.nodeId].location.lon, nodes[p.nodeId].location.lat, { id: p.nodeId })),
  };
}
