import { describe, expect, it } from "vitest";
import { circlePolygon, haversineM } from "./geo";
import { nodesToGeoJSON, tracksToGeoJSON } from "./layers";

describe("geo", () => {
  it("circle is closed and has the right radius", () => {
    const ring = circlePolygon(60.17, 24.94, 500, 32);
    expect(ring.length).toBe(33);
    expect(ring[0]).toEqual(ring[32]);
    for (const [lon, lat] of ring) expect(haversineM(60.17, 24.94, lat, lon)).toBeCloseTo(500, -1);
  });
});

describe("layers", () => {
  it("nodes and tracks become GeoJSON points with status", () => {
    const nodes = nodesToGeoJSON({
      n1: { node_id: "n1", location: { lat: 60.1, lon: 24.9, accuracy_m: 1 }, time_quality: "ntp", status: "stale" },
    });
    expect(nodes.features[0].geometry.coordinates).toEqual([24.9, 60.1]);
    expect(nodes.features[0].properties.status).toBe("stale");
    const tracks = tracksToGeoJSON({});
    expect(tracks.features).toEqual([]);
  });
});
