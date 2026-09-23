import { useEffect, useRef } from "react";
import { Map as MapLibreMap, type GeoJSONSource, type MapLayerMouseEvent } from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import type { Action, State } from "../state/reducer";
import { nodesToGeoJSON, pulsesToGeoJSON, trailsToGeoJSON, tracksToGeoJSON, uncertaintyToGeoJSON } from "../map/layers";

const STYLE_URL = "https://tiles.openfreemap.org/styles/liberty";
const HELSINKI: [number, number] = [24.9384, 60.1699];
const NODE_COLORS = ["match", ["get", "status"], "online", "#2e7d32", "stale", "#f9a825", "#9e9e9e"];
const TRACK_COLORS = ["match", ["get", "status"], "confirmed", "#d32f2f", "downgraded", "#757575", "#ffa000"];

export function MapView({ state, dispatch }: { state: State; dispatch: (a: Action) => void }) {
  const container = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MapLibreMap | null>(null);
  const loaded = useRef(false);
  // Deferred callbacks waiting for the map's "load" event, run once and cleared.
  // Plain refs instead of a custom maplibre event: maplibre-gl's `.on`/`.once`/`.fire`
  // are now typed to a closed set of known event names, so an app-defined event name
  // doesn't type-check there.
  const onReady = useRef<(() => void)[]>([]);

  useEffect(() => {
    if (!container.current) return;
    const map = new MapLibreMap({ container: container.current, style: STYLE_URL, center: HELSINKI, zoom: 13 });
    mapRef.current = map;
    map.on("load", () => {
      for (const id of ["nodes", "tracks", "uncertainty", "trails", "pulses"]) {
        map.addSource(id, { type: "geojson", data: { type: "FeatureCollection", features: [] } });
      }
      map.addLayer({ id: "uncertainty", type: "fill", source: "uncertainty",
        paint: { "fill-color": TRACK_COLORS as never, "fill-opacity": 0.15 } });
      map.addLayer({ id: "trails", type: "line", source: "trails",
        paint: { "line-color": TRACK_COLORS as never, "line-width": 2 } });
      map.addLayer({ id: "pulses", type: "circle", source: "pulses",
        paint: { "circle-radius": 18, "circle-color": "#1e88e5", "circle-opacity": 0.35 } });
      map.addLayer({ id: "nodes", type: "circle", source: "nodes",
        paint: { "circle-radius": 6, "circle-color": NODE_COLORS as never, "circle-stroke-width": 1, "circle-stroke-color": "#fff" } });
      map.addLayer({ id: "tracks", type: "circle", source: "tracks",
        paint: { "circle-radius": 9, "circle-color": TRACK_COLORS as never, "circle-stroke-width": 2, "circle-stroke-color": "#fff" } });
      map.on("click", "tracks", (e: MapLayerMouseEvent) => {
        const id = e.features?.[0]?.properties?.id;
        if (typeof id === "string") dispatch({ type: "select", trackId: id });
      });
      loaded.current = true;
      for (const cb of onReady.current) cb();
      onReady.current = [];
    });
    return () => map.remove();
  }, [dispatch]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    const update = () => {
      const set = (id: string, data: unknown) => (map.getSource(id) as GeoJSONSource | undefined)?.setData(data as never);
      set("nodes", nodesToGeoJSON(state.nodes));
      set("tracks", tracksToGeoJSON(state.tracks));
      set("uncertainty", uncertaintyToGeoJSON(state.tracks));
      set("trails", trailsToGeoJSON(state.trails, state.tracks));
      set("pulses", pulsesToGeoJSON(state.pulses, state.nodes));
    };
    if (loaded.current) update();
    else onReady.current.push(update);
  }, [state]);

  return <div ref={container} className="map" />;
}
