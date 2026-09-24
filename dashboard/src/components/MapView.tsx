import { useEffect, useRef } from "react";
import { Map as MapLibreMap, type GeoJSONSource, type MapLayerMouseEvent, setWorkerUrl } from "maplibre-gl";
import workerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url";
import "maplibre-gl/dist/maplibre-gl.css";
import type { Action, State } from "../state/reducer";
import { nodesToGeoJSON, pulsesToGeoJSON, trailsToGeoJSON, tracksToGeoJSON, uncertaintyToGeoJSON } from "../map/layers";
import { PALETTE } from "../theme";

// MapLibre GL v6 is ESM-only and locates its tile worker via
// `new URL("./maplibre-gl-worker.mjs", import.meta.url)` internally, which
// Vite doesn't resolve to a servable script (dev or prod) — the map then
// loads its style/tiles but never renders anything ("Worker failed to load").
// `?worker&url` (not plain `?url`) bundles the worker together with the
// sibling module it imports, giving it a real, servable URL up front.
setWorkerUrl(workerUrl);

// Dark slate-blue basemap that sits in Blueprint's dark-gray palette.
const STYLE_URL = "https://tiles.openfreemap.org/styles/fiord";
const HELSINKI: [number, number] = [24.9384, 60.1699];
const NODE_COLORS = ["match", ["get", "status"],
  "online", PALETTE.node, "stale", PALETTE.warning, PALETTE.muted];
const TRACK_COLORS = ["match", ["get", "status"],
  "confirmed", PALETTE.danger, "downgraded", PALETTE.muted, PALETTE.warning];
const NO_TRACK = "__none__";

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
    const map = new MapLibreMap({
      container: container.current, style: STYLE_URL, center: HELSINKI, zoom: 13,
      attributionControl: { compact: true },
    });
    mapRef.current = map;
    map.on("load", () => {
      for (const id of ["nodes", "tracks", "uncertainty", "trails", "pulses"]) {
        map.addSource(id, { type: "geojson", data: { type: "FeatureCollection", features: [] } });
      }
      map.addLayer({ id: "uncertainty", type: "fill", source: "uncertainty",
        paint: { "fill-color": TRACK_COLORS as never, "fill-opacity": 0.1 } });
      map.addLayer({ id: "uncertainty-edge", type: "line", source: "uncertainty",
        paint: { "line-color": TRACK_COLORS as never, "line-width": 1, "line-opacity": 0.7,
          "line-dasharray": [3, 3] } });
      map.addLayer({ id: "trails", type: "line", source: "trails",
        paint: { "line-color": TRACK_COLORS as never, "line-width": 2, "line-opacity": 0.8 } });
      map.addLayer({ id: "pulses", type: "circle", source: "pulses",
        paint: { "circle-radius": 16, "circle-color": PALETTE.primary, "circle-opacity": 0.25,
          "circle-stroke-width": 1, "circle-stroke-color": PALETTE.primary } });
      map.addLayer({ id: "nodes", type: "circle", source: "nodes",
        paint: { "circle-radius": 4, "circle-color": NODE_COLORS as never,
          "circle-stroke-width": 1, "circle-stroke-color": PALETTE.bg } });
      map.addLayer({ id: "track-selected", type: "circle", source: "tracks",
        filter: ["==", ["get", "id"], NO_TRACK],
        paint: { "circle-radius": 15, "circle-color": "rgba(0,0,0,0)",
          "circle-stroke-width": 1.5, "circle-stroke-color": PALETTE.text } });
      map.addLayer({ id: "tracks", type: "circle", source: "tracks",
        paint: { "circle-radius": 7, "circle-color": TRACK_COLORS as never,
          "circle-stroke-width": 2, "circle-stroke-color": PALETTE.bg } });
      map.addLayer({ id: "track-labels", type: "symbol", source: "tracks",
        layout: { "text-field": ["get", "callsign"], "text-font": ["Noto Sans Regular"],
          "text-size": 11, "text-offset": [0, 1.6], "text-letter-spacing": 0.1 },
        paint: { "text-color": PALETTE.text, "text-halo-color": PALETTE.bg, "text-halo-width": 1.5 } });
      map.on("click", "tracks", (e: MapLayerMouseEvent) => {
        const id = e.features?.[0]?.properties?.id;
        if (typeof id === "string") dispatch({ type: "select", trackId: id });
      });
      map.on("mouseenter", "tracks", () => { map.getCanvas().style.cursor = "pointer"; });
      map.on("mouseleave", "tracks", () => { map.getCanvas().style.cursor = ""; });
      loaded.current = true;
      for (const cb of onReady.current) cb();
      onReady.current = [];
    });
    return () => {
      // Reset with the map: under React StrictMode's dev double-mount the second
      // map must not inherit "loaded" or callbacks bound to the removed one.
      loaded.current = false;
      onReady.current = [];
      mapRef.current = null;
      map.remove();
    };
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
      map.setFilter("track-selected", ["==", ["get", "id"], state.selectedTrackId ?? NO_TRACK]);
    };
    if (loaded.current) update();
    else onReady.current.push(update);
  }, [state]);

  return <div ref={container} className="map" />;
}
