const R = 6_371_000;
const rad = (d: number) => (d * Math.PI) / 180;
const deg = (r: number) => (r * 180) / Math.PI;

export function haversineM(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const a = Math.sin(rad(lat2 - lat1) / 2) ** 2 +
    Math.cos(rad(lat1)) * Math.cos(rad(lat2)) * Math.sin(rad(lon2 - lon1) / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(a));
}

/** Closed ring of [lon, lat] points approximating a circle on the ground. */
export function circlePolygon(lat: number, lon: number, radiusM: number, steps = 48): [number, number][] {
  const ring: [number, number][] = [];
  const d = radiusM / R;
  for (let i = 0; i < steps; i++) {
    const b = (2 * Math.PI * i) / steps;
    const lat2 = Math.asin(Math.sin(rad(lat)) * Math.cos(d) + Math.cos(rad(lat)) * Math.sin(d) * Math.cos(b));
    const lon2 = rad(lon) + Math.atan2(
      Math.sin(b) * Math.sin(d) * Math.cos(rad(lat)),
      Math.cos(d) - Math.sin(rad(lat)) * Math.sin(lat2),
    );
    ring.push([deg(lon2), deg(lat2)]);
  }
  ring.push(ring[0]);
  return ring;
}
