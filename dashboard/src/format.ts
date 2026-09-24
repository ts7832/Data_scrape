/** Short, readable callsign for a track uuid, e.g. "T-3F2A". */
export function shortTrackId(trackId: string): string {
  return `T-${trackId.replace(/-/g, "").slice(0, 4).toUpperCase()}`;
}

const pad = (n: number) => String(n).padStart(2, "0");

/** HH:MM:SS in UTC for a millisecond timestamp. */
export function clockUtc(ms: number): string {
  const d = new Date(ms);
  return `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}:${pad(d.getUTCSeconds())}`;
}

/** HH:MM:SSZ (Zulu) for a millisecond timestamp, as shown in the UI. */
export const clockZ = (ms: number) => `${clockUtc(ms)}Z`;

/** HH:MM:SSZ for an ISO-8601 timestamp string. */
export const isoZ = (iso: string) => clockZ(Date.parse(iso));

export const percent = (x: number) => `${Math.round(x * 100)}%`;
