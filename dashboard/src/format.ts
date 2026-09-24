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

export const percent = (x: number) => `${Math.round(x * 100)}%`;
